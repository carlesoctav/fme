import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np
from io import open
from datasets import load_dataset
from jax.sharding import Mesh, PartitionSpec
from transformers import AutoTokenizer
from flax import struct
import jax.tree_util as jtu
import time
import logging
from tqdm.auto import tqdm

from src._logger import TrackioLogger
from src.data._training import make_dataloader
from src.data.masked_language_modeling import (
    masked_language_modeling_transforms,
)
from src.losses.cross_entropy import softmax_cross_entropy_with_integer_labels
from src._logger import setup_logger
from flaxformer.architectures.bert.bert import BertEncoder
from flaxformer.architectures.bert.heads import MLMHead

LOGGER = logging.getLogger(__name__)


DATASET_NAME = "carlesoctav/skripsi_UI_membership_30K"
DATASET_SPLIT = "train"
DATASET_SUBSET = None
COLUMN_NAME = "id_title"
MAX_LENGTH = 512
BATCH_SIZE = 64
NUM_STEPS = 100
LEARNING_RATE = 5e-5
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.01
SEED = 42
NUM_WORKERS = 4
WORKER_BUFFER_SIZE = 2
MLM_PROBABILITY = 0.15
MESH_AXIS_NAMES = ("dp",)


def _get_position_ids(batch, seq_length):
    if batch.position_ids is not None:
        return batch.position_ids
    batch_size = batch.input_ids.shape[0]
    return jnp.broadcast_to(jnp.arange(seq_length)[None, :], (batch_size, seq_length))


@struct.dataclass
class TrainState:
    params: dict
    opt_state: dict
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    encoder: BertEncoder = struct.field(pytree_node=False)
    mlm_head: MLMHead = struct.field(pytree_node=False)


def loss_function_flax(params, encoder, mlm_head, batch, key):
    encoded_inputs = encoder.apply(
        params['encoder'],
        token_ids=batch.input_ids,
        position_ids=_get_position_ids(batch, MAX_LENGTH),
        segment_ids=batch.token_type_ids,
        input_mask=batch.attention_mask,
        enable_dropout=False,
        rngs={'dropout': key},
    )
    
    logits = mlm_head.apply(
        params['mlm_head'],
        encoded_inputs,
        masked_positions=None,
    )
    
    labels = batch.labels
    valid_mask = labels != -100
    
    loss_per_token = softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=jnp.where(valid_mask, labels, 0),
        where=valid_mask,
    )
    
    total_loss = jnp.sum(loss_per_token)
    num_valid_tokens = jnp.sum(valid_mask)
    
    accuracy = jnp.sum(
        (jnp.argmax(logits, axis=-1) == labels) & valid_mask
    )
    
    aux = {
        "loss": (total_loss, num_valid_tokens),
        "acc": (accuracy, num_valid_tokens),
        "total_token": num_valid_tokens
    }
    
    return total_loss, aux


def make_flax_train_step():
    grad_fn = jax.value_and_grad(loss_function_flax, has_aux=True)
    
    def train_step(train_state, optimizer_placeholder, batch, key):
        (total_loss, aux), grads = grad_fn(
            train_state.params, 
            train_state.encoder, 
            train_state.mlm_head, 
            batch, 
            key
        )
        
        updates, new_opt_state = train_state.tx.update(
            grads, train_state.opt_state, train_state.params
        )
        new_params = optax.apply_updates(train_state.params, updates)
        
        new_train_state = train_state.replace(
            params=new_params,
            opt_state=new_opt_state
        )
        
        return new_train_state, optimizer_placeholder, aux
    
    return jax.jit(train_step)


def benchmark_loop_flax(
    train_state,
    optimizer_placeholder,
    train_step_fn,
    train_loader,
    logger,
    num_steps=100,
    trace_steps=None,
    trace_dir="./benchmark_traces",
    key=None,
):
    if logger is None:
        raise ValueError("logger is required")
    
    step_idx = -1
    train_step_times = []
    next_batch_times = []
    
    try:
        train_iterator = iter(train_loader)
    except TypeError as e:
        raise RuntimeError("train_loader is not iterable") from e
    
    progress_bar = tqdm(
        total=num_steps + 1,
        desc="Benchmarking Flax",
        disable=jax.process_index() != 0,
        leave=True,
    )
    
    try:
        while step_idx < num_steps:
            batch_start = time.perf_counter()
            try:
                batch = next(train_iterator)
            except StopIteration:
                LOGGER.info("Train data loader exhausted, ending benchmark loop.")
                break
            batch_end = time.perf_counter()
            
            step_idx += 1
            
            if (
                trace_steps is not None
                and step_idx == trace_steps[0]
                and jax.process_index() == 0
            ):
                from pathlib import Path
                trace_path = Path(trace_dir)
                trace_path.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Starting JAX profiler trace at step {step_idx}")
                jax.profiler.start_trace(str(trace_path))
            
            key, step_key = jax.random.split(key, 2) if key is not None else (key, None)
            
            if step_idx == 0 and jax.process_index() == 0:
                LOGGER.info("Running step 0 (compilation step)...")
            
            step_start = time.monotonic()
            with jax.profiler.StepTraceAnnotation("train_step", step=step_idx):
                train_state, optimizer_placeholder, aux = train_step_fn(
                    train_state, optimizer_placeholder, batch, key=step_key
                )
            
            jtu.tree_map(
                lambda x: x.block_until_ready()
                if hasattr(x, "block_until_ready")
                else x,
                train_state,
            )
            step_end = time.monotonic()

            if (
                trace_steps is not None
                and step_idx == trace_steps[1]
                and jax.process_index() == 0
            ):
                LOGGER.info(f"Stopping JAX profiler trace at step {step_idx}")
                jax.profiler.stop_trace()
            
            train_step_times.append(step_end - step_start)
            next_batch_times.append(batch_end - batch_start)
            
            if step_idx == 0 and jax.process_index() == 0:
                LOGGER.info(f"Step 0 (compilation) took {step_end - step_start:.4f}s")
            
            progress_bar.update()
        
        progress_bar.close()
        
        train_step_times = np.array(train_step_times)
        next_batch_times = np.array(next_batch_times)
        
        if len(train_step_times) == 0:
            LOGGER.warning("No timing data collected (all steps were skipped)")
            return train_state, optimizer_placeholder, {}
        
        for i, t in enumerate(train_step_times):
            logger.log({f"benchmark/train_step_time": t}, step=i)
        for i, t in enumerate(next_batch_times):
            logger.log({f"benchmark/next_batch_time": t}, step=i)
        
        for i, step_time in enumerate(train_step_times):
            batches_per_sec = 1.0 / step_time if step_time > 0 else 0
            logger.log({f"benchmark/batches_per_sec": batches_per_sec}, step=i)
        
        if jax.process_index() == 0:
            LOGGER.info("=" * 30)
            LOGGER.info(" Benchmark Results (Flax) ".center(30, "="))
            LOGGER.info("=" * 30)
            LOGGER.info(f"Train Step Time (avg): {train_step_times.mean():.4f}s")
            LOGGER.info(f"Train Step Time (median): {np.median(train_step_times):.4f}s")
            LOGGER.info(f"Train Step Time (std): {train_step_times.std():.4f}s")
            LOGGER.info(f"Next Batch Time (avg): {next_batch_times.mean():.4f}s")
            LOGGER.info(f"Batches/sec (avg): {1.0 / train_step_times.mean():.2f}")
            LOGGER.info("=" * 30)
        
        stats = {
            "train_step_time_mean": float(train_step_times.mean()),
            "train_step_time_median": float(np.median(train_step_times)),
            "train_step_time_std": float(train_step_times.std()),
            "next_batch_time_mean": float(next_batch_times.mean()),
            "batches_per_sec": float(1.0 / train_step_times.mean()),
        }
        
    except Exception:
        LOGGER.error("Exception during benchmark loop", exc_info=True)
        raise
    finally:
        LOGGER.info("Benchmark loop ended")
    
    return train_state, optimizer_placeholder, stats


def main():
    LOGGER.info("Starting Flax BERT Benchmark")
    logger = TrackioLogger(project="benchmark", name="bert-flax")
    
    key = jr.PRNGKey(SEED)
    key, model_key, mlm_key = jr.split(key, 3)
    
    devices = jax.devices()
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    encoder = BertEncoder(
        vocab_size=30522,
        hidden_size=768,
        intermediate_dim=3072,
        max_length=512,
        num_segments=2,
        num_hidden_layers=12,
        num_attention_heads=12,
        dropout_rate=0.1,
    )
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=NUM_STEPS,
        end_value=0.0,
    )
    
    grad_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=WEIGHT_DECAY),
    )
    
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_SUBSET,
        split=DATASET_SPLIT,
        streaming=False,
    )
    
    operations, batch_class = masked_language_modeling_transforms(
        dataset_type="huggingface",
        column=COLUMN_NAME,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        mlm_probability=MLM_PROBABILITY,
        packing=False,
    )
    
    train_loader = make_dataloader(
        datasets=[dataset],
        operations=operations,
        global_batch_size=BATCH_SIZE,
        pspec=PartitionSpec("dp"),
        mesh=mesh,
        num_epochs=None,
        shuffle=True,
        seed=SEED,
        worker_count=NUM_WORKERS,
        worker_buffer_size=WORKER_BUFFER_SIZE,
        drop_remainder=True,
        batch_class=batch_class,
    )
    
    batch = next(iter(train_loader))
    
    dummy_input = {
        'token_ids': batch.input_ids[:1],
        'position_ids': _get_position_ids(batch, MAX_LENGTH)[:1],
        'segment_ids': batch.token_type_ids[:1],
        'input_mask': batch.attention_mask[:1],
    }

    encoder_params = encoder.init(
        model_key,
        **dummy_input,
        enable_dropout=False,
    )
    
    mlm_head = MLMHead(
        encoder=encoder,
        hidden_size=768,
        vocab_size=30522,
        dropout_rate=0.1,
    )
    
    dummy_encoded = jnp.zeros((1, MAX_LENGTH, 768))
    mlm_params = mlm_head.init(
        mlm_key,
        dummy_encoded,
        masked_positions=None,
    )
    
    params = {
        'encoder': encoder_params,
        'mlm_head': mlm_params,
    }

    sharding = jax.NamedSharding(mesh, jax.P(None))
    start_dp = time.monotonic()
    params = jtu.tree_map(lambda x: jax.device_put(x, sharding), params)
    jax.block_until_ready(params)
    diff_dp = time.monotonic() - start_dp
    print(f"DEBUGPRINT[12]: train_with_flax.py:254: diff_dp={diff_dp}")
    
    opt_state = grad_tx.init(params)
    
    train_state = TrainState(
        params=params,
        opt_state=opt_state,
        tx=grad_tx,
        encoder=encoder,
        mlm_head=mlm_head,
    )
    
    optimizer_placeholder = None
    
    train_step_fn = make_flax_train_step()
    
    LOGGER.info("Running benchmark loop...")
    train_state, optimizer_placeholder, stats = benchmark_loop_flax(
        train_state=train_state,
        optimizer_placeholder=optimizer_placeholder,
        train_step_fn=train_step_fn,
        train_loader=train_loader,
        logger=logger,
        num_steps=NUM_STEPS,
        trace_steps=(1, 5),
        trace_dir="./flax_benchmark_traces",
        key=key,
    )
    
    LOGGER.info("Benchmark completed!")
    LOGGER.info(f"Stats: {stats}")


if __name__ == "__main__":
    setup_logger()
    main()
