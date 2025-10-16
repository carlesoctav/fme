from dataclasses import dataclass, replace
from pathlib import Path
import time

from tqdm.auto import tqdm
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import optax
import numpy as np
from datasets import load_dataset
from jax.sharding import Mesh, PartitionSpec
from transformers import AutoTokenizer, BertConfig

from src._training import make_module_opt
from src._logger import setup_logger
from src.data._training import make_dataloader
from src.data.masked_language_modeling import (
    masked_language_modeling_transforms,
)
from src.losses.cross_entropy import softmax_cross_entropy_with_integer_labels
from src.models.bert import BertForMaskedLM
from src._darray import DArray


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
GRADIENT_ACCUMULATION_STEPS = 1
SEED = 42
NUM_WORKERS = 0
WORKER_BUFFER_SIZE = 2
MLM_PROBABILITY = 0.15
MESH_SHAPE = (4,)
MESH_AXIS_NAMES = ("dp",)


def _is_param_leaf(x):
    return isinstance(x, DArray)


def _make_template(module):
    return jtu.tree_map(
        lambda leaf: replace(leaf, value=None) if isinstance(leaf, DArray) else leaf,
        module,
        is_leaf=_is_param_leaf,
    )


def _params_to_module(template, params):
    def combine(temp_leaf, param_leaf):
        if isinstance(temp_leaf, DArray):
            return replace(temp_leaf, value=param_leaf)
        return param_leaf

    return jtu.tree_map(
        combine,
        template,
        params,
        is_leaf=_is_param_leaf,
    )


class ArrayTrainState(eqx.Module):
    params: any
    opt_state: any


def array_benchmark_loop(
    train_state: ArrayTrainState,
    train_step_fn,
    train_loader,
    *,
    trace_dir: str,
    trace_steps: tuple[int, int] | None,
    key: jr.PRNGKey,
    num_steps: int,
):
    step_idx = -1
    train_step_times: list[float] = []
    next_batch_times: list[float] = []
    compile_step_time: float | None = None

    try:
        train_iterator = iter(train_loader)
    except TypeError as e:
        raise RuntimeError("train_loader is not iterable") from e

    progress_bar = tqdm(
        total=num_steps + 1,
        desc="Benchmarking",
        disable=jax.process_index() != 0,
        leave=True,
    )

    while step_idx < num_steps:
        batch_start = time.perf_counter()
        try:
            batch = next(train_iterator)
        except StopIteration:
            break
        batch_end = time.perf_counter()

        step_idx += 1

        if (
            trace_steps is not None
            and step_idx == trace_steps[0]
            and jax.process_index() == 0
        ):
            trace_path = Path(trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)
            jax.profiler.start_trace(str(trace_path))

        key, step_key = jr.split(key, 2) if key is not None else (key, None)

        step_start = time.monotonic()
        with jax.profiler.StepTraceAnnotation("train_step", step=step_idx):
            train_state, aux = train_step_fn(train_state, batch, step_key)

        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            train_state,
        )

        step_end = time.monotonic()

        if (
            trace_steps is not None
            and step_idx == trace_steps[1]
            and jax.process_index() == 0
        ):
            jax.profiler.stop_trace()

        if step_idx == 0:
            compile_step_time = step_end - step_start
        else:
            train_step_times.append(step_end - step_start)
            next_batch_times.append(batch_end - batch_start)

        progress_bar.update()

        if step_idx >= num_steps:
            break

    progress_bar.close()

    train_step_times_np = np.array(train_step_times)
    next_batch_times_np = np.array(next_batch_times)

    stats = {
        "train_step_time_mean": float(train_step_times_np.mean())
        if train_step_times_np.size
        else 0.0,
        "train_step_time_median": float(np.median(train_step_times_np))
        if train_step_times_np.size
        else 0.0,
        "train_step_time_std": float(train_step_times_np.std())
        if train_step_times_np.size
        else 0.0,
        "next_batch_time_mean": float(next_batch_times_np.mean())
        if next_batch_times_np.size
        else 0.0,
        "batches_per_sec": float(1.0 / train_step_times_np.mean())
        if train_step_times_np.size
        else 0.0,
        "train_step_times": train_step_times_np.tolist(),
        "next_batch_times": next_batch_times_np.tolist(),
    }

    if compile_step_time is not None:
        stats["compile_time"] = float(compile_step_time)

    return stats


def _get_position_ids(batch, seq_length):
    if batch.position_ids is not None:
        return batch.position_ids
    batch_size = batch.input_ids.shape[0]
    return jnp.broadcast_to(jnp.arange(seq_length)[None, :], (batch_size, seq_length))


def loss_function(module, batch, key):
    logits = module(
        input_ids=batch.input_ids,
        position_ids=_get_position_ids(batch, MAX_LENGTH),
        token_type_ids=batch.token_type_ids,
        attention_mask=batch.attention_mask,
        segment_ids=batch.segment_ids,
        key=key,
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

    accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == labels) & valid_mask)

    aux = {
        "loss": (total_loss, num_valid_tokens),
        "acc": (accuracy, num_valid_tokens),
        "total_token": num_valid_tokens,
    }

    return total_loss, aux


def main():
    key = jr.PRNGKey(SEED)
    key, model_key = jr.split(key)

    devices = jax.devices()
    mesh = Mesh(devices, MESH_AXIS_NAMES)

    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        _attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model = BertForMaskedLM(config, key=model_key)

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

    module_sharded, _ = make_module_opt(
        model,
        grad_tx,
        mesh=mesh,
        key=model_key,
    )

    template = _make_template(module_sharded)
    module_arrays_init = jtu.tree_map(
        lambda leaf: leaf.value if isinstance(leaf, DArray) else leaf,
        module_sharded,
        is_leaf=_is_param_leaf,
    )

    params, static_arrays = eqx.partition(module_arrays_init, eqx.is_array)
    opt_state = grad_tx.init(params)
    train_state = ArrayTrainState(params=params, opt_state=opt_state)

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

    def loss_fn_params(params, batch, key):
        module_arrays = eqx.combine(static_arrays, params)
        module_inst = _params_to_module(template, module_arrays)
        return loss_function(module_inst, batch, key)

    grad_fn = jax.value_and_grad(loss_fn_params, has_aux=True, allow_int=True)

    @jax.jit
    def train_step(state: ArrayTrainState, batch, key):
        (loss, aux), grads = grad_fn(state.params, batch, key)
        updates, new_opt_state = grad_tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return ArrayTrainState(new_params, new_opt_state), aux

    stats = array_benchmark_loop(
        train_state=train_state,
        train_step_fn=train_step,
        train_loader=train_loader,
        trace_dir="./mine_array_sdpa",
        trace_steps=(0, NUM_STEPS),
        key=key,
        num_steps=NUM_STEPS,
    )

    print(f"DEBUGPRINT[6]: benchmark.py:168: stats={stats}")


if __name__ == "__main__":
    setup_logger(log_file="./benchmark.log")
    main()
