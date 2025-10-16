import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from datasets import load_dataset
from jax.sharding import Mesh, PartitionSpec
from transformers import AutoTokenizer, BertConfig

from src._training import make_module_opt, make_train_step, Optimizer
from src.data._training import make_dataloader
from src.data.masked_language_modeling import (
    masked_language_modeling_transforms,
)
from src.losses.cross_entropy import softmax_cross_entropy_with_integer_labels
from src.models.bert import BertForMaskedLM
from src._logger import setup_logger
from src import DArray
import jax.tree_util as jtu
import time
import numpy as np
from dataclasses import replace


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


def loss_function(model, optimizer, batch, key):
    logits = model(
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
    
    accuracy = jnp.sum(
        (jnp.argmax(logits, axis=-1) == labels) & valid_mask
    )
    
    aux = {
        "loss": (total_loss, num_valid_tokens),
        "acc": (accuracy, num_valid_tokens),
        "total_token": num_valid_tokens
    }
    
    return total_loss, aux


def unbox_params(module):
    is_darray = lambda x: isinstance(x, DArray)

    def unbox(leaf):
        if isinstance(leaf, DArray):
            return leaf.value
        return leaf

    return jtu.tree_map(unbox, module, is_leaf=is_darray)


def test_filter_jit(model, optimizer, train_loader, key):
    print("\n=== Testing eqx.filter_jit (make_train_step) ===")
    
    train_step = make_train_step(loss_function=loss_function, jit=True)
    
    step_times = []
    train_iterator = iter(train_loader)
    current_model = model
    current_opt = optimizer
    
    for step in range(NUM_STEPS):
        batch = next(train_iterator)
        key, step_key = jr.split(key)
        
        if step == 1:
            jax.profiler.start_trace("./research/bert-mlm/trace_filter_jit_make_train_step")
        
        start = time.monotonic()
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            current_model, current_opt, aux = train_step(current_model, current_opt, batch, key=step_key)
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, current_model)
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, current_opt)
        end = time.monotonic()
        
        if step == 10:
            jax.profiler.stop_trace()
        
        if step > 0:
            step_times.append(end - start)
        else:
            print(f"Compilation time: {end - start:.3f}s")
    
    step_times = np.array(step_times)
    print(f"Mean step time: {step_times.mean():.4f}s")
    print(f"Median step time: {np.median(step_times):.4f}s")
    print(f"Std step time: {step_times.std():.4f}s")
    print(f"Batches/sec: {1.0/step_times.mean():.2f}")
    
    return {"mean": step_times.mean(), "median": np.median(step_times)}


def test_manual_jit(model, optimizer, train_loader, key):
    print("\n=== Testing manual jax.jit with partition/combine ===")
    
    params_init, static_parts = eqx.partition(model, eqx.is_array)
    
    def loss_fn_params(params, batch, key):
        module_inst = eqx.combine(params, static_parts)
        return loss_function(module_inst, optimizer, batch, key)
    
    grad_fn = jax.value_and_grad(loss_fn_params, has_aux=True, allow_int=True)
    
    @jax.jit
    def train_step(params, opt_state, batch, key):
        with jax.profiler.StepTraceAnnotation("gradient_compute"):
            (loss, aux), grads = grad_fn(params, batch, key)
        
        with jax.profiler.StepTraceAnnotation("weight_update"):
            updates, new_opt_state = optimizer.tx.update(grads, opt_state, params)
            new_params = eqx.apply_updates(params, updates)
        
        return new_params, new_opt_state, aux
    
    step_times = []
    train_iterator = iter(train_loader)
    current_params = params_init
    opt_state = optimizer.opt_state
    
    for step in range(NUM_STEPS):
        batch = next(train_iterator)
        key, step_key = jr.split(key)
        
        if step == 1:
            jax.profiler.start_trace("./research/bert-mlm/trace_manual_jit_make_train_step")
        
        start = time.monotonic()
        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            current_params, opt_state, aux = train_step(current_params, opt_state, batch, step_key)
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, current_params)
            jtu.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, opt_state)
        end = time.monotonic()
        
        if step == 10:
            jax.profiler.stop_trace()
        
        if step > 0:
            step_times.append(end - start)
        else:
            print(f"Compilation time: {end - start:.3f}s")
    
    step_times = np.array(step_times)
    print(f"Mean step time: {step_times.mean():.4f}s")
    print(f"Median step time: {np.median(step_times):.4f}s")
    print(f"Std step time: {step_times.std():.4f}s")
    print(f"Batches/sec: {1.0/step_times.mean():.2f}")
    
    return {"mean": step_times.mean(), "median": np.median(step_times)}


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
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        _attn_implementation="sdpa",
    )
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    model = BertForMaskedLM(config, key=key)
    
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
    
    model = unbox_params(model)
    model, optimizer = make_module_opt(
        model,
        grad_tx,
        mesh=mesh,
        key=model_key,
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
    
    filter_jit_stats = test_filter_jit(model, optimizer, train_loader, key)
    
    key = jr.PRNGKey(SEED)
    model = BertForMaskedLM(config, key=key)
    model = unbox_params(model)
    model, optimizer = make_module_opt(model, grad_tx, mesh=mesh, key=model_key)
    
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_SUBSET,
        split=DATASET_SPLIT,
        streaming=False,
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
    
    manual_jit_stats = test_manual_jit(model, optimizer, train_loader, key)
    
    print("\n=== Summary ===")
    print(f"filter_jit: {filter_jit_stats['mean']:.4f}s/step")
    print(f"manual jit: {manual_jit_stats['mean']:.4f}s/step")
    print(f"Speedup: {filter_jit_stats['mean']/manual_jit_stats['mean']:.2f}x")


if __name__ == "__main__":
    setup_logger(log_file="./research/bert-mlm/benchmark_make_train_step.log")
    main()
