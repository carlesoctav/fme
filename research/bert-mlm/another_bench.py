import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from datasets import load_dataset
from jax.sharding import Mesh, PartitionSpec
from transformers import AutoTokenizer, BertConfig

from src._training import make_module_opt, make_train_step, benchmark_loop
from src._logger import TrackioLogger
from src.data._training import make_dataloader
from src.data.masked_language_modeling import (
    masked_language_modeling_transforms,
)
from src.losses.cross_entropy import softmax_cross_entropy_with_integer_labels
from src.models.bert import BertForMaskedLM
from src._logger import setup_logger
from src import DArray
import logging
import jax.tree_util as jtu


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
NUM_WORKERS = 4
WORKER_BUFFER_SIZE = 2
MLM_PROBABILITY = 0.15
MESH_SHAPE = (4,)
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

    return jtu.tree_map(unbox, module, is_leaf = is_darray)


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
    
    # model = BertForMaskedLM(config, key=model_key)
    model =BertForMaskedLM(config, key = key)
    
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


    print(f"DEBUGPRINT[27]: another_bench.py:144: model={model}")
    
    # Extract static parts once before creating train step (captured in closure)
    params_init, static_parts = eqx.partition(model, eqx.is_array)
    
    def loss_fn_params(params, batch, key):
        # static_parts is captured from closure, not traced!
        module_inst = eqx.combine(params, static_parts)
        return loss_function(module_inst, optimizer, batch, key)
    
    grad_fn = jax.value_and_grad(loss_fn_params, has_aux=True, allow_int=True)
    
    def train_step_fn(module, opt, batch, key):
        # Extract params from current module (only arrays)
        params = eqx.filter(module, eqx.is_array)
        
        # Compute gradients
        (_, aux), grads = grad_fn(params, batch, key)
        
        # Update params
        updates, new_opt_state = opt.tx.update(grads, opt.opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        
        # Reconstruct module using static_parts from closure
        new_module = eqx.combine(new_params, static_parts)
        from dataclasses import replace
        new_opt = replace(opt, opt_state=new_opt_state)
        
        return new_module, new_opt, aux or {}
    
    # JIT compile the train step
    train_step_fn = jax.jit(train_step_fn)


    
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
    
    logger = TrackioLogger(project="benchmark", name="bert-do-1")
    
    module, optimizer, stats = benchmark_loop(
        model,
        optimizer,
        train_step_fn,
        train_loader,
        logger,
        num_steps=NUM_STEPS,
        trace_steps=(1, 101),
        key=key,
    )
    print(f"DEBUGPRINT[6]: benchmark.py:168: stats={stats}")


if __name__ == "__main__":
    setup_logger(log_file = "./benchmark.log")
    main()
