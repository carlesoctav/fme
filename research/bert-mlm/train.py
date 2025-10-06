import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from datasets import load_dataset
from jax.sharding import Mesh
from transformers import AutoTokenizer, BertConfig

from src import make_module_opt, make_train_step, SufficientMetric, train_loop
from src._logger import TrackioLogger
from src.callbacks import LearningRateMonitor, ModelCheckpoint
from src.data._training import make_dataloader
from src.data.masked_language_modeling import (
    MLMBatch,
    masked_language_modeling_transforms,
)
from src.losses.cross_entropy import softmax_cross_entropy_with_integer_labels
from src.models.bert import BertForMaskedLM


DATASET_NAME = "carlesoctav/skripsi_UI_membership_30K"
DATASET_SPLIT = "train"
DATASET_SUBSET = None
COLUMN_NAME = "id_title"
MAX_LENGTH = 512
BATCH_SIZE = 64
NUM_STEPS = 10000
LEARNING_RATE = 5e-5
WARMUP_STEPS = 1000
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION_STEPS = 1
SEED = 42
NUM_WORKERS = 4
WORKER_BUFFER_SIZE = 2
MLM_PROBABILITY = 0.15
LOG_EVERY_N_STEPS = 10
EVAL_INTERVAL = 500
SAVE_INTERVAL_STEPS = 100
MESH_SHAPE = (4,)
MESH_AXIS_NAMES = ("dp",)


def _get_position_ids(batch, seq_length):
    if batch.position_ids is not None:
        return batch.position_ids
    batch_size = batch.input_ids.shape[0]
    return jnp.broadcast_to(jnp.arange(seq_length)[None, :], (batch_size, seq_length))


def loss_function(model, optimizer, batch, key):
    batch_size = batch.input_ids.shape[0]
    
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
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        _attn_implementation="eager",
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
    
    model, optimizer = make_module_opt(
        model,
        grad_tx,
        mesh=mesh,
        key=model_key,
    )
    
    train_step_fn = make_train_step(
        loss_function=loss_function,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
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
        axis_names=MESH_AXIS_NAMES,
        mesh=mesh,
        num_epochs=None,
        shuffle=True,
        seed=SEED,
        worker_count=NUM_WORKERS,
        worker_buffer_size=WORKER_BUFFER_SIZE,
        use_thread_prefetch = True,
        drop_remainder=True,
        batch_class=batch_class,
    )
    
    logger = TrackioLogger(project="bert-mlm-fineweb")
    
    learning_rate_monitor = LearningRateMonitor(
        log_every_n_step=LOG_EVERY_N_STEPS,
        schedule_fn=schedule,
    )
    
    train_metric = SufficientMetric(name="train", log_every_n_step=LOG_EVERY_N_STEPS)
    
    modelcheckpoint = ModelCheckpoint(
        f"gs://carles-git-good/bert-mlm-fineweb",
        save_interval_steps=SAVE_INTERVAL_STEPS,
    )
    
    train_loop(
        model,
        optimizer,
        train_step_fn,
        train_loader,
        logger,
        train_metric,
        num_train_steps=NUM_STEPS,
        callbacks=[learning_rate_monitor],  # removed modelcheckpoint temporarily
        key=key,
    )


if __name__ == "__main__":
    main()
