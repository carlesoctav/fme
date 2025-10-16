import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from datasets import load_dataset
from jax.sharding import Mesh
from transformers import AutoTokenizer, BertConfig

from src import Eval, make_module_opt
from src._logger import TrackioLogger
from src.data._training import make_dataloader
from src.data.masked_language_modeling import (
    MLMBatch,
    masked_language_modeling_transforms,
)
from src.losses.cross_entropy import softmax_cross_entropy_with_integer_labels
from src.models.bert import BertForMaskedLM


DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_SPLIT = "train"
DATASET_SUBSET = "sample-10BT"
COLUMN_NAME = "text"
MAX_LENGTH = 128
BATCH_SIZE = 64
EVAL_STEPS = 100
SEED = 42
NUM_WORKERS = 4
WORKER_BUFFER_SIZE = 2
MLM_PROBABILITY = 0.15
MESH_SHAPE = (4,)
MESH_AXIS_NAMES = ("dp",)
CHECKPOINT_PATH = "./checkpoints/bert-mlm"


def _get_position_ids(batch, seq_length):
    if batch.position_ids is not None:
        return batch.position_ids
    batch_size = batch.input_ids.shape[0]
    return jnp.broadcast_to(jnp.arange(seq_length)[None, :], (batch_size, seq_length))


def loss_function(model, optimizer, batch, key):
    batch_size = batch.input_ids.shape[0]
    
    model_key, _ = jr.split(key, 2) if key is not None else (None, None)
    
    logits = jax.vmap(model)(
        input_ids=batch.input_ids,
        position_ids=_get_position_ids(batch, MAX_LENGTH),
        token_type_ids=batch.token_type_ids,
        attention_mask=batch.attention_mask,
        segment_ids=batch.segment_ids,
        key=jr.split(model_key, batch_size) if model_key is not None else None,
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
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        _attn_implementation="sdpa",
    )
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    model = BertForMaskedLM(config, key=model_key)
    
    dummy_grad_tx = optax.identity()
    
    model, optimizer = make_module_opt(
        model,
        dummy_grad_tx,
        mesh=mesh,
        key=model_key,
    )
    
    dataset = load_dataset(
        DATASET_NAME,
        name=DATASET_SUBSET,
        split=DATASET_SPLIT,
        streaming=True,
    )
    
    operations, batch_class = masked_language_modeling_transforms(
        dataset_type="huggingface",
        column=COLUMN_NAME,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        mlm_probability=MLM_PROBABILITY,
        packing=False,
    )
    
    eval_loader = make_dataloader(
        datasets=[dataset],
        operations=operations,
        global_batch_size=BATCH_SIZE,
        axis_names=MESH_AXIS_NAMES,
        mesh=mesh,
        num_epochs=1,
        shuffle=False,
        seed=SEED,
        worker_count=NUM_WORKERS,
        worker_buffer_size=WORKER_BUFFER_SIZE,
        drop_remainder=True,
        batch_class=batch_class,
    )
    
    eval_obj = Eval(
        name="bert_mlm_eval",
        dataset=eval_loader,
        loss_function=loss_function,
    )
    
    logger = TrackioLogger(project="bert-mlm-eval")
    
    eval_metrics, eval_logs = eval_obj.run(
        model,
        optimizer,
        logger=logger,
        key=key,
    )
    
    print(f"Eval metrics: {eval_logs}")


if __name__ == "__main__":
    main()
