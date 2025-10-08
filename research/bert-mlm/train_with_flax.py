import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from io import open
from datasets import load_dataset
from jax.sharding import Mesh, PartitionSpec
from transformers import AutoTokenizer, BertConfig
from flax import struct
import jax.tree_util as jtu

from src._training import make_module_opt, make_train_step, train_loop
from src._metrics import SufficientMetric
from src._logger import TrackioLogger
from src.callbacks import LearningRateMonitor, ModelCheckpoint
from src.data._training import make_dataloader
from src._utils import print_memory
from src.data.masked_language_modeling import (
    masked_language_modeling_transforms,
)
from src.losses.cross_entropy import softmax_cross_entropy_with_integer_labels
from src.models.bert import BertForMaskedLM
from src._logger import setup_logger
from flaxformer.architectures.bert.bert import BertEncoder
from flaxformer.architectures.bert.heads import MLMHead
import time
import logging
from jax import P

LOGGER = logging.getLogger(__name__)


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


def main():
    setup_logger()
    LOGGER.info("hallo")
    logger = TrackioLogger(project="bert-mlm-fineweb")
    key = jr.PRNGKey(SEED)
    key, model_key, mlm_key = jr.split(key, 3)
    
    devices = jax.devices()
    mesh = Mesh(devices, MESH_AXIS_NAMES)
    
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoder = BertEncoder(
        vocab_size = 30522,
        hidden_size = 768,
        intermediate_dim = 3072,
        max_length = 512,
        num_segments = 2,
        num_hidden_layers = 12,
        num_attention_heads = 12,
        dropout_rate = 0.1,
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
    
    
    learning_rate_monitor = LearningRateMonitor(
        log_every_n_step=LOG_EVERY_N_STEPS,
        schedule_fn=schedule,
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
        dropout_rate=0,
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


    sharding = jax.NamedSharding(mesh, P(None))
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
    
    timing = time.monotonic()
    compiled_text = train_step_fn.lower(train_state, optimizer_placeholder, batch, key=jax.random.key(10)).as_text()
    compiled = train_step_fn.lower(train_state, optimizer_placeholder, batch, key=jax.random.key(10)).compile()
    diff = time.monotonic() - timing
    print(f"DEBUGPRINT[9]: train_with_flax.py:190: diff for compile={diff}")
    print(f"DEBUGPRINT[11]: train_with_flax.py:198: compiled={compiled}")

    with open("./compiled_with_flax_no_dropout.txt", "w") as f:
        f.write(compiled_text)

    print_memory(compiled.memory_analysis())
    return
    

if __name__ == "__main__":
    main()
