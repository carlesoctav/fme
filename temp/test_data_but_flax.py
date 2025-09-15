import jax.numpy as jnp
from typing import Callable
import numpy as np
import grain
import jax
from datasets import load_dataset
from grain._src.python.samplers import IndexSampler
from grain.transforms import Batch
from transformers import BertTokenizer, FlaxAutoModelForMaskedLM
from transformers.models.bert.configuration_bert import BertConfig
import optax
from flax.training import train_state

from src.data import DataTransformsForMaskedLMGivenText


ds = load_dataset("carlesoctav/en-id-parallel-sentences", split="QED")
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
transformations = [
    DataTransformsForMaskedLMGivenText(tokenizer, columns="text_en", max_length=20),
    Batch(batch_size=2, drop_remainder=True),
]

# datasets = (
#     grain.MapDataset.source(ds)
#     .random_map(DataTransformsForMaskedLanguageModeling(tokenizer, columns = "text_en", max_length = 32), seed = 43)
#     .batch(2)
# )
#
datasets = grain.DataLoader(
    data_source=ds,
    operations=transformations,
    sampler=IndexSampler(num_records=len(ds), shuffle=True, seed=42),
)

config = BertConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=128,
    type_vocab_size=2,
    hidden_dropout_prob=0.0,
    layer_norm_eps=1e-5,
)


key = jax.random.PRNGKey(0)

model = FlaxAutoModelForMaskedLM.from_config(config, seed=0)

params = model.params
columns = "text_en"


def masked_lm_loss_function(
    params,
    batch,
    apply_fn,
    *,
    ignore_index: int = -100,
):
    """Compute masked LM loss, ignoring positions with label == ignore_index."""
    x, y = batch[columns], batch["labels"]

    # Apply model to get logits using apply_fn with proper parameter passing
    outputs = apply_fn(params=params, **x)
    logits = outputs.logits

    # Create mask for valid labels (not ignore_index)
    label_mask = jnp.where(y != ignore_index, 1.0, 0.0)

    # Compute cross entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y) * label_mask

    # Normalize by number of valid tokens
    return loss.sum() / label_mask.sum()


# Create TrainState using the model's __call__ method
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-4, eps=1e-4),
)

state = train_state.TrainState.create(apply_fn=model.__call__, params=params, tx=tx)


@jax.jit
def train_step(state, batch):
    """Train step using Flax TrainState."""

    def loss_fn(params):
        return masked_lm_loss_function(params, batch, state.apply_fn)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


key, train_key = jax.random.split(key)

for i, batch in enumerate(datasets):
    print(f"DEBUGPRINT[215]: test_data.py:115: i={i}")
    batch_size, seq_len = batch[columns]["input_ids"].shape
    train_key, subkey = jax.random.split(train_key)

    batch[columns]["position_ids"] = np.broadcast_to(
        np.arange(seq_len, dtype=np.int32), (batch_size, seq_len)
    )

    state, loss = train_step(state, batch)
    print(f"iter={i} loss={loss}")

    if i >= 3:  # Limit iterations for testing
        break
