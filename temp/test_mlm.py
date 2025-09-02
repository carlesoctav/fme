from collections.abc import Callable

import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from grain._src.core.transforms import Batch
from grain._src.python.samplers import IndexSampler
from jaxtyping import Array, PyTree
from optax import GradientTransformationExtraArgs
from transformers import BertConfig

from src.models.bert.modeling_bert import BertModelForMaskedLM


config = BertConfig(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
    max_position_embeddings=128,
    type_vocab_size=2,
    hidden_dropout_prob=0.1,
    layer_norm_eps=1e-5,
)

key = jax.random.key(10)
model = BertModelForMaskedLM(config, key=key, dtype=jnp.float32)


class Dataset(grain.MapDataset):
    def __init__(self, num_sample: int, seq_len: int = 32, vocab_size: int = 100):
        self.x = np.random.randint(1, vocab_size, size=(num_sample, seq_len))
        masked_indices = np.random.binomial(1, 0.15, size=self.x.shape).astype(bool)

        self.y = np.full(self.x.shape, -100, dtype=int)
        self.y[masked_indices] = self.x[masked_indices]
        self.x[masked_indices] = 0

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
        pass


transformations = [Batch(batch_size=4, drop_remainder=True)]

ds = Dataset(num_sample=1000)

data_loader = grain.DataLoader(
    data_source=ds,
    operations=transformations,
    sampler=IndexSampler(num_records=len(ds), shuffle=True, seed=42),
)


def masked_lm_loss_function(
    model: BertModelForMaskedLM,
    batch,
    *,
    ignore_index: int = -100,
    key,
):
    """Compute masked LM loss, ignoring positions with label == ignore_index.

    - Vectorizes the model over batch.
    - Uses safe labels for ignored positions to avoid invalid indexing.
    - Normalizes by the number of unmasked tokens.
    """
    x, y = batch["input"], batch["labels"]
    batch_size, seq_len = x["input_ids"].shape

    batched_key = jax.random.split(key, batch_size)
    logits = jax.vmap(model)(**x, key=batched_key)

    label_mask = jnp.where(y != ignore_index, 1.0, 0.0)

    loss = (
        optax.softmax_cross_entropy_with_integer_labels(logits, y) * label_mask
    )  # (..., ) (..., )

    return loss.sum() / label_mask.sum()


grad_fn = eqx.filter_value_and_grad(masked_lm_loss_function, has_aux=False)


class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    wrt: Callable[[PyTree[Array]], bool] = eqx.field(static=True)
    step: int = eqx.field(static=True)
    tx: GradientTransformationExtraArgs = eqx.field(static=True)

    def __init__(self, optimizer, model, *, wrt=eqx.is_inexact_array):
        self.tx = optimizer
        self.wrt = wrt
        self.opt_state = self.tx.init(eqx.filter(model, self.wrt))
        self.step = 0

    def __call__(
        self, grads: PyTree[Array], model: eqx.Module
    ) -> tuple[eqx.Module, "Optimizer"]:
        updates, opt_state = self.tx.update(
            grads, self.opt_state, eqx.filter(model, self.wrt)
        )
        new_model = eqx.apply_updates(model, updates)
        new_self = eqx.tree_at(lambda o: o.opt_state, self, opt_state)
        return new_model, new_self


@eqx.filter_jit
def train_step(model, batch, optimizer: Optimizer, *, key):
    loss, grads = grad_fn(model, batch, key=key)
    model, optimizer = optimizer(grads, model)
    return model, optimizer, loss


key, train_key = jax.random.split(key)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-4, eps=1e-4),  # Much larger epsilon
)
optimizer = Optimizer(tx, model)

for i, (x, y) in enumerate(data_loader):
    seq_len = 32
    train_key, subkey = jax.random.split(train_key)
    input = {
        "input": {
            "input_ids": x,
            "token_type_ids": np.zeros_like(x),
            "position_ids": np.broadcast_to(np.arange(seq_len)[None, :], x.shape),
            "attention_mask": np.ones_like(x),
        },
        "labels": y,
    }
    model, optimizer, loss = train_step(model, input, optimizer, key=subkey)
    print(f"DEBUGPRINT[227]: test_mlm.py:129: loss={loss}")
    if i == 3:
        break
