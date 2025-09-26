from collections.abc import Callable

import equinox as eqx
import grain
import jax
import numpy as np
import optax
from grain._src.python.samplers import IndexSampler
from grain.transforms import Batch
from jaxtyping import Array, PyTree
from optax import GradientTransformationExtraArgs

from src.nn import Linear


transformations = [
    Batch(batch_size=50, drop_remainder = True)
]

# datasets = (
#     grain.MapDataset.source(ds)
#     .random_map(DataTransformsForMaskedLanguageModeling(tokenizer, columns = "text_en", max_length = 32), seed = 43)
#     .batch(2)
# )
#


class SuperLinear(eqx.Module):
    linear1: Linear
    linear2: Linear

    def __init__(self, feat: list[int], key):
        lkey1, lkey2 = jax.random.split(key, 2)
        self.linear1 = Linear(feat[0], feat[1], key = lkey1)
        self.linear2 = Linear(feat[1], feat[2], key = lkey2)

    def __call__(self, x):
        return self.linear2(jax.nn.tanh(self.linear1(x)))


key = jax.random.PRNGKey(0)
key, model_key = jax.random.split(key)

def loss_fn(model, batch):
    x, y  = batch
    y = y.reshape(-1, 1)
    logits = jax.vmap(model)(x)
    loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    return loss

grad_fn = eqx.filter_value_and_grad(loss_fn)

model = SuperLinear([2, 8, 1], key = model_key)


class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    step: int
    wrt: Callable[[ PyTree[Array] ], bool] = eqx.field(static=True)
    tx: GradientTransformationExtraArgs = eqx.field(static=True)

    def __init__(self, optimizer, model, *, wrt=eqx.is_inexact_array):
        self.tx = optimizer
        self.wrt = wrt
        self.opt_state = self.tx.init(eqx.filter(model, self.wrt))
        self.step = 0

    def __call__(self, grads: PyTree[Array], model: eqx.Module) -> tuple[eqx.Module, "Optimizer"]:
        updates, opt_state = self.tx.update(
            grads, self.opt_state, eqx.filter(model, self.wrt)
        )
        new_model = eqx.apply_updates(model, updates)
        new_self = eqx.tree_at(lambda o: [ o.opt_state, o.step], self, [opt_state, self.step + 1])
        return new_model, new_self


def train_step(model, batch, optimizer: Optimizer,):
    loss, grads = grad_fn(model, batch)
    jax.debug.print("grads: {grads}", grads=grads)
    model, optimizer = optimizer(grads, model)
    return model, optimizer, loss


key, train_key = jax.random.split(key)

tx = optax.sgd(1e-1)

optimizer = Optimizer(tx, model)

for i, batch in enumerate(data_loader):
    train_key, subkey = jax.random.split(train_key)
    model, optimizer, loss = train_step(model, batch, optimizer)
    print(f"DEBUGPRINT[225]: test_train.py:115: optimizer={optimizer.step}")
