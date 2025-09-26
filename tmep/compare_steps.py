import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from flax import linen as flnn

from src import nn, Optimizer

BATCH_SIZE = 64
INPUT_DIM = 2
HIDDEN_DIM = 64
OUTPUT_DIM = 1
NUM_STEPS = 100
WARMUP_STEPS = 1
LEARNING_RATE = 1e-2


def make_batches(num_batches: int, batch_size: int, *, key: jax.Array):
    keys = jax.random.split(key, num_batches)
    batches: list[tuple[jax.Array, jax.Array]] = []
    for k in keys:
        x = jax.random.randint(k, shape=(batch_size, INPUT_DIM), minval=0, maxval=2)
        x = x.astype(jnp.float32)
        y = (x.sum(axis=1, keepdims=True) == 1).astype(jnp.float32)
        batches.append((x, y))
    return batches


class FlaxMLP(flnn.Module):
    hidden_dim: int

    @flnn.compact
    def __call__(self, x):
        x = flnn.Dense(self.hidden_dim)(x)
        x = jax.nn.tanh(x)
        x = flnn.Dense(OUTPUT_DIM)(x)
        return x


class EqxMLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, *, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.linear1 = eqx.nn.Linear(INPUT_DIM, HIDDEN_DIM, key=k1)
        self.linear2 = eqx.nn.Linear(HIDDEN_DIM, OUTPUT_DIM, key=k2)

    def __call__(self, x):
        def forward(sample):
            h = self.linear1(sample)
            h = jax.nn.tanh(h)
            y = self.linear2(h)
            return y

        return jax.vmap(forward)(x)


class CustomMLP(eqx.Module):
    linear1: nn.Linear
    linear2: nn.Linear

    def __init__(self, *, key: jax.Array):
        k1, k2 = jax.random.split(key, 2)
        self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM, key=k1)
        self.linear2 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM, key=k2)

    def __call__(self, x):
        def forward(sample):
            h = self.linear1(sample)
            h = jax.nn.tanh(h)
            y = self.linear2(h)
            return y

        return jax.vmap(forward)(x)


@dataclass
class BenchmarkResult:
    variant: str
    avg_step_s: float
    final_loss: float
    final_acc: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "variant": self.variant,
            "avg_step_s": self.avg_step_s,
            "final_loss": self.final_loss,
            "final_acc": self.final_acc,
        }


def _binary_metrics(logits: jax.Array, labels: jax.Array) -> tuple[jax.Array, jax.Array]:
    labels_bool = labels > 0.5
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = jnp.mean((logits > 0) == labels_bool)
    return loss, acc


def benchmark_flax(batches: list[tuple[jax.Array, jax.Array]]) -> BenchmarkResult:
    model = FlaxMLP(hidden_dim=HIDDEN_DIM)
    init_key = jax.random.PRNGKey(0)
    sample_x, _ = batches[0]
    params = model.init(init_key, sample_x)["params"]
    tx = optax.sgd(LEARNING_RATE)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(p):
            logits = model.apply({"params": p}, x)
            logits = logits.reshape(y.shape)
            loss, acc = _binary_metrics(logits, y)
            return loss, (logits, acc)

        (loss, (logits, acc)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, acc

    params, opt_state, _, _ = train_step(params, opt_state, *batches[0])

    start = time.perf_counter()
    loss = acc = None
    for x, y in batches[1:]:
        params, opt_state, loss, acc = train_step(params, opt_state, x, y)
    elapsed = time.perf_counter() - start
    steps = len(batches) - WARMUP_STEPS
    return BenchmarkResult(
        variant="flax.linen",
        avg_step_s=elapsed / steps,
        final_loss=float(loss),
        final_acc=float(acc),
    )


def benchmark_eqx(batches: list[tuple[jax.Array, jax.Array]]) -> BenchmarkResult:
    model = EqxMLP(key=jax.random.PRNGKey(1))
    tx = optax.sgd(LEARNING_RATE)
    optimizer = Optimizer(tx, model)

    @eqx.filter_jit
    def train_step(model, optimizer, x, y):
        def loss_fn(m):
            logits = m(x)
            logits = logits.reshape(y.shape)
            loss, acc = _binary_metrics(logits, y)
            return loss, (logits, acc)

        (loss, (logits, acc)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        grads = eqx.filter(grads, eqx.is_array)
        model, optimizer = optimizer(grads, model)
        return model, optimizer, loss, acc

    model, optimizer, _, _ = train_step(model, optimizer, *batches[0])

    start = time.perf_counter()
    loss = acc = None
    for x, y in batches[1:]:
        model, optimizer, loss, acc = train_step(model, optimizer, x, y)
    elapsed = time.perf_counter() - start
    steps = len(batches) - WARMUP_STEPS
    return BenchmarkResult(
        variant="equinox.nn",
        avg_step_s=elapsed / steps,
        final_loss=float(loss),
        final_acc=float(acc),
    )


def benchmark_custom(batches: list[tuple[jax.Array, jax.Array]]) -> BenchmarkResult:
    model = CustomMLP(key=jax.random.PRNGKey(2))
    tx = optax.sgd(LEARNING_RATE)
    optimizer = Optimizer(tx, model)

    @eqx.filter_jit
    def train_step(model, optimizer, x, y):
        def loss_fn(m):
            logits = m(x)
            logits = logits.reshape(y.shape)
            loss, acc = _binary_metrics(logits, y)
            return loss, (logits, acc)

        (loss, (logits, acc)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        grads = eqx.filter(grads, eqx.is_array)
        model, optimizer = optimizer(grads, model)
        return model, optimizer, loss, acc

    model, optimizer, _, _ = train_step(model, optimizer, *batches[0])

    start = time.perf_counter()
    loss = acc = None
    for x, y in batches[1:]:
        model, optimizer, loss, acc = train_step(model, optimizer, x, y)
    elapsed = time.perf_counter() - start
    steps = len(batches) - WARMUP_STEPS
    return BenchmarkResult(
        variant="custom.nn",
        avg_step_s=elapsed / steps,
        final_loss=float(loss),
        final_acc=float(acc),
    )


if __name__ == "__main__":
    total_batches = NUM_STEPS + WARMUP_STEPS
    batches = make_batches(total_batches, BATCH_SIZE, key=jax.random.PRNGKey(42))

    results = [
        benchmark_flax(batches),
        benchmark_eqx(batches),
        benchmark_custom(batches),
    ]

    for result in results:
        stats = result.as_dict()
        print(
            f"{stats['variant']}: avg_step_s={stats['avg_step_s']:.6f}, "
            f"final_loss={stats['final_loss']:.4f}, final_acc={stats['final_acc']:.4f}"
        )
