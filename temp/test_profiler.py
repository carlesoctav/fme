import time

import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from src import Optimizer, make_train_step, nn
import numpy as np
from src.loggers import TensorBoardLogger
from src import train_loop 

BATCH_SIZE = 64
HIDDEN_DIM = 64
NUM_WARMUP = 1
NUM_STEPS = 100
LEARNING_RATE = 1e-2
NUM_SAMPLES = 1000
REPEAT_EPOCHS = 100


class XORDataset(grain.sources.RandomAccessDataSource):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        key = jr.PRNGKey(1234)
        self.data = np.random.randint(0, 2, size=(num_samples, 2))
        self.label = (self.data.sum(axis=1, keepdims=True) == 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        idx = idx % self.num_samples
        return self.data[idx], self.label[idx]


def loss_function(model, optimizer, batch, key):
    del optimizer, key
    x, y = batch
    logits = model(x)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, y).mean()
    probs = jax.nn.sigmoid(logits)
    acc = jnp.mean((probs > 0.5) == (y > 0.5))
    aux = {"loss": loss, "acc": acc}
    return loss, aux


class SuperLinear(eqx.Module):
    linear1: nn.Linear
    linear2: nn.Linear

    def __init__(self, params: list[int], key):
        k1, k2 = jr.split(key, 2)
        self.linear1 = nn.Linear(params[0], params[1], key=k1)
        self.linear2 = nn.Linear(params[1], params[2], key=k2)

    def __call__(self, x):
        h = jax.nn.tanh(self.linear1(x))
        return self.linear2(h)

def main():
    key = jr.PRNGKey(10)
    key, model_key = jr.split(key)
    model = SuperLinear([2, HIDDEN_DIM, 1], model_key)

    grad_tx = optax.sgd(LEARNING_RATE)
    optimizer = Optimizer(grad_tx, model)

    train_step = make_train_step(loss_function, gradient_accumulation_steps = 100)
    logger = TensorBoardLogger(log_dir = "log", experiment_name = "test_experiment", flush_interval = 100)
    map_ds = grain.MapDataset.source(XORDataset(NUM_SAMPLES)).repeat(REPEAT_EPOCHS)
    iter_ds = map_ds.to_iter_dataset(grain.ReadOptions(num_threads = 0, prefetch_buffer_size = 0)).batch(BATCH_SIZE)

    # module: _ModuleInput,
    # optimizer: _OptimizerInput,
    # train_step: _TrainStepCallable[_ModuleInput, _OptimizerInput],
    # train_loader: tp.Iterable[tp.Any],
    # num_train_steps: int | None = None,
    _ = train_loop(
        model,
        optimizer,
        train_step,
        iter_ds,
        100,
        logger =logger
    )


    loss = float(last_aux["loss"]) if last_aux is not None else float("nan")
    acc = float(last_aux["acc"]) if last_aux is not None else float("nan")

    print(f"Average step time (excluding first) over {NUM_STEPS} steps: {avg_step:.6f}s")
    print(f"Final loss: {loss:.6f}, final acc: {acc:.6f}")


if __name__ == "__main__":
    main()
