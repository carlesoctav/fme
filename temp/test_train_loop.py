
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
import trackio

from src import Eval, make_train_step, nn, Optimizer, SufficientMetric, train_loop
from src.callbacks import LearningRateMonitor, ModelCheckpoint
from ._logger import TrackioLogger


BATCH_SIZE = 100
HIDDEN_DIM = 64
NUM_WARMUP = 1
NUM_STEPS = 100
LEARNING_RATE = 1e-2
NUM_SAMPLES = 10000
REPEAT_EPOCHS = 100


class XORDataset(grain.sources.RandomAccessDataSource):
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.data = np.random.randint(0, 2, size=(num_samples, 2))
        self.label = (self.data.sum(axis=1, keepdims=True) == 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        idx = idx % self.num_samples
        return self.data[idx], self.label[idx]


def loss_function(model, optimizer, batch, key):
    x, y = batch
    batch_size = x.shape[0]
    logits = model(x)
    loss = optax.losses.sigmoid_binary_cross_entropy(logits, y).sum()
    probs = jax.nn.sigmoid(logits)
    acc = jnp.sum((probs > 0.5) == (y > 0.5))
    aux = {"loss": (loss, batch_size), "acc": (acc, batch_size)}
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


def get_mask(path, leaf):
    if isinstance(leaf, jax.ShapeDtypeStruct):
        pathstr = jtu.keystr(path)
        print(f"DEBUGPRINT[353]: test_train_loop.py:68: pathstr={pathstr}")
        if pathstr == ".linear1.weight.value":
            return True 
        else:
            return False 
    return leaf
    

def main():
    key = jr.PRNGKey(10)
    key, model_key = jr.split(key)
    abstract_model = eqx.filter_eval_shape(SuperLinear, [2, HIDDEN_DIM, 1] , model_key)
    model = SuperLinear([2, HIDDEN_DIM, 1], model_key)
    mask = jtu.tree_map_with_path(get_mask, abstract_model)
    print(f"DEBUGPRINT[352]: test_train_loop.py:78: mask={mask}")

    schedule = optax.exponential_decay(1e-2, 100, 0.01)
    grad_tx = optax.adamw(schedule, mask = mask)
    print(f"DEBUGPRINT[354]: test_train_loop.py:86: grad_tx={grad_tx}")
    optimizer = Optimizer(grad_tx, model)

    train_step_fn = make_train_step(loss_function)

    map_ds = grain.MapDataset.source(XORDataset(NUM_SAMPLES)).repeat(1)
    iter_ds = map_ds.to_iter_dataset(grain.ReadOptions(num_threads = 0, prefetch_buffer_size = 0)).batch(BATCH_SIZE)
    eval_ds = grain.MapDataset.source(XORDataset(NUM_SAMPLES)).to_iter_dataset(grain.ReadOptions(num_threads = 0, prefetch_buffer_size = 0)).batch(BATCH_SIZE)

    eval = Eval(name = "xor_eval", dataset = eval_ds, loss_function = loss_function)

    logger = TrackioLogger(project="test_project", name="test_run") 
    learning_rate_monitor = LearningRateMonitor(log_every_n_step = 10, schedule_fn = schedule)
    train_metric = SufficientMetric(name = "train", log_every_n_step = 10)

    modelcheckpoint = ModelCheckpoint(f"./checkpoints/{logger.name}", save_interval_steps = 10) 

    train_loop(
        model,
        optimizer,
        train_step_fn,
        iter_ds,
        logger,
        train_metric,
        evals = [eval],
        eval_interval = 10,
        callbacks= [learning_rate_monitor, modelcheckpoint],
    )

    logger.finish()

if __name__ == "__main__":
    main()
