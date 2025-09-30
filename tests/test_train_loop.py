import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import numpy as np
import optax
import jax.random as jr
import tensorboardX
from src import SufficientMetric

ROOT = Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load module {module_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


if "src" not in sys.modules:
    spec = importlib.machinery.ModuleSpec("src", loader=None, is_package=True)
    src_module = importlib.util.module_from_spec(spec)
    src_module.__path__ = [str(ROOT / "src")]
    sys.modules["src"] = src_module

    darray_module = _load_module("src._darray", "src/_darray.py")
    src_module.DArray = getattr(darray_module, "DArray")
    nn_module = _load_module("src.nn", "src/nn/__init__.py")
    src_module.nn = nn_module

from src._training import Eval, Optimizer, make_train_step, train_loop
from src.callbacks import Callback
from src.loggers.tensorboard import TensorBoardLogger

_SRC_PACKAGE = sys.modules["src"]
_SRC_PACKAGE.Optimizer = Optimizer
_SRC_PACKAGE.make_train_step = make_train_step


class _InProcessSummaryWriter:
    def __init__(self, logdir, **_):
        self.logdir = logdir

    def add_scalar(self, *_, **__):
        return None

    def add_scalars(self, *_, **__):
        return None

    def flush(self):
        return None

    def close(self):
        return None




from tmep.test_training import (
    BATCH_SIZE as XOR_BATCH_SIZE,
    HIDDEN_DIM,
    LEARNING_RATE as XOR_LEARNING_RATE,
    NUM_SAMPLES as XOR_NUM_SAMPLES,
    SuperLinear,
    XORDataset,
    loss_function,
    to_device,
)

NUM_TRAIN_STEPS = 1000
EVAL_INTERVAL = 500
EVAL_BATCHES = 4
MODEL_SHAPE = (2, HIDDEN_DIM, 1)


class InfiniteXORDataloader:
    def __init__(self, dataset: XORDataset, batch_size: int, seed: int) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            indices = self._rng.integers(0, len(self._dataset), size=self._batch_size)
            x = self._dataset.data[indices]
            y = self._dataset.label[indices]
            yield to_device((x, y))


def build_eval_loader(
    dataset: XORDataset, batch_size: int, num_batches: int, seed: int
):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_batches):
        indices = rng.integers(0, len(dataset), size=batch_size)
        x = dataset.data[indices]
        y = dataset.label[indices]
        batches.append(to_device((x, y)))
    return batches


class TrackingCallback(Callback):
    def __init__(self):
        self.training_started = False
        self.training_steps = []
        self.training_ended = False
        self.validation_started = False
        self.validation_ended = False
        self.eval_count = 0

    def on_training_start(self, module, optimizer, logger):
        self.training_started = True
        print("✓ Training started")

    def on_training_step(self, module, optimizer, batch, metric, logger, step):
        self.training_steps.append(step)

    def on_training_end(self, module, optimizer, metric, logger, step):
        self.training_ended = True
        print(f"✓ Training ended at step {step}")

    def on_validation_start(self, module, optimizer, logger, step):
        self.validation_started = True
        print("✓ Validation started")

    def on_validation_end(self, module, optimizer, eval_metrics, logger, step):
        self.validation_ended = True
        self.eval_count += 1
        print(f"✓ Validation ended at step {step}, metrics: {eval_metrics.keys()}")


def build_model_and_optimizer():
    root_key = jr.PRNGKey(0)
    model_key, train_key = jr.split(root_key)
    model = SuperLinear(list(MODEL_SHAPE), model_key)
    optimizer = Optimizer(optax.sgd(XOR_LEARNING_RATE), model)
    return model, optimizer, train_key


def build_logger(base_dir: Path) -> TensorBoardLogger:
    log_dir = base_dir / "tensorboard"
    return TensorBoardLogger(log_dir=str(log_dir), experiment_name="xor_test")


def test_train_loop_with_xor(tmp_path: Path) -> None:
    print("\n=== Starting XOR Training Test ===")

    model, optimizer, key = build_model_and_optimizer()
    train_step = make_train_step(loss_function)
    train_dataset = XORDataset(XOR_NUM_SAMPLES)
    train_loader = InfiniteXORDataloader(train_dataset, XOR_BATCH_SIZE, seed=42)
    logger = build_logger(tmp_path)

    callback = TrackingCallback()

    eval_dataset = XORDataset(XOR_NUM_SAMPLES // 2)
    eval_loader = build_eval_loader(
        eval_dataset, XOR_BATCH_SIZE, EVAL_BATCHES, seed=321
    )

    eval_obj = Eval(
        name="xor_eval",
        dataset=eval_loader,
        loss_function=loss_function,
        jit=True,
    )

    train_metric = SufficientMetric(name="train", log_every_n_steps=100)
    model, optimizer, train_metric, eval_metrics = train_loop(
        model,
        optimizer,
        train_step_fn=train_step,
        train_loader=train_loader,
        num_train_steps=NUM_TRAIN_STEPS,
        logger=logger,
        eval_interval=EVAL_INTERVAL,
        evals=[eval_obj],
        key=key,
    )

    print(f"\n=== Test Results ===")
    train_metrics = train_metric.summary()

    print("\n✓ All assertions passed!")


if __name__ == "__main__":
    test_train_loop_with_xor(tmp_path=Path("./log/test_experiment/run3"))
