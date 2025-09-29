import importlib.machinery
import importlib.util
import sys

import numpy as np
import optax
import pytest

import jax.random as jr

from pathlib import Path

import tensorboardX

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
from src.callbacks.learning_rate import LearningRateMonitor
from src.callbacks.model_checkpoint import ModelCheckpoint
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


tensorboardX.SummaryWriter = _InProcessSummaryWriter

tb_module = sys.modules.get("src.loggers.tensorboard")
if tb_module is not None:
    tb_module.SummaryWriter = _InProcessSummaryWriter

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

NUM_TRAIN_STEPS = 10_000
EVAL_INTERVAL = 5_000
LOG_EVERY = 100
EVAL_BATCHES = 4
MODEL_SHAPE = (2, HIDDEN_DIM, 1)


class TrackingEval(Eval):
    def __init__(self, dataset: XORDataset, batch_size: int, num_batches: int, seed: int) -> None:
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._rng = np.random.default_rng(seed)
        self.steps_seen = 0
        super().__init__(
            name="xor_eval",
            load_dataset_fn=self._load_batches,
            loss_function=loss_function,
        )

    def _load_batches(self) -> list[tuple]:
        batches: list[tuple] = []
        for _ in range(self._num_batches):
            indices = self._rng.integers(0, len(self._dataset), size=self._batch_size)
            x = self._dataset.data[indices]
            y = self._dataset.label[indices]
            batches.append(to_device((x, y)))
        return batches

    def on_eval_step_end(self, module, optimizer, batch, aux, logs, step, *, logger=None) -> None:
        self.steps_seen += 1
        super().on_eval_step_end(module, optimizer, batch, aux, logs, step, logger=logger)


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


def build_eval_loader(dataset: XORDataset, batch_size: int, num_batches: int, seed: int):
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(num_batches):
        indices = rng.integers(0, len(dataset), size=batch_size)
        x = dataset.data[indices]
        y = dataset.label[indices]
        batches.append(to_device((x, y)))
    return batches


def xor_eval_step(module, optimizer, batch, *, key=None):
    del optimizer, key
    _, aux = loss_function(module, optimizer, batch, None)
    return dict(aux)


def build_model_and_optimizer():
    root_key = jr.PRNGKey(0)
    model_key, train_key = jr.split(root_key)
    model = SuperLinear(list(MODEL_SHAPE), model_key)
    optimizer = Optimizer(optax.sgd(XOR_LEARNING_RATE), model)
    return model, optimizer, train_key


def build_logger(base_dir: Path, mode: str) -> TensorBoardLogger:
    log_dir = base_dir / f"tensorboard_{mode}"
    return TensorBoardLogger(log_dir=str(log_dir), experiment_name=f"xor_{mode}")


def build_callbacks(base_dir: Path, logger: TensorBoardLogger, mode: str):
    ckpt_dir = base_dir / f"checkpoints_{mode}"
    checkpoint = ModelCheckpoint(
        directory=str(ckpt_dir),
        monitor="loss",
        mode="min",
        save_on="eval",
        enable_async_checkpointing=False,
    )
    lr_monitor = LearningRateMonitor(
        scheduler=lambda step: XOR_LEARNING_RATE,
        logger=logger,
        every_n_steps=LOG_EVERY,
    )
    return [checkpoint]


def test_train_loop_with_xor(eval_mode: str, tmp_path: Path) -> None:
    model, optimizer, key = build_model_and_optimizer()
    train_step = make_train_step(loss_function)
    train_dataset = XORDataset(XOR_NUM_SAMPLES)
    train_loader = InfiniteXORDataloader(train_dataset, XOR_BATCH_SIZE, seed=42)
    logger = build_logger(tmp_path, eval_mode)
    callbacks = build_callbacks(tmp_path, logger, eval_mode)

    eval_kwargs = {}
    if eval_mode == "eval_class":
        eval_dataset = XORDataset(XOR_NUM_SAMPLES // 2)
        eval_kwargs["evals"] = [TrackingEval(eval_dataset, XOR_BATCH_SIZE, EVAL_BATCHES, seed=123)]
    else:
        eval_dataset = XORDataset(XOR_NUM_SAMPLES // 2)
        eval_kwargs["eval_loader"] = build_eval_loader(
            eval_dataset,
            XOR_BATCH_SIZE,
            EVAL_BATCHES,
            seed=321,
        )
        eval_kwargs["eval_step"] = xor_eval_step

    train_loop(
        model,
        optimizer,
        train_step_fn=train_step,
        train_loader=train_loader,
        num_train_steps=NUM_TRAIN_STEPS,
        logger=logger,
        log_every_n_steps=LOG_EVERY,
        eval_interval=EVAL_INTERVAL,
        callbacks=callbacks,
        key=key,
        **eval_kwargs,
    )


if __name__ == "__main__":
    test_train_loop_with_xor(eval_mode = "eval_loader", tmp_path = Path("./logs"))
