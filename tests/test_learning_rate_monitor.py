from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from src.callbacks.learning_rate import LearningRateMonitor
from src.loggers import CSVLogger, Logger


class RecordingLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[tuple[int, str, dict[str, float]]] = []

    def log_metrics(self, metrics, *, step: int, mode: str) -> bool:
        self.records.append((step, mode, dict(metrics)))
        return True


def test_learning_rate_monitor_logs_by_step() -> None:
    logger = RecordingLogger()
    monitor = LearningRateMonitor(
        scheduler=lambda step: float(step) * 0.5,
        logger=logger,
        every_n_steps=2,
    )
    monitor.on_train_step_end(step=1)
    monitor.on_train_step_end(step=2)
    monitor.on_train_step_end(step=3)
    monitor.on_train_step_end(step=4)
    assert logger.records == [
        (2, "lr", {"learning_rate": 1.0}),
        (4, "lr", {"learning_rate": 2.0}),
    ]


def test_learning_rate_monitor_logs_on_eval() -> None:
    logger = RecordingLogger()
    monitor = LearningRateMonitor(
        scheduler=lambda step: step,
        logger=logger,
        every_n_steps=10,
    )
    monitor.on_eval_end(step=5)
    assert logger.records == [(5, "lr", {"learning_rate": 5.0})]


def test_learning_rate_monitor_uses_separate_csv(tmp_path) -> None:
    logger = CSVLogger(log_dir=tmp_path)
    monitor = LearningRateMonitor(
        scheduler=lambda step: jnp.asarray(step, dtype=jnp.float32),
        logger=logger,
        every_n_steps=1,
    )
    monitor.on_train_step_end(step=3)
    lr_file = Path(tmp_path) / "learning_rate.csv"
    assert lr_file.exists()
    contents = lr_file.read_text().strip().splitlines()
    assert contents[0] == "step,mode,metric,value"
    assert contents[1] == "3,lr,learning_rate,3.0"
