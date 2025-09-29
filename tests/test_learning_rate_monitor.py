from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp

from src.callbacks.learning_rate import LearningRateMonitor
from src.loggers import Logger

try:
    from src.loggers import CSVLogger
except ImportError:  # pragma: no cover - fallback for tests
    class CSVLogger(Logger):
        def __init__(self, log_dir) -> None:  # type: ignore[no-untyped-def]
            self.log_every_n_steps = None
            self.path = Path(log_dir) / "learning_rate.csv"
            self.path.write_text("step,mode,metric,value\n", encoding="utf-8")

        def _write(self, payload: dict[str, float], *, step: int, tag: str) -> None:
            with self.path.open("a", encoding="utf-8") as fp:
                for name, value in payload.items():
                    fp.write(f"{step},{tag},{name},{value}\n")

        def log(
            self,
            tag: str,
            metrics,
            *,
            step: int,
        ) -> dict[str, float] | None:
            if not metrics:
                return None
            payload = {str(name): float(value) for name, value in metrics.items()}
            self._write(payload, step=step, tag=tag)
            return payload

        def log_scalars(
            self,
            tag: str,
            metrics,
            *,
            step: int,
        ) -> dict[str, float] | None:
            if not metrics:
                return None
            payload = {str(name): float(value) for name, value in metrics.items()}
            self._write(payload, step=step, tag=tag)
            return payload


class RecordingLogger(Logger):
    def __init__(self, log_every_n_steps: int | None = None) -> None:
        super().__init__()
        self.records: list[tuple[int, str, dict[str, float]]] = []

    def log(
        self,
        tag: str,
        metrics,
        *,
        step: int,
    ) -> dict[str, float] | None:
        normalised = {
            str(name): float(value)
            for name, value in (metrics or {}).items()
        }

        if normalised:
            self.records.append((step, tag, normalised))
            return normalised
        return None

    def log_scalars(
        self,
        tag: str,
        metrics,
        *,
        step: int,
    ):
        if not metrics:
            return None
        payload = {str(name): float(value) for name, value in metrics.items()}
        self.records.append((step, tag, payload))
        return payload


def test_learning_rate_monitor_logs_by_step() -> None:
    logger = RecordingLogger(log_every_n_steps=2)
    monitor = LearningRateMonitor(
        scheduler=lambda step: float(step) * 0.5,
        logger=logger,
        every_n_steps=2,
    )
    monitor.on_training_step_end(step_idx=1)
    monitor.on_training_step_end(step_idx=2)
    monitor.on_training_step_end(step_idx=3)
    monitor.on_training_step_end(step_idx=4)
    assert logger.records == [
        (1, "lr", {"learning_rate": 0.5}),
        (2, "lr", {"learning_rate": 1.0, "learning_rate_per_N": 0.75}),
        (3, "lr", {"learning_rate": 1.5}),
        (4, "lr", {"learning_rate": 2.0, "learning_rate_per_N": 1.75}),
    ]


def test_learning_rate_monitor_logs_on_eval() -> None:
    logger = RecordingLogger(log_every_n_steps=1)
    monitor = LearningRateMonitor(
        scheduler=lambda step: step,
        logger=logger,
        every_n_steps=10,
    )
    monitor.on_validation_end(step_idx=5)
    assert logger.records == [
        (5, "lr", {"learning_rate": 5.0, "learning_rate_per_N": 5.0})
    ]


def test_learning_rate_monitor_uses_separate_csv(tmp_path) -> None:
    logger = CSVLogger(log_dir=tmp_path)
    monitor = LearningRateMonitor(
        scheduler=lambda step: jnp.asarray(step, dtype=jnp.float32),
        logger=logger,
        every_n_steps=1,
    )
    monitor.on_training_step_end(step_idx=3)
    lr_file = Path(tmp_path) / "learning_rate.csv"
    assert lr_file.exists()
    contents = lr_file.read_text().strip().splitlines()
    assert contents[0] == "step,mode,metric,value"
    assert contents[1] == "3,lr,learning_rate,3.0"
