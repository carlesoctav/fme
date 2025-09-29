from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import jax
import numpy as np
from tensorboardX import SummaryWriter

from .base import Logger


@dataclass
class TensorBoardLogger(Logger):
    """TensorBoard-backed logger with optional tuple reductions."""

    log_dir: str | Path
    experiment_name: str | None = None
    flush_interval: int = 1
    reduce_fn: Callable[[Mapping[str, float]], Mapping[str, float]] | None = field(default=None)
    time_fn: Callable[[], float] = field(default=time.perf_counter)

    def __post_init__(self) -> None:
        base_dir = Path(self.log_dir)
        if self.experiment_name:
            base_dir = base_dir / self.experiment_name
        base_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(logdir=str(base_dir))
        self._flush_interval = max(1, self.flush_interval)
        self._writes_since_flush = 0

    def _to_float(self, value: object) -> float:
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, jax.Array):
            if value.ndim == 0:
                return float(jax.device_get(value))
            return float(jax.device_get(value).mean())
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:  # pragma: no cover - defensive
                pass
        raise TypeError(f"Unsupported metric value type: {type(value)!r}")

    def _normalise_metrics(self, metrics: Mapping[str, object] | None) -> dict[str, float]:
        if not metrics:
            return {}
        normalised = {name: self._to_float(value) for name, value in metrics.items()}
        if self.reduce_fn is not None and normalised:
            try:
                reduced = self.reduce_fn(normalised)
                if isinstance(reduced, Mapping):
                    normalised = {str(name): float(value) for name, value in reduced.items()}
            except Exception:  # pragma: no cover - defensive
                pass
        return normalised


    def _flush_required(self) -> None:
        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_interval:
            self._writer.flush()
            self._writes_since_flush = 0

    def log_scalar(self, tag: str, metric: float, *, step: int) -> Mapping[str, float]:
        payload = {tag: metric}
        self._writer.add_scalar(tag=tag, scalar_value=metric, global_step=step)
        self._flush_required()
        return payload

    def log(
        self,
        tag: str,
        metrics: Mapping[str, object] | None,
        *,
        step: int,
    ) -> Mapping[str, float] | None:
        normalised = self._normalise_metrics(metrics)
        if not normalised:
            return None
        self._writer.add_scalars(tag=tag, tag_scalar_dict=normalised, global_step=step)
        self._flush_required()
        return normalised

    def log_scalars(
        self,
        tag: str,
        metrics: Mapping[str, object] | None,
        *,
        step: int,
    ) -> Mapping[str, float] | None:
        return self.log(tag, metrics, step=step)

    def finalize(self, status: str = "success") -> None: 
        self._writer.flush()
        self._writer.close()

    @contextlib.contextmanager
    def wc(
        self,
        name,
        step,
    ):
        try:
            start = self.time_fn()
        finally:
            end = self.time_fn()
            duration = max(0.0, end-start)
            self._writer.add_scalar(tag=name, scalar_value=duration, global_step=step)

__all__ = ["TensorBoardLogger"]
