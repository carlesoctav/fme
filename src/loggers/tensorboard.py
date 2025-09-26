from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from tensorboardX import SummaryWriter

from .base import Logger


ReduceMethod = str


@dataclass
class TensorBoardLogger(Logger):
    """TensorBoard-backed logger with optional tuple reductions."""

    log_dir: str | Path
    experiment_name: str | None = None
    flush_interval: int = 1
    enabled: bool = True
    reduce: dict[str, ReduceMethod] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Logger.__init__(self, enabled=self.enabled)
        if not self.enabled:
            return
        base_dir = Path(self.log_dir)
        if self.experiment_name:
            base_dir = base_dir / self.experiment_name
        base_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(logdir=str(base_dir))
        self._flush_interval = max(1, self.flush_interval)
        self._writes_since_flush = 0

    def log_metrics(self, metrics: dict[str, object], *, step: int, mode: str) -> bool:
        if not self.enabled or not metrics:
            return False
        wrote = False
        for name, value in metrics.items():
            metric_name = f"{mode}/{name}"
            reduced = self._reduce_value(metric_name, value)
            if reduced is None:
                continue
            self._writer.add_scalar(tag=metric_name, scalar_value=reduced, global_step=step)
            wrote = True
        if wrote:
            self._writes_since_flush += 1
            if self._writes_since_flush >= self._flush_interval:
                self._writer.flush()
                self._writes_since_flush = 0
        return wrote

    def log_histogram(self, name: str, values: Iterable[float], *, step: int, mode: str | None = None) -> None:
        if not self.enabled:
            return
        values = np.asarray(list(values), dtype=np.float32)
        if values.size == 0:
            return
        tag = f"{mode}/{name}" if mode else name
        self._writer.add_histogram(tag=tag, values=values, global_step=step)
        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_interval:
            self._writer.flush()
            self._writes_since_flush = 0

    def finalize(self, status: str = "success") -> None:  # noqa: D401 - keep Logger signature
        if not self.enabled:
            return
        self._writer.flush()
        self._writer.close()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reduce_value(self, name: str, value: object) -> float | None:
        reduce_name = name
        if "/" in name:
            # allow reduce dict to be keyed by plain metric name without mode prefix
            _, bare = name.split("/", 1)
        else:
            bare = name
        method = self.reduce.get(name) or self.reduce.get(bare) or "mean" if isinstance(value, tuple) else None

        if isinstance(value, tuple):
            if method == "sum":
                return float(np.sum(np.asarray(value, dtype=np.float64)))
            if method == "first":
                return float(np.asarray(value[0], dtype=np.float64))
            if method == "last":
                return float(np.asarray(value[-1], dtype=np.float64))
            # default/mean path expects (numerator, denominator)
            if len(value) == 2:
                numerator, denominator = value
                denom = float(denominator)
                if denom == 0:
                    return None
                return float(numerator) / denom
            # fallback to average over all elements
            arr = np.asarray(value, dtype=np.float64)
            if arr.size == 0:
                return None
            return float(arr.mean())

        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except Exception:  # pragma: no cover - defensive
                return None
        return None


__all__ = ["TensorBoardLogger"]
