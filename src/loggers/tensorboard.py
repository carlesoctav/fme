from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from tensorboardX import SummaryWriter
import jax.tree_util as jtu
from jaxtyping import PyTree
import contextlib
import time

from .base import Logger

@dataclass
class TensorBoardLogger(Logger):
    """TensorBoard-backed logger with optional tuple reductions."""

    log_dir: str | Path
    experiment_name: str | None = None
    flush_interval: int = 1
    enabled: bool = True
    reduce_fn: Callable[[PyTree], PyTree] | None = field(default = None)
    time_fn: Callable[[], float] = field(default = time.perf_counter)

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

    def log_metric(
        self,
        name: str,
        metric: float,
        step: int,
        tag: str
    ): 
        if not self.enabled or not metrics:
            return False

        name = f"{tag}/{name}"

        self._writer.add_scalar(name, metric, global_step=step)

        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_interval:
            self._writer.flush()
            self._writes_since_flush = 0

    def log_metrics(
        self,
        metrics: dict[str, object],
        step: int,
        tag: str
    ) -> bool:
        if not self.enabled or not metrics:
            return 

        def _reduce(x):
            if isinstance(x, tuple):
                return x[0] /x[1]
            return x

        reduced = self.reduce_fn(metrics) if self.reduce_fn else jtu.tree_map(_reduce, metrics, is_leaf=lambda x: isinstance(x, tuple))
        self._writer.add_scalars(tag=tag, tag_scalar_dict=reduced, global_step=step)

        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_interval:
            self._writer.flush()
            self._writes_since_flush = 0

        return reduced

    def finalize(self, status: str = "success") -> None: 
        if not self.enabled:
            return
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
