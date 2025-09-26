from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from tensorboardX import SummaryWriter
import jax.tree_util as jtu
from jaxtyping import PyTree

from .base import Logger

@dataclass
class TensorBoardLogger(Logger):
    """TensorBoard-backed logger with optional tuple reductions."""

    log_dir: str | Path
    experiment_name: str | None = None
    flush_interval: int = 1
    enabled: bool = True
    reduce_fn: Callable[[PyTree], PyTree] | None = field(default = None)

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

    def log_metrics(self, metrics: dict[str, object], *, step: int, tag: str) -> bool:
        if not self.enabled or not metrics:
            return False

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
        return True

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

    def finalize(self, status: str = "success") -> None: 
        if not self.enabled:
            return
        self._writer.flush()
        self._writer.close()

__all__ = ["TensorBoardLogger"]
