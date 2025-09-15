from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np


def _to_scalar(x: Any) -> Any:
    try:
        x = jax.device_get(x)
    except Exception:
        pass
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    if isinstance(x, (float, int, str)):
        return x
    if isinstance(x, np.ndarray) and x.size == 1:
        return x.reshape(()).item()
    return x


@dataclass
class FileLogger:
    log_dir: str | Path
    filename: str = "train.log"
    flush_every: int = 1

    def __post_init__(self):
        self.log_dir = str(self.log_dir)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._path = Path(self.log_dir) / self.filename
        self._counter = 0

    def log_metrics(self, metrics: dict[str, Any], step: int, prefix: str | None = None):
        line_parts = [f"step={step}"]
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        for k, v in metrics.items():
            v = _to_scalar(v)
            line_parts.append(f"{k}={v}")
        line = " ".join(line_parts)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            self._counter += 1
            if self._counter % self.flush_every == 0:
                f.flush()

    # Training lifecycle hooks (no-op by default except writing a line)
    def on_training_start(self):
        with open(self._path, "a", encoding="utf-8") as f:
            f.write("[event] training_start\n")
            f.flush()

    def on_training_end(self):
        with open(self._path, "a", encoding="utf-8") as f:
            f.write("[event] training_end\n")
            f.flush()

    def on_training_epoch_start(self, epoch_idx: int):
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(f"[event] training_epoch_start epoch={epoch_idx}\n")
            f.flush()

    def on_training_epoch_end(self, metrics: dict[str, Any] | None, epoch_idx: int):
        with open(self._path, "a", encoding="utf-8") as f:
            if metrics:
                flat = " ".join(f"{k}={_to_scalar(v)}" for k, v in metrics.items())
                f.write(f"[event] training_epoch_end epoch={epoch_idx} {flat}\n")
            else:
                f.write(f"[event] training_epoch_end epoch={epoch_idx}\n")
            .flush()
