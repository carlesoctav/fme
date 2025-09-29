from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

import jax

from .base import Logger


@dataclass
class WandbLogger(Logger):
    project: str | None = None
    name: str | None = None
    config: Mapping[str, object] | None = None
    log_interval: int = 1
    enabled: bool = True
    init_kwargs: dict[str, object] = field(default_factory=dict)
    log_every_n_steps: int | None = None

    def __post_init__(self) -> None:
        Logger.__init__(self, enabled=self.enabled)
        self._run = None
        if not self.enabled:
            return
        try:
            import wandb
        except Exception as exc:
            raise ImportError("wandb is required for WandbLogger") from exc
        kwargs = dict(self.init_kwargs)
        if self.project is not None:
            kwargs.setdefault("project", self.project)
        if self.name is not None:
            kwargs.setdefault("name", self.name)
        self._wandb = wandb
        self._run = wandb.init(**kwargs)
        if self.config:
            self._run.config.update(dict(self.config), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int, mode: str) -> bool:
        if not self.enabled or not metrics:
            return False
        if self.log_interval <= 0 or step % self.log_interval != 0:
            return False
        payload = {f"{mode}/{name}": value for name, value in metrics.items()}
        if self._run is None:
            return False
        self._wandb.log(payload, step=step)
        return True

    def _to_float(self, value: object) -> float:
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, (np.floating, np.integer)):
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

    def log(
        self,
        tag: str,
        metrics: Mapping[str, object] | None,
        *,
        step: int,
    ) -> Mapping[str, float] | None:
        if not self.enabled:
            return None

        normalised = {
            name: self._to_float(value)
            for name, value in (metrics or {}).items()
        }

        if normalised and self._run is not None:
            payload = {f"{tag}/{name}": value for name, value in normalised.items()}
            self._wandb.log(payload, step=step)

        return normalised or None

    def log_scalars(
        self,
        tag: str,
        metrics: Mapping[str, object] | None,
        *,
        step: int,
    ) -> Mapping[str, float] | None:
        return self.log(tag, metrics, step=step)

    def log_hyperparams(self, params: Mapping[str, object] | None = None) -> None:
        if not self.enabled or params is None or self._run is None:
            return None
        self._run.config.update(dict(params), allow_val_change=True)
        return None

    def finalize(self, status: str = "success") -> None:
        if not self.enabled or self._run is None:
            return None
        self._run.finish(exit_code=0 if status == "success" else 1)
        self._run = None
        return None
