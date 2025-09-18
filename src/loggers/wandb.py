from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .base import Logger


@dataclass
class WandbLogger(Logger):
    project: str | None = None
    name: str | None = None
    config: Mapping[str, object] | None = None
    log_interval: int = 1
    enabled: bool = True
    init_kwargs: dict[str, object] = field(default_factory=dict)

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
