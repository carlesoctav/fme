from __future__ import annotations

import typing as tp

import jax

from ._callbacks import Callback
from ..loggers import Logger
from .._utils import rank_zero


_SchedulerFn = tp.Callable[[int], tp.Any]


class LearningRateMonitor(Callback):
    def __init__(
        self,
        *,
        scheduler: _SchedulerFn,
        logger: Logger,
        metric_name: str = "learning_rate",
        train: str = "lr",
        every_n_steps: int = 50,
        log_on_eval_end: bool = True,
        log_on_train_end: bool = True,
    ) -> None:
        self.scheduler = scheduler
        self.logger = logger
        self.metric_name = metric_name
        self.tag = train
        self.every_n_steps = every_n_steps
        self.log_on_eval_end = log_on_eval_end
        self.log_on_train_end = log_on_train_end
        self._latest_value: float | None = None

    def on_training_step_end(self, *, step: int, **kwargs: tp.Any) -> None:
        if self.every_n_steps <= 0:
            return
        if step % self.every_n_steps != 0:
            return
        self._emit(step)

    def on_validation_end(self, *, step: int, **kwargs: tp.Any) -> None:
        if not self.log_on_eval_end:
            return
        self._emit(step)

    def on_training_end(self, *, step: int, **kwargs: tp.Any) -> None:
        if not self.log_on_train_end:
            return
        self._emit(step)

    @rank_zero
    def _emit(self, step: int) -> None:
        value = self.scheduler(step)
        if isinstance(value, jax.Array):
            value = jax.device_get(value)
        if hasattr(value, "item"):
            value = tp.cast(float, value.item())
        else:
            value = float(value)
        self._latest_value = value
        if not getattr(self.logger, "enabled", True):
            return
        self.logger.log_metrics({self.metric_name: value}, step=step, tag=self.tag)

    @property
    def latest(self) -> float | None:
        return self._latest_value
