from __future__ import annotations

import typing as tp

import jax

from ._callbacks import Callback
from ..loggers import CSVLogger, Logger


_SchedulerFn = tp.Callable[[int], tp.Any]


class LearningRateMonitor(Callback):
    def __init__(
        self,
        *,
        scheduler: _SchedulerFn,
        logger: Logger,
        metric_name: str = "learning_rate",
        mode: str = "lr",
        every_n_steps: int = 50,
        log_on_eval_end: bool = True,
        log_on_train_end: bool = True,
    ) -> None:
        self.scheduler = scheduler
        if isinstance(logger, CSVLogger):
            logger = CSVLogger(
                log_dir=logger.log_dir,
                filename="learning_rate.csv",
                log_interval=logger.log_interval,
                enabled=logger.enabled,
            )
        self.logger = logger
        self.metric_name = metric_name
        self.mode = mode
        self.every_n_steps = every_n_steps
        self.log_on_eval_end = log_on_eval_end
        self.log_on_train_end = log_on_train_end
        self._latest_value: float | None = None

    def on_train_step_end(self, *, step: int, **kwargs: tp.Any) -> None:
        if self.every_n_steps <= 0:
            return
        if step % self.every_n_steps != 0:
            return
        self._emit(step)

    def on_eval_end(self, *, step: int, **kwargs: tp.Any) -> None:
        if not self.log_on_eval_end:
            return
        self._emit(step)

    def on_training_end(self, *, step: int, **kwargs: tp.Any) -> None:
        if not self.log_on_train_end:
            return
        self._emit(step)

    def _emit(self, step: int) -> None:
        if not getattr(self.logger, "enabled", True):
            return
        value = self.scheduler(step)
        if isinstance(value, jax.Array):
            value = jax.device_get(value)
        if hasattr(value, "item"):
            value = tp.cast(float, value.item())
        else:
            value = float(value)
        self._latest_value = value
        self.logger.log_metrics({self.metric_name: value}, step=step, file=self.mode)

    @property
    def latest(self) -> float | None:
        return self._latest_value
