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
        self._window_sum: float = 0.0
        self._window_count: int = 0

    def on_training_step_end(self, *, step_idx: int | None = None, **kwargs: tp.Any) -> None:
        if step_idx is None or self.every_n_steps <= 0:
            return
        if step_idx % self.every_n_steps != 0:
            return
        self._accumulate(step_idx)
        if self.every_n_steps <= 0 or step_idx % self.every_n_steps != 0:
            return
        self._log_payload(step_idx, force_reduce=False)

    def on_validation_end(self, *, step_idx: int | None = None, **kwargs: tp.Any) -> None:
        if not self.log_on_eval_end or step_idx is None:
            return
        self._accumulate(step_idx)
        self._log_payload(step_idx, force_reduce=True)

    def on_training_end(self, *, step_idx: int | None = None, **kwargs: tp.Any) -> None:
        if not self.log_on_train_end or step_idx is None:
            return
        self._accumulate(step_idx)
        self._log_payload(step_idx, force_reduce=True)

    def _accumulate(self, step: int) -> float:
        value = self.scheduler(step)
        if isinstance(value, jax.Array):
            value = jax.device_get(value)
        if hasattr(value, "item"):
            value = tp.cast(float, value.item())
        else:
            value = float(value)
        self._latest_value = value
        self._window_sum += value
        self._window_count += 1
        return value

    def _log_payload(self, step: int, *, force_reduce: bool) -> None:
        if self._latest_value is None:
            return
        payload = {self.metric_name: self._latest_value}
        should_reduce = force_reduce or (
            self.every_n_steps > 0 and step % self.every_n_steps == 0
        )
        if should_reduce and self._window_count > 0:
            payload[f"{self.metric_name}_per_N"] = self._window_sum / self._window_count
            self._window_sum = 0.0
            self._window_count = 0

        log_method = getattr(self.logger, "log", None)
        if log_method is not None:
            rank_zero(log_method, self.tag, payload, step=step)
        else:
            rank_zero(self.logger.log_scalars, self.tag, payload, step=step)

    @property
    def latest(self) -> float | None:
        return self._latest_value
