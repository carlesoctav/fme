from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.tree_util as jtu

from ._callbacks import Callback


LOGGER = logging.getLogger(__name__)


def _block_until_ready(tree: Any) -> None:
    def _maybe_block(x: Any) -> Any:
        blocker = getattr(x, "block_until_ready", None)
        return blocker() if callable(blocker) else x

    try:
        jtu.tree_map(_maybe_block, tree)
    except Exception:
        LOGGER.debug("Failed to block on object while stopping profiler", exc_info=True)


class JaxProfiler(Callback):
    """Callback that schedules JAX profiler traces during training."""

    def __init__(
        self,
        *,
        log_dir: str | Path = "log",
        profile_log_dir: str = "tensorboard",
        main_process_only: bool = False,
        profile_every_n_minutes: float = 60,
        profile_first_step: int = 10,
        profile_n_steps: int = 5,
        profile_max_seconds: float | None = None,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.log_path = Path(log_dir) / profile_log_dir
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.profile_every_n_minutes = profile_every_n_minutes
        self.profile_first_step = profile_first_step
        self.profile_n_steps = max(1, profile_n_steps)
        self.profile_max_seconds = profile_max_seconds
        self.main_process_only = main_process_only
        self._time = time_fn

        self._active = False
        self._last_profile_time = self._time()
        self._trace_start_step: int | None = None
        self._deadline: float | None = None
        LOGGER.debug("JaxProfiler initialized: %s", self.log_path)

    def on_training_start(self, **_: Any) -> None:
        self._active = False
        self._trace_start_step = None
        self._deadline = None
        self._last_profile_time = self._time()

    def on_training_step_end(self, *, step: int, aux: Any | None = None, metrics: Any | None = None, **kwargs: Any) -> None:
        del kwargs

        if self.main_process_only and jax.process_index() != 0:
            return

        now = self._time()

        if self._active:
            should_stop = False
            if self._trace_start_step is not None and step >= self._trace_start_step + self.profile_n_steps:
                should_stop = True
            if self._deadline is not None and now >= self._deadline:
                should_stop = True
            payload = metrics if metrics is not None else aux
            if should_stop:
                self._stop(payload)
            return

        should_start = False
        if step == self.profile_first_step:
            should_start = True
        elif self.profile_every_n_minutes >= 0:
            elapsed = now - self._last_profile_time
            if elapsed >= self.profile_every_n_minutes * 60:
                should_start = True

        if should_start:
            self._start(step)

    def on_training_end(self, **kwargs: Any) -> None:  # type: ignore[override]
        del kwargs
        self._stop(None)

    def on_validation_start(self, **kwargs: Any) -> None:  # type: ignore[override]
        del kwargs
        if self.main_process_only and jax.process_index() != 0:
            return
        self._stop(None)

    def _start(self, step: int) -> None:
        if self._active:
            LOGGER.debug("Profiler already active; start request ignored (step=%s)", step)
            return

        trace_path = self.log_path / f"profile-step-{step}"
        trace_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Starting JAX profiler trace at step %s (path=%s)", step, trace_path)
        try:
            jax.profiler.start_trace(str(trace_path))
        except Exception:
            LOGGER.exception("Failed to start JAX profiler trace")
            return

        self._active = True
        self._trace_start_step = step
        self._deadline = (
            self._time() + self.profile_max_seconds
            if self.profile_max_seconds is not None
            else None
        )

    def _stop(self, metrics: Any | None) -> None:
        if not self._active:
            return

        LOGGER.info("Stopping JAX profiler trace")
        if metrics is not None:
            _block_until_ready(metrics)
        try:
            jax.profiler.stop_trace()
        except Exception:
            LOGGER.exception("Failed to stop JAX profiler trace")
        self._active = False
        self._trace_start_step = None
        self._deadline = None
        self._last_profile_time = self._time()


__all__ = ["JaxProfiler"]
