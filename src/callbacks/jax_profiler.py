from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.tree_util as jtu

from ._callbacks import Callback
from .._utils import rank_zero


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
        log_dir: str | Path = "./log/jaxprofiler",
        profile_every_n_minutes: float = 300,
        profile_first_step: int = 10,
        profile_n_steps: int = 4,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.log_path = Path(log_dir)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.profile_every_n_minutes = profile_every_n_minutes
        self.profile_first_step = profile_first_step
        self.profile_n_steps = max(1, profile_n_steps)
        self._time = time_fn

        self._active = False
        self._last_profile_time = self._time()
        self._trace_start_step: int | None = None
        LOGGER.debug("JaxProfiler initialized: %s", self.log_path)

    @rank_zero
    def on_training_step(
        self,
        module,
        optimizer,
        batch,
        logs,
        logger,
        step_idx,
    ) -> None:
        now = self._time()

        if self._active:
            should_stop = False
            if (
                self._trace_start_step is not None
                and (step_idx - self._trace_start_step) >= self.profile_n_steps
            ):
                should_stop = True
            if should_stop:
                self._stop(module, optimizer, logger)
            return

        should_start = False
        if step_idx == self.profile_first_step:
            should_start = True
        elif self.profile_every_n_minutes >= 0:
            elapsed = now - self._last_profile_time
            if elapsed >= self.profile_every_n_minutes * 60:
                should_start = True

        if should_start:
            self._start(step_idx)

    @rank_zero
    def on_training_end(self, module, optimizer, logs, logger, step) -> None:
        self._stop(module, optimizer)

    @rank_zero
    def on_validation_start(self, module, optimizer, logger, step) -> None:
        self._stop(module, optimizer)

    def _start(self, step: int) -> None:
        if self._active:
            LOGGER.debug(
                "Profiler already active; start request ignored (step=%s)", step
            )
            return

        trace_path = self.log_path / f"profile-step-{step}"
        trace_path.mkdir(parents=True, exist_ok=True)
        LOGGER.info(
            "Starting JAX profiler trace at step %s (path=%s)", step, trace_path
        )
        try:
            jax.profiler.start_trace(str(trace_path))
        except Exception:
            LOGGER.exception("Failed to start JAX profiler trace")
            return

        self._active = True
        self._trace_start_step = step

    def _stop(self, module, optimizer) -> None:
        if not self._active:
            return

        LOGGER.info("Stopping JAX profiler trace")
        _block_until_ready(module)
        _block_until_ready(module)
        try:
            jax.profiler.stop_trace()
        except Exception:
            LOGGER.exception("Failed to stop JAX profiler trace")
        self._active = False
        self._trace_start_step = None
        self._deadline = None
        self._last_profile_time = self._time()


__all__ = ["JaxProfiler"]
