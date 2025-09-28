"""Utilities for scheduling JAX profiler traces inside the training loop."""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - allow import without JAX installed (e.g., docs).
    import jax
except Exception:  # pragma: no cover
    jax = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class _ModeConfig:
    name: str
    enabled: bool
    start_step: int
    profile_steps: int
    repeat_every_steps: Optional[int]
    start_after_seconds: Optional[float]
    repeat_after_seconds: Optional[float]
    max_profile_seconds: Optional[float]


@dataclasses.dataclass
class _ModeState:
    config: _ModeConfig
    next_step_trigger: Optional[int]
    last_start_time: Optional[float]
    has_triggered: bool = False


class JaxProfiler:
    """Controls when to start and stop `jax.profiler` traces.

    This mirrors the behaviour of MaxText's profiler controller but runs purely
    locally, with both step-based and time-based scheduling options. The class
    supports distinct settings for training and evaluation phases and can be
    disabled on non-primary hosts when running multi-process jobs.
    """

    def __init__(
        self,
        logdir: str | Path,
        *,
        enabled: bool = True,
        main_process_only: bool = True,
        train_start_step: int = 0,
        train_profile_steps: int = 5,
        train_repeat_every_steps: int | None = None,
        train_start_after_seconds: float | None = None,
        train_repeat_after_seconds: float | None = None,
        train_max_profile_seconds: float | None = None,
        trace_during_eval: bool = False,
        eval_start_step: int = 0,
        eval_profile_steps: int = 2,
        eval_repeat_every_steps: int | None = None,
        eval_start_after_seconds: float | None = None,
        eval_repeat_after_seconds: float | None = None,
        eval_max_profile_seconds: float | None = None,
        timestamp_fn = time.time,
    ) -> None:
        process_index = jax.process_index() if jax is not None else 0
        self._enabled = enabled and (not main_process_only or process_index == 0)
        self._logdir = Path(logdir)
        self._logdir.mkdir(parents=True, exist_ok=True)
        self._timestamp_fn = timestamp_fn
        self._creation_time = self._timestamp_fn()
        self._active_mode: str | None = None
        self._active_stop_step: Optional[int] = None
        self._active_deadline: Optional[float] = None

        self._train_state = _ModeState(
            config=_ModeConfig(
                name="train",
                enabled=self._enabled,
                start_step=train_start_step,
                profile_steps=max(1, train_profile_steps),
                repeat_every_steps=train_repeat_every_steps,
                start_after_seconds=train_start_after_seconds,
                repeat_after_seconds=train_repeat_after_seconds,
                max_profile_seconds=train_max_profile_seconds,
            ),
            next_step_trigger=train_start_step,
            last_start_time=None,
        )
        self._eval_state = _ModeState(
            config=_ModeConfig(
                name="eval",
                enabled=self._enabled and trace_during_eval,
                start_step=eval_start_step,
                profile_steps=max(1, eval_profile_steps),
                repeat_every_steps=eval_repeat_every_steps,
                start_after_seconds=eval_start_after_seconds,
                repeat_after_seconds=eval_repeat_after_seconds,
                max_profile_seconds=eval_max_profile_seconds,
            ),
            next_step_trigger=eval_start_step,
            last_start_time=None,
        )

        self._last_trace_path: Optional[Path] = None

    def reset_eval_window(self) -> None:
        """Resets evaluation scheduling at the beginning of an eval loop."""
        self._eval_state.next_step_trigger = self._eval_state.config.start_step
        self._eval_state.last_start_time = None
        self._eval_state.has_triggered = False

    def enabled(self) -> bool:
        return self._enabled

    def reset_train_window(self) -> None:
        """Allows manual re-arming of the training trace schedule."""
        self._train_state.next_step_trigger = self._train_state.config.start_step
        self._train_state.last_start_time = None
        self._train_state.has_triggered = False

    def maybe_start_train(self, step: int, *, metadata: Optional[dict] = None) -> bool:
        return self._maybe_start("train", step=step, metadata=metadata)

    def maybe_stop_train(self, step: int, *, block_on: Optional[object] = None) -> bool:
        return self._maybe_stop("train", step=step, block_on=block_on)

    def on_eval_start(self, *, force_stop: bool = True) -> None:
        if force_stop and self._active_mode == "train":
            self.stop()
        self.reset_eval_window()

    def maybe_start_eval(self, eval_step: int, *, metadata: Optional[dict] = None) -> bool:
        return self._maybe_start("eval", step=eval_step, metadata=metadata)

    def maybe_stop_eval(self, eval_step: int, *, block_on: Optional[object] = None) -> bool:
        return self._maybe_stop("eval", step=eval_step, block_on=block_on)

    def stop(self, *, block_on: Optional[object] = None) -> bool:
        if not self._enabled or self._active_mode is None:
            return False

        mode = self._active_mode

        if block_on is not None and jax is not None:
            try:
                jax.block_until_ready(block_on)
            except Exception:  # pragma: no cover - defensive, avoid masking stop.
                LOGGER.exception("Failed to block on object before stopping profiler")

        try:
            if jax is not None:
                jax.profiler.stop_trace()
        except Exception:  # pragma: no cover
            LOGGER.exception("Failed to stop JAX profiler trace")
            return False

        LOGGER.info("Stopped JAX profiler trace (mode=%s, path=%s)", mode, self._last_trace_path)
        self._active_mode = None
        self._active_stop_step = None
        self._active_deadline = None

        state = self._train_state if mode == "train" else self._eval_state
        if state.config.repeat_every_steps is None and state.config.repeat_after_seconds is None:
            state.next_step_trigger = None

        return True

    def _maybe_start(self, mode: str, *, step: int, metadata: Optional[dict]) -> bool:
        if not self._enabled:
            return

        state = self._train_state if mode == "train" else self._eval_state
        cfg = state.config
        if not cfg.enabled:
            return
        if self._active_mode is not None:
            return False
        if state.has_triggered and cfg.repeat_every_steps is None and cfg.repeat_after_seconds is None:
            return False

        now = self._timestamp_fn()
        if cfg.start_after_seconds is not None and (now - self._creation_time) < cfg.start_after_seconds:
            return False
        if state.last_start_time is not None and cfg.repeat_after_seconds is not None:
            if (now - state.last_start_time) < cfg.repeat_after_seconds:
                return False
        if state.next_step_trigger is not None and step < state.next_step_trigger:
            return False

        trace_dir = self._allocate_trace_dir(mode=mode, step=step, metadata=metadata)
        started = self._start_trace(trace_dir)
        if not started:
            return False

        state.last_start_time = now
        state.has_triggered = True
        self._active_mode = mode
        self._active_stop_step = step + cfg.profile_steps - 1
        self._active_deadline = (
            now + cfg.max_profile_seconds if cfg.max_profile_seconds is not None else None
        )

        if cfg.repeat_every_steps is not None:
            state.next_step_trigger = step + cfg.repeat_every_steps
        else:
            state.next_step_trigger = None

        LOGGER.info(
            "Started JAX profiler trace (mode=%s, step=%d, duration_steps=%d, trace_dir=%s)",
            mode,
            step,
            cfg.profile_steps,
            trace_dir,
        )
        return True

    def _maybe_stop(self, mode: str, *, step: int, block_on: Optional[object]) -> bool:
        if not self._enabled or self._active_mode != mode:
            return False

        should_stop = False
        if self._active_stop_step is not None and step >= self._active_stop_step:
            should_stop = True
        if not should_stop and self._active_deadline is not None:
            if self._timestamp_fn() >= self._active_deadline:
                should_stop = True

        if not should_stop:
            return False

        return self.stop(block_on=block_on)

    def _allocate_trace_dir(self, *, mode: str, step: int, metadata: Optional[dict]) -> Path:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._timestamp_fn()))
        suffix_parts = [f"{mode}-step{step}", timestamp]
        if metadata:
            for key, value in metadata.items():
                suffix_parts.append(f"{key}-{value}")
        subdir = "-".join(suffix_parts)
        trace_dir = self._logdir / mode / subdir
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._last_trace_path = trace_dir
        return trace_dir

    def _start_trace(self, trace_dir: Path) -> bool:
        if jax is None:
            LOGGER.warning("JAX is unavailable; profiler trace request ignored")
            return False
        try:
            jax.profiler.start_trace(str(trace_dir))
        except Exception:  # pragma: no cover
            LOGGER.exception("Failed to start JAX profiler trace")
            return False
        return True


__all__ = ["JaxProfiler"]
