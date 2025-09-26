"""Local wall-clock measurement utilities for training pipelines.

These helpers mirror the intent of MaxText's goodput recorder but avoid
external dependencies. They allow you to measure arbitrary regions of code
using context managers, aggregate wall-clock statistics, and optionally expose
per-step timings for logging or TensorBoard integration.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import numpy as np

try:  # Optional: JAX may not be available at import time for some tools.
    import jax
except Exception:  # pragma: no cover - fallback when JAX is unavailable.
    jax = None  # type: ignore


@dataclasses.dataclass
class _EventStats:
    """Keeps streaming statistics for a single event."""

    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = 0.0
    mean: float = 0.0
    m2: float = 0.0  # Sum of squares of differences from the mean (for variance).
    samples: deque[float] | None = None

    def add(self, duration: float) -> None:
        self.count += 1
        self.total += duration
        self.minimum = min(self.minimum, duration)
        self.maximum = max(self.maximum, duration)

        # Welford's online algorithm for mean and variance.
        delta = duration - self.mean
        self.mean += delta / self.count
        delta2 = duration - self.mean
        self.m2 += delta * delta2

        if self.samples is not None:
            self.samples.append(duration)

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def as_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "count": 0,
                "total": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        summary = {
            "count": float(self.count),
            "total": self.total,
            "mean": self.mean,
            "std": self.stddev,
            "min": self.minimum,
            "max": self.maximum,
        }

        if self.samples is not None and self.samples:
            # Use numpy for percentile calculations on the recent history.
            sample_array = np.asarray(self.samples, dtype=np.float64)
            summary.update(
                {
                    "p50": float(np.percentile(sample_array, 50)),
                    "p90": float(np.percentile(sample_array, 90)),
                    "p95": float(np.percentile(sample_array, 95)),
                    "p99": float(np.percentile(sample_array, 99)),
                }
            )

        return summary


class ProgramWallClock:
    """Collects wall-clock timings for labelled code regions.

    Use `ProgramWallClock.measure(..)` as a context manager around code regions
    you want to time. Per-event summaries accumulate automatically, and per-step
    metrics can be retrieved with `collect_pending_metrics` to feed into loggers
    or TensorBoard.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        main_process_only: bool = True,
        store_samples: bool = True,
        history_limit: int | None = 1024,
        time_fn: Callable[[], float] = time.perf_counter,
    ) -> None:
        process_index = jax.process_index() if jax is not None else 0
        self._enabled = enabled and (not main_process_only or process_index == 0)
        self._store_samples = store_samples
        self._history_limit = history_limit
        self._time_fn = time_fn

        self._event_stats: Dict[str, _EventStats] = {}
        self._pending: Dict[str, Dict[str, float]] = defaultdict(dict)

    def enabled(self) -> bool:
        return self._enabled

    def measure(
        self,
        name: str,
        *,
        mode: str | None = None,
        step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        """Returns a context manager that times a named region.

        Args:
            name: Logical identifier for the region (e.g. "data.next").
            mode: Optional higher-level grouping (e.g. "train" or "eval").
            step: Optional step index associated with this measurement.
            metadata: Extra annotations stored alongside the measurement.
        """

        if not self._enabled:
            return nullcontext()

        return _Timer(self, name=name, mode=mode, step=step, metadata=dict(metadata or {}))

    def record(
        self,
        name: str,
        duration: float,
        *,
        mode: str | None = None,
        step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self._enabled:
            return

        key = f"{mode}.{name}" if mode else name
        if key not in self._event_stats:
            samples = deque(maxlen=self._history_limit) if self._store_samples else None
            self._event_stats[key] = _EventStats(samples=samples)
        self._event_stats[key].add(duration)

        if mode is not None:
            pending = self._pending[mode]
            pending[name] = duration
            if step is not None:
                pending[f"{name}/step"] = float(step)
        if metadata:
            pending_meta = self._pending.setdefault("metadata", {})
            for meta_key, value in metadata.items():
                pending_meta[f"{mode}.{name}.{meta_key}" if mode else f"{name}.{meta_key}"] = value

    def collect_pending_metrics(self, mode: str, *, prefix: str = "time") -> Dict[str, float]:
        """Returns and clears pending per-step metrics for a mode."""
        if not self._enabled or mode not in self._pending:
            return {}
        payload = self._pending.pop(mode)
        return {f"{prefix}/{mode}/{name}": value for name, value in payload.items() if not name.endswith("/step")}

    def collect_metadata(self) -> Dict[str, Any]:
        if not self._enabled or "metadata" not in self._pending:
            return {}
        return self._pending.pop("metadata")

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {name: stats.as_dict() for name, stats in self._event_stats.items()}

    def reset(self) -> None:
        self._event_stats.clear()
        self._pending.clear()

    def now(self) -> float:
        return self._time_fn()

    def histogram(self, name: str, *, mode: str | None = None) -> Iterable[float]:
        """Return a copy of the recorded samples for `name` (if any)."""
        key = f"{mode}.{name}" if mode else name
        stats = self._event_stats.get(key)
        if stats is None or stats.samples is None:
            return []
        return list(stats.samples)


class _Timer:
    """Context manager produced by ProgramWallClock.measure."""

    __slots__ = ("_recorder", "_name", "_mode", "_step", "_metadata", "_start")

    def __init__(
        self,
        recorder: ProgramWallClock,
        *,
        name: str,
        mode: str | None,
        step: int | None,
        metadata: Mapping[str, Any],
    ) -> None:
        self._recorder = recorder
        self._name = name
        self._mode = mode
        self._step = step
        self._metadata = metadata
        self._start = 0.0

    def __enter__(self):
        self._start = self._recorder.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = self._recorder.now()
        duration = max(0.0, end - self._start)
        self._recorder.record(
            self._name,
            duration,
            mode=self._mode,
            step=self._step,
            metadata=self._metadata,
        )
        return False


__all__ = ["ProgramWallClock"]
