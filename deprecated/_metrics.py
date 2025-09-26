from __future__ import annotations

import dataclasses as dc
import typing as tp


_Number = tp.Union[int, float]


@dc.dataclass
class _MetricAccumulator:
    total: float = 0.0
    count: float = 0.0

    def add(self, values: _Number, counts: _Number = 1) -> None:
        self.total += float(values)
        self.count += float(counts)

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0.0

    def state(self) -> dict[str, float]:
        return {"values": self.total, "counts": self.count}


class MetricsAgg:

    def __init__(self, **reducers: dict[str, str]) -> None:
        self._reducers: dict[str, str] = {k: v for k, v in reducers.items()}
        self._acc: dict[str, _MetricAccumulator] = {
            k: _MetricAccumulator() for k in self._reducers
        }

    def _ensure(self, key: str) -> _MetricAccumulator:
        if key not in self._acc:
            self._acc[key] = _MetricAccumulator()
            self._reducers.setdefault(key, "mean")
        return self._acc[key]

    def update(self, metrics: tp.Mapping[str, tp.Any] | None) -> None:
        if not metrics:
            return
        for name, payload in metrics.items():
            values, counts = _extract_values_counts(payload)
            self._ensure(name).add(values, counts)

    def compute(
        self,
        *,
        reset: bool = False,
        reduce: bool = True,
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        result: dict[str, tp.Any] = {}
        for name, acc in self._acc.items():
            raw = acc.state()
            if reduce:
                reducer = self._reducers.get(name, "mean")
                result[name] = _reduce(raw, reducer)
            else:
                result[name] = raw
        if reset:
            self.reset()
        return result

    def state(self) -> dict[str, dict[str, float]]:
        return {name: acc.state() for name, acc in self._acc.items()}

    def reset(self) -> None:
        for acc in self._acc.values():
            acc.reset()

    def __bool__(self) -> bool: 
        return any(acc.count > 0 for acc in self._acc.values())


def _extract_values_counts(payload: tp.Any) -> tuple[float, float]:
    if payload is None:
        return 0.0, 0.0
    if isinstance(payload, (tuple, list)) and len(payload) >= 2:
        return float(payload[0]), float(payload[1])
    if isinstance(payload, dict):
        values = payload.get("values", payload.get("value", 0.0))
        counts = payload.get("counts", payload.get("count", 1.0))
        return float(values), float(counts)
    if hasattr(payload, "values") and hasattr(payload, "counts"):
        return float(payload.values), float(payload.counts)
    return float(payload), 1.0


def _reduce(raw: dict[str, float], reducer: str) -> float:
    reducer = reducer.lower()
    values = float(raw.get("values", 0.0))
    counts = float(raw.get("counts", 0.0))
    if reducer == "mean":
        denom = counts if counts > 0 else 1.0
        return values / denom
    if reducer == "sum":
        return values
    if reducer == "count":
        return counts
    raise ValueError(f"Unknown reducer '{reducer}'")


__all__ = ["MetricsAgg"]
