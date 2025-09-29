from __future__ import annotations

from typing import Mapping


class Logger:
    log_every_n_steps: int | None = None

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - base class
        ...

    def log(
        self,
        tag: str,
        metrics: Mapping[str, object] | None,
        *,
        step: int,
    ) -> Mapping[str, float] | None:
        raise NotImplementedError

    def log_scalars(
        self,
        tag: str,
        metrics: Mapping[str, object] | None,
        *,
        step: int,
    ) -> Mapping[str, float] | None:
        raise NotImplementedError

    def log_scalar(self, tag: str, metric: float, *, step: int) -> Mapping[str, float]:
        raise NotImplementedError

    def wc(self, name, step):
        raise NotImplementedError

    def log_hyperparams(self, params: Mapping[str, object] | None = None) -> None:
        raise NotImplementedError

    def finalize(self, status: str = "success") -> None:
        return None
