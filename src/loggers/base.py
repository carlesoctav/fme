from __future__ import annotations

from typing import Mapping


class Logger:
    def __init__(self, *args, **kwargs) -> None:
        ...

    def log_scalars(self, metrics: dict[str, object], step: int, tag: str) -> dict[str, float]:
        raise NotImplementedError

    def log_scalar(self, name: str, metric: float, step: int, tag: str)-> dict[str, float]: 
        raise NotImplementedError

    def wc(self, name, step):
        raise NotImplementedError

    def log_hyperparams(self, params: Mapping[str, object] | None = None) -> None:
        raise NotImplementedError

    def finalize(self, status: str = "success") -> None:
        return None
