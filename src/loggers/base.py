from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping


class Logger(ABC):
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], *, step: int, mode: str) -> bool:
        ...

    def log_hyperparams(self, params: Mapping[str, object] | None = None) -> None:
        return None

    def finalize(self, status: str = "success") -> None:
        return None
