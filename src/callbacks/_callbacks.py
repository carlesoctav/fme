from __future__ import annotations

import dataclasses as dc
import typing as tp


CallbackT = tp.TypeVar("CallbackT", bound="Callback")


class Callback:
    """Minimal callback base class with opt-in event hooks."""

    def on_training_start(self, **kwargs) -> None:
        pass

    def on_training_end(self, **kwargs) -> None:
        pass

    def on_train_step(self, **kwargs) -> None:
        pass

    def on_train_step_end(self, **kwargs) -> None:
        pass

    def on_eval_start(self, **kwargs) -> None:
        pass

    def on_eval_step(self, **kwargs) -> None:
        pass

    def on_eval_step_end(self, **kwargs) -> None:
        pass

    def on_eval_end(self, **kwargs) -> None:
        pass



@dc.dataclass
class CallbackManager:
    """Utility that fans out events to a list of callbacks."""

    callbacks: list[Callback] = dc.field(default_factory=list)

    def __post_init__(self) -> None:
        # Defensive copy so callers can pass tuples/iterables safely.
        self.callbacks = list(self.callbacks)

    def add(self, callback: CallbackT) -> CallbackT:
        self.callbacks.append(callback)
        return callback

    def extend(self, callbacks: tp.Iterable[Callback]) -> None:
        for cb in callbacks:
            self.add(cb)

    def call(self, event: str, /, **kwargs) -> None:
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method is None:
                continue
            method(**kwargs)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self.callbacks)
