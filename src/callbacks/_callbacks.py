from __future__ import annotations

import dataclasses as dc
import typing as tp


CallbackT = tp.TypeVar("CallbackT", bound="Callback")


class Callback:
    """Minimal callback base class with opt-in event hooks."""

    def on_training_start(
        self,
        module,
        optimizer
    ) -> None:
        pass

    def on_training_step_start(self, module, optimizer, batch, step) -> None:
        pass

    def on_training_step_end(self, module, optimizer, batch, aux, logs, step) -> None:
        pass

    def on_training_end(self, module, optimizer, batch, aux, logs, step ) -> None:
        pass

    def on_validation_start(self, module, optimizer) -> None:
        pass

    def on_validation_step_start(self, module, optimizer, batch, step) -> None:
        pass

    def on_validation_step_end(self, module, optimizer, batch, aux, logs, step) -> None:
        pass

    def on_validation_end(self, module, optimizer, aux, logs, step) -> None:
        pass

