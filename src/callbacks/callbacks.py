from __future__ import annotations
import typing as tp


class Callback(tp.Protocol):
    def on_training_start(self, module, optimizer, logger) -> None:
        pass

    def on_training_step(self, module, optimizer, batch, logs, logger, step) -> None:
        pass

    def on_training_end(self, module, optimizer, logs, logger, step) -> None:
        pass

    def on_validation_start(self, module, optimizer, logger, step) -> None:
        pass

    def on_validation_step(
        self, module, optimizer, batch, logs, logger, step
    ) -> None:
        pass

    def on_validation_end(self, module, optimizer, logs, logger, step) -> None:
        pass
