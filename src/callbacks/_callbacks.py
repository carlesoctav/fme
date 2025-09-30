from __future__ import annotations


class Callback:
    def on_training_start(self, module, optimizer, logger) -> None:
        pass

    def on_training_step(self, module, optimizer, batch, metric, logger, step) -> None:
        pass

    def on_training_end(self, module, optimizer, metric, logger, step) -> None:
        pass

    def on_validation_start(self, module, optimizer, logger) -> None:
        pass

    def on_validation_step(
        self, module, optimizer, batch, metric, logger, step
    ) -> None:
        pass

    def on_validation_end(self, module, optimizer, eval_metrics, logger, step) -> None:
        pass
