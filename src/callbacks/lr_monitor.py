from ._callbacks  import Callback
import typing as tp 

class LearningRateMonitor:
    def __init__(
        self,
        log_every_n_step: int,
        schedule_fn: tp.Callable[[int], float],
    ):
        self.log_every_n_step = log_every_n_step
        self.schedule_fn = schedule_fn


    def on_training_step(self, model, optimizer, batch, logs, logger, step):
        if step % self.log_every_n_step == 0:
            logger.log({"lr": float(self.schedule_fn(step))}, step = step)

