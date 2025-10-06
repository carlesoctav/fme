from typing import Protocol, Any
import trackio
import jax
from ._utils import rank_zero


class Logger(Protocol):
    def log(self, *args, **kwargs):
        ...

class TrackioLogger(Logger):
    def __init__(
        self,
        project: str,
        space_id: str | None = None,
        space_storage: Any = None,
        dataset_id: str | None = None,
        config: dict | None = None,
        resume: str = "never",
        settings: Any = None,
        private: bool | None = None,
        embed: bool = True,
    ):
        self.logger = rank_zero(
            trackio.init
        )(
            project,
            space_id,
            space_storage,
            dataset_id,
            config,
            resume,
            settings,
            private,
            embed
        )

    @rank_zero
    def log(self, logs, step, **kwargs):
        self.logger(logs, step = step, **kwargs)
