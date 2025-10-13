import logging
import multiprocessing
import os
from typing import Any, Protocol

import jax
import trackio

from ._utils import rank_zero


class Logger(Protocol):
    def log(self, *args, **kwargs):
        ...

class TrackioLogger(Logger):
    def __init__(
        self,
        project: str,
        name: str | None = None,
        space_id: str | None = None,
        space_storage: Any = None,
        dataset_id: str | None = None,
        config: dict | None = None,
        resume: str = "never",
        settings: Any = None,
        private: bool | None = None,
        embed: bool = True,
    ):
        self.name = project
        self.logger = rank_zero(
            trackio.init
        )(
            project=project,
            name = name,
            space_id=space_id,
            space_storage=space_storage,
            dataset_id=dataset_id,
            config=config,
            resume=resume,
            settings=settings,
            private=private,
            embed=embed
        )

    @rank_zero
    def log(self, logs, step, **kwargs):
        self.logger.log(logs, step = step, **kwargs)


    @rank_zero
    def finish(self,):
        self.logger.finish()




def get_multiprocess_index():
    identity = getattr(multiprocessing.current_process(), "_identity", None)
    return identity[0] if identity else 0


class CustomFormatter(logging.Formatter):
    def format(self, record):
        process_index = getattr(record, "process_index", jax.process_index())
        multiprocess_index = getattr(record, "multiprocess_index", get_multiprocess_index())
        pid = os.getpid()
        timing = self.formatTime(record, self.datefmt)
        filename = record.filename
        prefix = f"[{process_index},{multiprocess_index},{pid}] {timing} [{filename}] {record.levelname}"
        original = super().format(record)
        return f"{prefix} {original}"

def setup_logger(log_file="./train.log"):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logger = logging.getLogger("distributed_logger")
    logger.propagate = False

    formatter = CustomFormatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.handlers.clear()
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    process_index = jax.process_index()
    multiprocess_index = get_multiprocess_index()
    level = logging.DEBUG if (process_index == 0 and multiprocess_index == 0) else logging.ERROR
    logger.setLevel(level)

    return logging.LoggerAdapter(logger, {
        "process_index": process_index,
        "multiprocess_index": multiprocess_index,
    })

