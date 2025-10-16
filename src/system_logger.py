import logging
import os
import jax
import multiprocessing


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


def setup_logger(log_file="distributed.log"):
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logger = logging.getLogger("distributed_logger")
    logger.propagate = False

    formatter = CustomFormatter("%(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

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

