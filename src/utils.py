import contextlib
import functools
import logging
import time
import typing as tp

import jax
import jax.numpy as jnp


LOGGER = logging.getLogger(__name__)


if tp.TYPE_CHECKING:
    from .loggers import Logger


A = tp.TypeVar("A")
K = tp.TypeVar("K")
V = tp.TypeVar("V")


def rank_zero(fn: tp.Callable[..., A]) -> tp.Callable[..., A | None]:
    @functools.wraps(fn)
    def _wrapped(*args: tp.Any, **kwargs: tp.Any) -> A | None:
        if not jax.process_index() == 0:
            return None
        return fn(*args, **kwargs)

    return _wrapped


@contextlib.contextmanager
def wallclock(
    name: str,
    logger: "Logger",
    step: int | None = None,
    noop: bool = False,
):
    try:
        t0 = time.monotonic()
        yield
    finally:
        diff = time.monotonic() - t0
        if not noop:
            logger.log({f"time/{name}": diff}, step=step)


def first_from(*args: A | None, error_msg: str) -> A:
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError(error_msg)


class GeneralInterface(tp.MutableMapping[K, V], tp.Generic[K, V]):
    """
    Dict-like object keeping track of a class-wide mapping, as well as a local one. Allows to have library-wide
    modifications though the class mapping, as well as local modifications in a single file with the local mapping.
    """

    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key: K) -> V:
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: tp.Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())


def print_memory(compiled_stats):
    """Prints a summary of the compiled memory statistics."""

    if compiled_stats is None:
        return

    def bytes_to_gb(num_bytes):
        return num_bytes / (1024**3)

    output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
    temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
    argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
    alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
    host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
    total_gb = output_gb + temp_gb + argument_gb - alias_gb
    print(
        f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
        f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB."
    )


def is_in_jit() -> bool:
    return isinstance(jnp.zeros((), dtype=jnp.float32), jax.core.Tracer)


__all__ = [
    "first_from",
    "rank_zero",
    "GeneralInterface",
    "wallclock",
    "print_memory",
    "is_in_jit",
]
