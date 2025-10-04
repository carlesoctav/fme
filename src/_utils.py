from __future__ import annotations

import contextlib
import functools
import typing as tp

import jax


if tp.TYPE_CHECKING:
    from .loggers import Logger


A = tp.TypeVar("A")
K = tp.TypeVar("K")
V = tp.TypeVar("V")


# @contextlib.contextmanager
# def wallclock(
#     name: str,
#     logger: "Logger" | None = None,
#     step: int | None = None,
# ):
#     try:
#         t0 = time.monotonic()
#         yield
#     finally:
#         diff = time.monotonic() - t0
#         if logger:
#             rank_zero(logger.log_scalar)(name, diff, step)

def wallclock(
    name: str,
    logger: Logger | None = None,
    step: int | None = None,
):
    return contextlib.nullcontext()
    # return contextlib.nullcontext()


def first_from(*args: A | None, error_msg: str) -> A:
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError(error_msg)







def rank_zero(fn: tp.Callable[..., A]) -> tp.Callable[..., A | None]:
    """Decorate ``fn`` so it executes only on JAX process rank 0."""

    @functools.wraps(fn)
    def _wrapped(*args: tp.Any, **kwargs: tp.Any) -> A | None:
        if not jax.process_index() == 0:
            return None
        return fn(*args, **kwargs)

    return _wrapped



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

__all__ = ["first_from", "rank_zero"]
