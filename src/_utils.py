from __future__ import annotations

import contextlib
import functools
import typing as tp
import time

import jax
from typing import TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .loggers import Logger


A = TypeVar("A")


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
    logger: "Logger" | None = None,
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


__all__ = ["first_from", "rank_zero"]
