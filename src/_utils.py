from __future__ import annotations

import functools
import typing as tp

import jax

from typing import TypeVar


A = TypeVar("A")


def first_from(*args: A | None, error_msg: str) -> A:
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError(error_msg)


def _is_rank_zero() -> bool:
    try:
        return jax.process_index() == 0
    except Exception:
        return True


def rank_zero(
    fn: tp.Callable[..., A] | None = None,
    /,
    *args: tp.Any,
    **kwargs: tp.Any,
) -> A | None | tp.Callable[..., A | None]:
    """Invoke or decorate ``fn`` so it executes only on JAX process rank 0."""

    def _call(callable_fn: tp.Callable[..., A], *call_args: tp.Any, **call_kwargs: tp.Any) -> A | None:
        if not _is_rank_zero():
            return None
        return callable_fn(*call_args, **call_kwargs)

    if fn is None:
        return lambda inner_fn: rank_zero(inner_fn)

    if args or kwargs:
        return _call(fn, *args, **kwargs)

    @functools.wraps(fn)
    def _wrapped(*wrapped_args: tp.Any, **wrapped_kwargs: tp.Any) -> A | None:
        return _call(fn, *wrapped_args, **wrapped_kwargs)

    return _wrapped


__all__ = ["first_from", "rank_zero"]
