from __future__ import annotations

from typing import TypeVar


A = TypeVar("A")


def first_from(*args: A | None, error_msg: str) -> A:
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError(error_msg)
