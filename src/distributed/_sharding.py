from typing import Any

import jax.lax as lax
from equinox import combine, is_array, partition
from jax import P
from jaxtyping import PyTree


def filter_pspec(
    x: PyTree[Any],
    pspec: PyTree[P] 
):
    dynamic, static = partition(x, is_array)
    dynamic = lax.with_sharding_constraint(dynamic, pspec)
    return combine(dynamic, static)
