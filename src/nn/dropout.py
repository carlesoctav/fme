from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import PRNGKeyArray

from ..module_utils import PrepareableModule


Array = jax.Array
A = TypeVar("A")


class Dropout(PrepareableModule):
    p: float = eqx.field(static=True)
    inference: bool = eqx.field(static=True)

    def __init__(
        self,
        p: float = 0.5,
        *,
        inference: bool = False,
        key: PRNGKeyArray | None = None,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")

        self.p = p
        self.inference = inference

    def __call__(
        self,
        x: Array,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        (x,) = self.maybe_prepare_module((x,))

        if self.inference or self.p == 0.0:
            return self.maybe_prepare_output(x)

        if not self.inference and key is None:
            raise RuntimeError(
                "Dropout requires a key when running in non-inference mode."
            )

        if self.p == 1.0:
            return jnp.zeros_like(x)

        keep_prob = 1.0 - lax.stop_gradient(self.p)
        mask = jax.random.bernoulli(key, keep_prob, shape=x.shape)
        output = jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))

        return self.maybe_prepare_output(output)
