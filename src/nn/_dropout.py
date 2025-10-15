from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import PRNGKeyArray

from ._utils import promote_dtype


Array = jax.Array
A = TypeVar('A')



class Dropout(eqx.Module):
    p: float = eqx.field(static=True)
    inference: bool = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    params_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        p: float = 0.5,
        *,
        inference: bool = False,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray | None = None,
    ):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        
        self.p = p
        self.inference = inference
        self.dtype = dtype
        self.params_dtype = params_dtype

    def __call__(
        self, 
        x: Array, 
        *, 
        key: PRNGKeyArray | None = None,
    ) -> Array:


        if self.inference or self.p == 0.0: 
            return x 

        if not self.inference and key is None:
            raise RuntimeError(
                "Dropout requires a key when running in non-inference mode."
            )

        if self.p == 1.0:
            (x_,) = promote_dtype(x, dtype=self.dtype)
            return jnp.zeros_like(x_)
        
        (x_,) = promote_dtype(x, dtype=self.dtype)
        keep_prob = 1.0 - lax.stop_gradient(jnp.asarray(self.p, dtype=self.dtype))
        mask = jax.random.bernoulli(key, keep_prob, shape=x_.shape)
        output = jax.lax.select(mask, x_/keep_prob, jnp.zeros_like(x_))
        
        return output
