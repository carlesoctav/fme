from typing import TypeVar
import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import PRNGKeyArray

Array = jax.Array
A = TypeVar('A')


def first_from(*args: A | None, error_msg: str) -> A:
  for arg in args:
    if arg is not None:
      return arg
  raise ValueError(error_msg)

class Dropout(eqx.Module):
    p: float 
    inference: bool 

    def __init__(
        self,
        p: float = 0.5,
        *,
        inference: bool = False,
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
        inference: bool | None = None
    ) -> Array:

        inference = first_from(
            inference,
            self.inference,
            error_msg="""No `inference` argument was provided to Dropout 
                as either a __call__ argument or class attribute""",
        )

        if inference: 
            return x 

        if not inference and key is None:
            raise RuntimeError(
                "Dropout requires a key when running in non-inference mode."
            )

        if self.p == 1.0:
            return jnp.zeros_like(x)
        
        keep_prob = 1.0 - lax.stop_gradient(self.p) 
        mask = jax.random.bernoulli(key, keep_prob, shape=x.shape)
        output = jnp.where(mask, x / keep_prob, 0.0)
        
        return output

