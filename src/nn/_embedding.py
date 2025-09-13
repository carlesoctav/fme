import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field, is_array_like
from jax.nn.initializers import normal
from jaxtyping import Array, Int, PRNGKeyArray

from src import Darray


default_init = normal(stddev=0.02)

class Embedding(eqx.Module):
    weight: Darray
    num_embeddings: int = field(static=True) 
    embedding_dim: int = field(static=True)
    dtype: jnp.dtype = field(static=True)
    params_dtype: jnp.dtype = field(static=True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
        weight_spec: str | tuple[str, ...] | None = None,
        output_pspec: jax.P | None = None,
        input_pspec: jax.P | None = None
    ):
        self.dtype = dtype
        self.params_dtype = params_dtype

        wkey, _ = jax.random.split(key, 2)
        wvalue = default_init(
            wkey, (num_embeddings, embedding_dim), self.params_dtype
        )

        self.weight = Darray(value=wvalue, pspec=weight_spec)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input_pspec = input_pspec
        self.output_pspec = output_pspec


    def __call__(
        self,
        x: Int[Array, " ..."]
    ) -> Array:
        """Lookup embeddings for arbitrary leading axes of indices.

        If x has shape (...,), returns (..., embedding_dim).
        """
        weight = getattr(self.weight, "value", self.weight)
        out = weight[x].astype(self.dtype)
        return out 
