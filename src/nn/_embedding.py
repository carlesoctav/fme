import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field
from jax.nn.initializers import Initializer, normal
from jaxtyping import Array, Int, PRNGKeyArray

from src import Darray


default_init = normal(stddev=0.02)

class Embedding(eqx.Module):
    weight: Darray
    num_embeddings: int = field(static=True) 
    embedding_dim: int = field(static=True)
    dtype: jnp.dtype = field(static=True)
    params_dtype: jnp.dtype = field(static=True)
    initializer: Initializer = eqx.field(static = True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        initializer: Initializer = None,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
        weight_spec: str | tuple[str, ...] | None = None,
    ):
        wkey, _ = jax.random.split(key, 2)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.initializer = default_init if initializer is None else initializer
        self.dtype = dtype
        self.params_dtype = params_dtype

        wvalue = self.initializer(
            wkey, (num_embeddings, embedding_dim), self.params_dtype
        )

        self.weight = Darray(value=wvalue, pspec=weight_spec)


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

    def init_weights(self, *, key: PRNGKeyArray | None = None) -> "Embedding":
        if key is None:
            raise ValueError("A PRNGKeyArray 'key' must be provided.")

        new_w = self.initializer(key, (self.num_embeddings, self.embedding_dim), self.params_dtype)
        new_self = eqx.tree_at(lambda m: m.weight, self, Darray(value=new_w, pspec=self.weight.pspec))
        return new_self
