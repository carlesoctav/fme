import equinox as eqx
import jax.numpy as jnp
import jax
from jaxtyping import PRNGKeyArray, Int
from equinox import field, is_array_like
from jax.nn.initializers import normal

Array = jax.Array

default_init = normal(stddev=0.02)

class Embedding(eqx.Module, strict=True):
    weight: Array
    num_embeddings: int = field(static=True)  # vocab size
    embedding_dim: int = field(static=True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        dtype = jnp.float16,
        key: PRNGKeyArray,
    ):

        wkey, _ = jax.random.split(key, 2)
        wvalue = default_init(wkey, (num_embeddings, embedding_dim), dtype) 

        self.weight = wvalue 
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim


    def __call__(
        self,
        x: Int[Array, " seq_len"],  # noqa: F722
    ) -> Array:
        if is_array_like(x) and jnp.shape(x) == ():
            return self.weight[x]
        else:
            raise ValueError(
                "`Embedding()(x)` should be called with a scalar index `x`. "
            )
