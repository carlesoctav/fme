import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field
from jax.nn.initializers import Initializer, normal
from jaxtyping import Array, Int, PRNGKeyArray


default_init = normal(stddev=0.02)


class Embedding(eqx.Module):
    weight: Array
    num_embeddings: int = field(static=True)
    embedding_dim: int = field(static=True)
    initializer: Initializer = eqx.field(static=True)

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        initializer: Initializer = None,
        key: PRNGKeyArray,
        weight_spec: str | tuple[str, ...] | None = None,
    ):
        wkey, _ = jax.random.split(key, 2)

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.initializer = default_init if initializer is None else initializer

        self.weight = self.initializer(wkey, (num_embeddings, embedding_dim))

    def __call__(
        self,
        x: Int[Array, " ..."],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Array:
        return self.weight[x]

    def init_weights(self, *, key: PRNGKeyArray | None = None) -> "Embedding":
        if key is None:
            raise ValueError("A PRNGKeyArray 'key' must be provided.")

        new_w = self.initializer(key, (self.num_embeddings, self.embedding_dim))
        new_self = eqx.tree_at(lambda m: m.weight, self, new_w)
        return new_self
