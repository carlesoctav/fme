import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field
from jax.nn.initializers import Initializer, normal
from jaxtyping import Array, Int

from ..modeling_utils import PrepareableModule, Rngs


default_init = normal(stddev=0.02)


class Embedding(PrepareableModule):
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
        rngs: Rngs,
        weight_spec: str | tuple[str, ...] | None = None,
    ):
        wkey = rngs.make_rng("params")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.initializer = default_init if initializer is None else initializer

        self.weight = self.initializer(wkey, (num_embeddings, embedding_dim))

    def __call__(
        self,
        x: Int[Array, " ..."],
        *,
        rngs: Rngs | None = None,
    ) -> Array:
        (x,) = self.maybe_prepare_input((x,))
        output = self.weight[x]
        return self.maybe_prepare_output(output)

    def init_weights(self, *, rngs: Rngs) -> "Embedding":
        new_w = self.initializer(
            rngs.make_rng("params"), (self.num_embeddings, self.embedding_dim)
        )
        new_self = eqx.tree_at(lambda m: m.weight, self, new_w)
        return new_self
