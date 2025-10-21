import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import (
    Initializer,
    ones as ones_init,
    zeros as zeros_init,
)
from jaxtyping import Float

from ..modeling_utils import PrepareableModule, Rngs


Array = jax.Array


class LayerNorm(PrepareableModule):
    weight: Array | None
    bias: Array | None
    normalized_shape: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    elementwise_affine: bool = eqx.field(static=True)
    initializer: Initializer = eqx.field(static=True)

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        initializer: Initializer = None,
        rngs: Rngs,
        weight_spec: str | tuple[str, ...] | None = None,
        bias_spec: str | tuple[str, ...] | None = None,
        input_pspec: jax.P | None = None,
        output_pspec: jax.P | None = None,
    ):
        self.normalized_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else normalized_shape
        )
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.initializer = ones_init if initializer is None else initializer

        if self.elementwise_affine:
            wkey = rngs.make_rng("params")
            bkey = rngs.make_rng("params")
            self.weight = self.initializer(wkey, normalized_shape, dtype=jnp.float32)
            if bias:
                self.bias = zeros_init(bkey, normalized_shape, dtype=jnp.float32)
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None

    def __call__(
        self,
        x: Float[Array, " *normalized_shape"],
        *,
        rngs: Rngs | None = None,
    ) -> Array:
        (x,) = self.maybe_prepare_input((x,))

        nd = x.ndim
        k = len(self.normalized_shape)

        if k == 0 or nd < k:
            raise ValueError(
                f"Input rank {nd} too small for normalized_shape {self.normalized_shape}"
            )

        if tuple(x.shape[-k:]) != tuple(self.normalized_shape):
            raise ValueError(
                f"Trailing shape {x.shape[-k:]} does not match normalized_shape {self.normalized_shape}"
            )

        axes = tuple(range(nd - k, nd))
        mean = jnp.mean(x, axis=axes, keepdims=True)
        var = jnp.var(x, axis=axes, keepdims=True)
        inv = lax.rsqrt(jnp.maximum(var, 0.0) + self.eps)
        y = (x - mean) * inv

        if self.weight is not None:
            y = self.weight * y
        if self.bias is not None:
            y = y + self.bias

        return self.maybe_prepare_output(y)

    def init_weights(self, *, rngs: Rngs) -> "LayerNorm":
        w_key = rngs.make_rng("params")
        b_key = rngs.make_rng("params")

        if not self.elementwise_affine:
            return self

        w_shape = self.normalized_shape

        new_w = self.initializer(w_key, w_shape, dtype=jnp.float32)
        new_self = eqx.tree_at(
            lambda m: m.weight,
            self,
            new_w,
        )

        if self.bias is not None:
            b_shape = self.normalized_shape
            new_b = zeros_init(b_key, b_shape, dtype=jnp.float32)
            new_self = eqx.tree_at(lambda m: m.bias, new_self, new_b)

        return new_self
