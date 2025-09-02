import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import ones as ones_init, zeros as zeros_init
from jaxtyping import DTypeLike, PRNGKeyArray

Array = jax.Array


class LayerNorm(eqx.Module):
    weight: Array | None
    bias: Array | None
    normalized_shape: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    elementwise_affine: bool = eqx.field(static=True)

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype = jnp.float16,
        key: PRNGKeyArray | None = None,
    ):
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps

        if self.elementwise_affine:
            wkey, bkey = jax.random.split(key, 2)
            wvalue = ones_init(wkey, normalized_shape, dtype = dtype)
            self.weight = wvalue 
            if bias:
                bvalue = zeros_init(bkey, normalized_shape, dtype = dtype)
                self.bias = bvalue 
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None


    def __call__(
        self,
        x: Array,
    ) -> Array:
        if x.shape != self.normalized_shape:
            raise ValueError(
                f"Input shape {x.shape} does not match normalized shape {self.normalized_shape}"
            )

        # Compute statistics in float32 for numerical stability (esp. with fp16 params).
        x32 = x.astype(jnp.float32)
        mean = jnp.mean(x32, keepdims=True)
        var = jnp.var(x32, keepdims=True)
        inv = lax.rsqrt(jnp.maximum(var, 0.0) + jnp.asarray(self.eps, dtype=jnp.float32))
        y32 = (x32 - mean) * inv

        if self.weight is not None:
            y32 = self.weight.astype(jnp.float32) * y32
        if self.bias is not None:
            y32 = y32 + self.bias.astype(jnp.float32)

        return y32.astype(x.dtype)
