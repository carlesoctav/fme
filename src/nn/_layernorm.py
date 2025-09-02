import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import ones as ones_init, zeros as zeros_init
from jaxtyping import PRNGKeyArray

from ._utils import promote_dtype


Array = jax.Array


class LayerNorm(eqx.Module):
    weight: Array | None
    bias: Array | None
    normalized_shape: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    elementwise_affine: bool = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    params_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray, 
    ):
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.dtype = dtype
        self.params_dtype = params_dtype

        if self.elementwise_affine:
            wkey, bkey = jax.random.split(key, 2)
            wvalue = ones_init(wkey, normalized_shape, dtype=self.params_dtype)
            self.weight = wvalue 
            if bias:
                bvalue = zeros_init(bkey, normalized_shape, dtype=self.params_dtype)
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

        (x_,) = promote_dtype(x, dtype=self.dtype)
        mean = jnp.mean(x_, keepdims=True)
        var = jnp.var(x_, keepdims=True)
        inv = lax.rsqrt(jnp.maximum(var, 0.0) + jnp.asarray(self.eps, dtype=self.dtype))
        y = (x_ - mean) * inv

        if self.weight is not None:
            (w,) = promote_dtype(self.weight, dtype=self.dtype)
            y = w * y
        if self.bias is not None:
            (b,) = promote_dtype(self.bias, dtype=self.dtype)
            y = y + b

        return y
