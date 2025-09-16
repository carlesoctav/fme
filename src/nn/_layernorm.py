import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import (
    Initializer,
    ones as ones_init,
    zeros as zeros_init,
)
from jaxtyping import Float, PRNGKeyArray

from src import Darray

from ._utils import promote_dtype


Array = jax.Array


class LayerNorm(eqx.Module):
    weight: Darray | None
    bias: Darray | None
    normalized_shape: tuple[int, ...] = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    elementwise_affine: bool = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    params_dtype: jnp.dtype = eqx.field(static=True)
    initializer: Initializer = eqx.field(static = True) 

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        initializer: Initializer = None,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray, 
        weight_spec: str | tuple[str, ...] | None = None,
        bias_spec: str | tuple[str, ...] | None = None,
        input_pspec: jax.P | None = None,
        output_pspec: jax.P | None = None,
    ):
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.initializer = ones_init if initializer is None else initializer
        self.dtype = dtype
        self.params_dtype = params_dtype

        if self.elementwise_affine:
            wkey, bkey = jax.random.split(key, 2)
            wvalue = self.initializer(wkey, normalized_shape, dtype=self.params_dtype)
            self.weight = Darray(value=wvalue, pspec=weight_spec)
            if bias:
                bvalue = zeros_init(bkey, normalized_shape, dtype=self.params_dtype)
                self.bias = Darray(value=bvalue, pspec=bias_spec) 
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None



    def __call__(
        self,
        x: Float[Array, " *normalized_shape"],
    ) -> Array:
        """
        Applies LayerNorm over the last `len(normalized_shape)` dims.
        """
        (x_,) = promote_dtype(x, dtype=self.dtype)
        nd = x_.ndim
        k = len(self.normalized_shape)

        if k == 0 or nd < k:
            raise ValueError(
                f"Input rank {nd} too small for normalized_shape {self.normalized_shape}"
            )

        if tuple(x_.shape[-k:]) != tuple(self.normalized_shape):
            raise ValueError(
                f"Trailing shape {x_.shape[-k:]} does not match normalized_shape {self.normalized_shape}"
            )

        axes = tuple(range(nd - k, nd))
        mean = jnp.mean(x_, axis=axes, keepdims=True)
        var = jnp.var(x_, axis=axes, keepdims=True)
        inv = lax.rsqrt(jnp.maximum(var, 0.0) + jnp.asarray(self.eps, dtype=self.dtype))
        y = (x_ - mean) * inv

        if self.weight is not None:
            weight = getattr(self.weight, "value", self.weight)
            (w,) = promote_dtype(weight, dtype=self.dtype)
            y = w * y
        if self.bias is not None:
            bias = getattr(self.bias, "value", self.bias)
            (b,) = promote_dtype(bias, dtype=self.dtype)
            y = y + b

        return y

    def init_weights(self, *, key: PRNGKeyArray | None = None) -> "LayerNorm":
        if key is None:
            raise ValueError("A PRNGKeyArray 'key' must be provided.")

        w_key, b_key = jax.random.split(key, 2)

        if not self.elementwise_affine:
            return self

        w_shape = self.normalized_shape
        w_dtype = self.params_dtype

        new_w = self.initializer(w_key, w_shape, dtype=w_dtype)
        new_self = eqx.tree_at(lambda m: m.weight, self, Darray(value=new_w, pspec=self.weight.pspec if self.weight is not None else None))

        if self.bias is not None:
            b_shape = self.normalized_shape
            b_dtype = self.params_dtype
            new_b = zeros_init(b_key, b_shape, dtype=b_dtype)
            new_self = eqx.tree_at(lambda m: m.bias, new_self, Darray(value=new_b, pspec=self.bias.pspec))

        return new_self
