import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.nn.initializers import ones as ones_init, zeros as zeros_init
from jaxtyping import PRNGKeyArray

from src import Darray
from src.distributed import maybe_shard

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
    input_pspec: jax.P | None = eqx.field(static = True)
    output_pspec: jax.P | None = eqx.field(static = True)

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
        weight_spec: str | tuple[str, ...] | None = None,
        bias_spec: str | tuple[str, ...] | None = None,
        input_pspec: jax.P | None = None,
        output_pspec: jax.P | None = None,
    ):
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.dtype = dtype
        self.params_dtype = params_dtype

        if self.elementwise_affine:
            wkey, bkey = jax.random.split(key, 2)
            wvalue = ones_init(wkey, normalized_shape, dtype=self.params_dtype)
            self.weight = Darray(value=wvalue, pspec=weight_spec)
            if bias:
                bvalue = zeros_init(bkey, normalized_shape, dtype=self.params_dtype)
                self.bias = Darray(value=bvalue, pspec=bias_spec) 
            else:
                self.bias = None
        else:
            self.weight = None
            self.bias = None

        self.input_pspec = input_pspec
        self.output_pspec = output_pspec


    def __call__(
        self,
        x: Array,
    ) -> Array:
        """Applies LayerNorm over the last `len(normalized_shape)` dims.

        Supports any leading axes. For example, if normalized_shape=(H,), inputs can be
        (.., H). If normalized_shape=(C,H), inputs can be (.., C, H).
        """
        (x_,) = promote_dtype(x, dtype=self.dtype)
        x_ = maybe_shard(x_, self.input_pspec)
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

        y = maybe_shard(y, self.output_pspec)
        return y
