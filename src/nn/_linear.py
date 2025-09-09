import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, kaiming_normal, zeros
from jaxtyping import Array, PRNGKeyArray, Float

from src import Darray
from src.distributed import maybe_shard

from ._utils import promote_dtype


default_init = kaiming_normal()


class Linear(eqx.Module):
    weight: Darray
    bias: Darray | None
    in_features: int = eqx.field(static = True)
    out_features: int = eqx.field(static = True)
    use_bias: bool =  eqx.field(static = True)
    dtype: jnp.dtype = eqx.field(static=True)
    params_dtype: jnp.dtype = eqx.field(static=True)
    initializer: Initializer = eqx.field(static = True)
    input_pspec: jax.P | None = eqx.field(static = True)
    output_pspec: jax.P | None = eqx.field(static = True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        initializer: Initializer = None,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray = None,
        weight_spec: str | tuple[str, ...] | None = None,
        bias_spec: str | tuple[str, ...] | None = None,
        input_pspec: jax.P | None = None,
        output_pspec: jax.P | None = None,
    ):
        wkey , bkey = jax.random.split(key, 2)

        self.initializer = default_init if initializer is None else initializer

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.bias = None

        self.dtype = dtype
        self.params_dtype = params_dtype

        w = self.initializer(wkey, (out_features, in_features), dtype=self.params_dtype)
        self.weight = Darray(value=w, pspec=weight_spec)

        if use_bias:
            b = zeros(bkey, (out_features,), dtype=self.params_dtype)
            self.bias = Darray(value=b, pspec=bias_spec)

        self.input_pspec = input_pspec
        self.output_pspec = output_pspec

    def __call__(
        self,
        x: Float[Array, "... in_features"],
        /,
    ):
        weight = getattr(self.weight, "value", self.weight)
        w, x_ = promote_dtype(weight, x, dtype=self.dtype)
        x_ = maybe_shard(x_, self.input_pspec)
        # Support any leading axes by multiplying on the last dimension.
        # Shapes: x_[..., in] @ w.T[in, out] -> out[..., out]
        output = x_ @ w.T
        if self.use_bias:
            bias = getattr(self.bias, "value", self.bias) if self.bias is not None else None
            (b,) = promote_dtype(bias, dtype=self.dtype) if bias is not None else (None,)
            if b is not None:
                output = output + b

        return maybe_shard(output, self.output_pspec) 
