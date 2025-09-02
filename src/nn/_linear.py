import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, kaiming_normal, zeros
from jaxtyping import PRNGKeyArray

from ._utils import promote_dtype


default_init = kaiming_normal()


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array | None
    in_features: eqx.field(static = True)
    out_features: eqx.field(static = True)
    use_bias: eqx.field(static = True)
    dtype: jnp.dtype = eqx.field(static=True)
    params_dtype: jnp.dtype = eqx.field(static=True)


    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        initializer: Initializer = None,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray = None
    ):
        wkey , bkey = jax.random.split(key, 2)
        if initializer is None:
            initializer = default_init

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.bias = None

        self.dtype = dtype
        self.params_dtype = params_dtype

        self.weight = initializer(wkey, (out_features, in_features), dtype=self.params_dtype)

        if use_bias:
            self.bias = zeros(bkey, (out_features,), dtype=self.params_dtype)

    def __call__(
        self,
        x: jax.Array,
    ):
        w, x_ = promote_dtype(self.weight, x, dtype=self.dtype)
        output = w @ x_
        if self.use_bias:
            (b,) = promote_dtype(self.bias, dtype=self.dtype) if self.bias is not None else (None,)
            if b is not None:
                output = output + b

        return output
