import equinox as eqx
import jax
from jax.nn.initializers import Initializer, normal, zeros, kaiming_normal
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

default_init = kaiming_normal()


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array | None
    in_features: eqx.field(static = True)
    out_features: eqx.field(static = True)
    use_bias: eqx.field(static = True)


    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        initializer: Initializer = None,
        dtype = jnp.float32,
        key: PRNGKeyArray = None
    ):
        wkey , bkey = jax.random.split(key, 2)
        if initializer is None:
            initializer = default_init

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.bias = None

        self.weight = initializer(wkey, (out_features, in_features), dtype = dtype)

        if use_bias:
            self.bias = zeros(bkey, (out_features,), dtype=dtype)

    def __call__(
        self,
        x: jax.Array,
    ):
        output = self.weight @ x
        if self.use_bias:
            output = output + self.bias

        return output
