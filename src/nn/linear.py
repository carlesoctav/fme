import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, kaiming_normal, zeros
from jaxtyping import Array, Float, PRNGKeyArray

from ..module_utils import PrepareableModule


default_init = kaiming_normal()


class Linear(PrepareableModule):
    weight: Array
    bias: Array | None
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    initializer: Initializer = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        *,
        initializer: Initializer = None,
        key: PRNGKeyArray = None,
        weight_spec: str | tuple[str, ...] | None = None,
        bias_spec: str | tuple[str, ...] | None = None,
    ):
        wkey, bkey = jax.random.split(key, 2)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        self.initializer = default_init if initializer is None else initializer
        self.bias = None

        self.weight = self.initializer(wkey, (out_features, in_features))

        if use_bias:
            self.bias = zeros(bkey, (out_features,))

    def __call__(
        self,
        x: Float[Array, "... in_features"],
        *,
        key: PRNGKeyArray | None = None,
    ):
        (x,) = self.maybe_prepare_module((x,))

        output = x @ self.weight.T
        if self.use_bias and self.bias is not None:
            output = output + self.bias

        return self.maybe_prepare_output(output)

    def init_weights(self, *, key: PRNGKeyArray | None = None) -> "Linear":
        if key is None:
            raise ValueError("A PRNGKeyArray 'key' must be provided.")

        k_w, k_b = jax.random.split(key, 2)
        w_shape = (self.out_features, self.in_features)
        new_w = self.initializer(k_w, w_shape)

        new_self = eqx.tree_at(lambda m: m.weight, self, new_w)

        if self.use_bias and self.bias is not None:
            b_shape = (self.out_features,)
            new_bias = zeros(k_b, b_shape)
            new_self = eqx.tree_at(lambda m: m.bias, new_self, new_bias)

        return new_self
