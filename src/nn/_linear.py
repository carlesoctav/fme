import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer, kaiming_normal, zeros
from jaxtyping import Array, Float, PRNGKeyArray

from src import Darray

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
    ):
        wkey , bkey = jax.random.split(key, 2)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        self.initializer = default_init if initializer is None else initializer
        self.bias = None
        self.dtype = dtype
        self.params_dtype = params_dtype


        w = self.initializer(wkey, (out_features, in_features), dtype=self.params_dtype)
        self.weight = Darray(value=w, pspec=weight_spec)

        if use_bias:
            b = zeros(bkey, (out_features,), dtype=self.params_dtype)
            self.bias = Darray(value=b, pspec=bias_spec)


    def __call__(
        self,
        x: Float[Array, "... in_features"],
        /,
    ):
        weight = getattr(self.weight, "value", self.weight)
        w, x_ = promote_dtype(weight, x, dtype=self.dtype)
        # Support any leading axes by multiplying on the last dimension.
        # Shapes: x_[..., in] @ w.T[in, out] -> out[..., out]
        output = x_ @ w.T
        if self.use_bias:
            bias = getattr(self.bias, "value", self.bias) if self.bias is not None else None
            (b,) = promote_dtype(bias, dtype=self.dtype) if bias is not None else (None,)
            if b is not None:
                output = output + b

        return output 

    def init_weights(self, *, key: PRNGKeyArray | None = None) -> "Linear": 
        if key is None:
            raise ValueError("A PRNGKeyArray 'key' must be provided.")

        k_w, k_b = jax.random.split(key, 2)
        w_shape = (self.out_features, self.in_features)
        w_dtype = self.params_dtype
        new_w = self.initializer(k_w, w_shape, dtype=w_dtype)
        new_bias = None
        if self.use_bias and self.bias is not None:
            b_shape = (self.out_features,)
            b_dtype = self.params_dtype
            new_bias = zeros(k_b, b_shape, dtype=b_dtype)

        new_self = self
        new_self = eqx.tree_at(lambda m: m.weight, new_self, Darray(value=new_w, pspec=self.weight.pspec))
        if self.use_bias and self.bias is not None:
            new_self = eqx.tree_at(lambda m: m.bias, new_self, Darray(value=new_bias, pspec=self.bias.pspec))
        return new_self
