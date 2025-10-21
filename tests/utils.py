from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def _to_jax_array(value) -> jnp.ndarray:
    import torch

    if isinstance(value, torch.Tensor):
        return jnp.asarray(value.detach().cpu().numpy())
    return jnp.asarray(np.asarray(value))


def update_module(module, getter: Callable[[Any], Any], new_value: Any):
    return eqx.tree_at(getter, module, new_value)


def update_linear(module, torch_linear):
    module = update_module(
        module, lambda m: m.weight.value, _to_jax_array(torch_linear.weight)
    )
    has_bias = (
        getattr(module, "use_bias", False) and getattr(module, "bias", None) is not None
    )
    if has_bias and torch_linear.bias is not None:
        module = update_module(
            module, lambda m: m.bias.value, _to_jax_array(torch_linear.bias)
        )
    return module


def update_embedding(module, torch_embedding):
    return update_module(
        module, lambda m: m.weight.value, _to_jax_array(torch_embedding.weight)
    )


def update_layernorm(module, torch_layernorm):
    if getattr(module, "weight", None) is not None:
        module = update_module(
            module, lambda m: m.weight.value, _to_jax_array(torch_layernorm.weight)
        )
    if getattr(module, "bias", None) is not None and torch_layernorm.bias is not None:
        module = update_module(
            module, lambda m: m.bias.value, _to_jax_array(torch_layernorm.bias)
        )
    return module


def assert_close(a, b, *, atol=1e-4, rtol=1e-4):
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def has_shape_dtype_struct(tree) -> bool:
    try:
        from jax import ShapeDtypeStruct
    except ImportError:  # pragma: no cover - Should not happen in tests
        return False

    found = False

    def _inspect(leaf):
        nonlocal found
        candidate = getattr(leaf, "value", leaf)
        if isinstance(candidate, ShapeDtypeStruct):
            found = True
        return leaf

    jax.tree_util.tree_map(_inspect, tree)
    return found


def t2np(t):
    import torch

    return t.detach().cpu().numpy()


def set_attr(module, path, value):
    parts = path.split(".")

    def getter(m):
        result = m
        for part in parts:
            if part.isdigit():
                result = result[int(part)]
            else:
                result = getattr(result, part)
        return result

    return eqx.tree_at(getter, module, jnp.asarray(value))
