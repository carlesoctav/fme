from __future__ import annotations

from dataclasses import is_dataclass, fields
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from ._darray import Darray


InitFn = Callable[[jax.Array, tuple[int, ...], Any], jax.Array]


def _is_shape_dtype_struct(x: Any) -> bool:
    try:
        from jax import ShapeDtypeStruct
        return isinstance(x, ShapeDtypeStruct)
    except Exception:
        return False


def _default_weight_init() -> InitFn:
    return jax.nn.initializers.normal(stddev=0.02)


def _default_bias_init() -> InitFn:
    def zeros(key, shape, dtype):  # noqa: ARG001 - key unused
        return jnp.zeros(shape, dtype)
    return zeros


def _layernorm_weight_init() -> InitFn:
    def ones(key, shape, dtype):  # noqa: ARG001
        return jnp.ones(shape, dtype)
    return ones


def _choose_init(module: Any, field_name: str) -> InitFn:
    # 1) Module-specific initializer attribute (e.g., nn.Linear.initializer)
    init_attr = getattr(module, "initializer", None)
    if callable(init_attr) and field_name.lower() == "weight":
        return init_attr  # type: ignore[return-value]

    # 2) LayerNorm heuristics
    name = module.__class__.__name__.lower()
    if "layernorm" in name and field_name.lower() == "weight":
        return _layernorm_weight_init()
    if "layernorm" in name and field_name.lower() == "bias":
        return _default_bias_init()

    # 3) Bias default
    if field_name.lower() == "bias":
        return _default_bias_init()

    # 4) Fallback
    return _default_weight_init()


def _reseed(key: jax.Array, tag: str) -> jax.Array:
    # Deterministic per-leaf RNG via fold_in of hash(tag)
    return jax.random.fold_in(key, (hash(tag) & 0xFFFFFFFF))


def reinit_module(module: eqx.Module, *, rng: jax.Array) -> eqx.Module:
    """
    Reinitialize all Darray leaves under `module` using initializers inferred from context.

    Rules:
    - If the module has an `initializer` attribute and the field name is "weight", use it.
    - If the module looks like LayerNorm: weight -> ones, bias -> zeros.
    - If field name contains "bias": zeros.
    - Otherwise, normal(0.02).

    Preserves Darray.pspec on replacement.
    Also materializes Darray.value when it is a jax.ShapeDtypeStruct.
    """
    if not is_dataclass(module):
        return module

    updates: dict[str, Any] = {}
    for f in fields(module.__class__):
        try:
            val = getattr(module, f.name)
        except Exception:
            continue
        if isinstance(val, Darray):
            value = val.value
            if value is None:
                continue
            if isinstance(value, jax.Array):
                shape = value.shape
                dtype = value.dtype
            elif _is_shape_dtype_struct(value):
                shape = value.shape
                dtype = value.dtype
            else:
                # Not an array-like or ShapeDtypeStruct
                continue
            init = _choose_init(module, f.name)
            subkey = _reseed(rng, f.name)
            new_val = init(subkey, shape, dtype)
            updates[f.name] = Darray(value=new_val, pspec=val.pspec)
        elif is_dataclass(val) and isinstance(val, eqx.Module):
            # Recurse
            subkey = _reseed(rng, f.name + ":sub")
            new_sub = reinit_module(val, rng=subkey)
            if new_sub is not val:
                updates[f.name] = new_sub
        elif isinstance(val, (list, tuple)):
            changed = False
            new_seq = []
            for i, item in enumerate(val):
                if isinstance(item, eqx.Module):
                    subkey = _reseed(rng, f.name + f"[{i}]")
                    new_item = reinit_module(item, rng=subkey)
                    if new_item is not item:
                        changed = True
                    new_seq.append(new_item)
                else:
                    new_seq.append(item)
            if changed:
                updates[f.name] = type(val)(new_seq)
        # dicts skipped by default to keep traversal simple; add if needed

    if not updates:
        return module
    new_mod = module
    for k, v in updates.items():
        try:
            object.__setattr__(new_mod, k, v)
        except Exception:
            pass
    return new_mod


def materialize_abstract(module: eqx.Module, *, rng: jax.Array) -> eqx.Module:
    """
    Convenience alias: same as reinit_module but often used when parameters were abstract (ShapeDtypeStruct).
    """
    return reinit_module(module, rng=rng)

