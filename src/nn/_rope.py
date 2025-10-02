from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from jaxtyping import Array
from transformers import PretrainedConfig
from .._utils import first_from


RotaryInitFn = Callable[[Any, jnp.dtype], Tuple[Array, Array]]

def _base_angles(dim: int, base: float, dtype: jnp.dtype) -> Array:
    half_dim = dim // 2
    exponent = jnp.arange(half_dim, dtype=dtype) * (2.0 / dim)
    return 1.0 / (base ** exponent)


def r_from_inv(inv_freq: Array, seq_len: int, dtype: jnp.dtype) -> Array:
    positions = jnp.arange(seq_len, dtype=dtype) #(seq_len)
    angles = positions[:, None] * inv_freq[None, :] #(seq_len, dim/2) (m theta)
    return jnp.exp(1j * angles.astype(jnp.float32)) # (seq_len, dim/2) ( cos(theta) +  i sin(theata) )


def default_rope(
    config: PretrainedConfig,
    dtype: jnp.dtype = jnp.float32,
) -> Tuple[Array, float]:
    """
    compute R_{o,m}^d
    """

    base = config.rope_theta
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads

    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = _base_angles(dim, base, dtype)
    seq_len = first_from(
        seq_len,
        getattr( config, "max_position_embeddings", None ),
        error_msg="Either seq_len or config.max_position_embeddings must be set for RoPE initialization.",
    )

    r_theta = r_from_inv(inv_freq, seq_len, dtype)
    return r_theta, jnp.asarray(1.0, dtype=dtype)


ROPE_INIT_FUNCTIONS: Dict[str, RotaryInitFn] = {
    "default": default_rope,
}


def make_rope_init_fn(rope_type: str) -> RotaryInitFn:
    try:
        return ROPE_INIT_FUNCTIONS[rope_type]
    except KeyError as err:
        raise KeyError(f"Unsupported RoPE type '{rope_type}'. Available: {sorted(ROPE_INIT_FUNCTIONS)}") from err
