from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ._utils import promote_dtype


def dot_product_attention_weights(
    query: Float[Array, "... q_size nheads head_size"],
    key: Float[Array, "... kv_size nheads head_size"],
    mask: Bool[Array, "... q_size nheads kv_size"] | None = None,
) -> Float[Array, "... q_size nheads kv_size"]:
    query, key = promote_dtype(query, key)
    query = query / jnp.sqrt(query.shape[-1])

    logits = jnp.einsum("...tnh, ...snh -> ...tns", query, key)

    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(f"Mask shape {mask.shape} must match logits shape {logits.shape}")

        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        logits = cast(Array, logits)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(logits.dtype, jnp.float32)
    weights = jax.nn.softmax(logits.astype(dtype)).astype(logits.dtype)
    return weights


def dot_product_attention(
    query: Float[Array, "... q_size nheads head_size"], 
    key_: Float[Array, "... kv_size nheads head_size"], 
    value: Float[Array, "... kv_size nheads head_size"], 
    mask: Bool[Array, "... q_size nheads kv_size"] | None = None, 
    dropout = None,
    *,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "... q_size nheads head_size"]: 
    query, key_, value = promote_dtype(query, key_, value)
    weights = dot_product_attention_weights(query, key_, mask)
    if dropout is not None:
        weights = dropout(weights, key=key)

    attn = jnp.einsum("...tns,...snh -> ...tnh", weights, value)
    return attn


def make_4D_attention_mask(
    seq_attention_mask: Int[Array, "... seq_length"],
    attention_heads: int,
) -> Int[Array, "... seq_length attention_heads seq_length"]:
    # Base 2D mask (..., T, S)
    base = (seq_attention_mask[..., :, None] * seq_attention_mask[..., None, :]).astype(jnp.int32)
    # Insert head axis to match (..., T, N, S)
    expanded = base[..., :, None, :]
    attn_mask = jnp.broadcast_to(
        expanded, (*base.shape[:-2], base.shape[-2], attention_heads, base.shape[-1])
    )
    return attn_mask
