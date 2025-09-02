import math
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ._utils import promote_dtype


def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    mask: Bool[Array, "q_seq kv_seq"] | None = None,
) -> Float[Array, "q_seq kv_seq"]:
    # Compute in a promoted compute dtype to avoid integer/float mismatches.
    query, key = promote_dtype(query, key)
    query = query / math.sqrt(query.shape[-1])
    logits = jnp.einsum("sd,Sd->sS", query, key)
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(
                f"mask must have shape (query_seq_length, "
                f"kv_seq_length)=({query.shape[0]}, "
                f"{key.shape[0]}). Got {mask.shape}."
            )
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
        logits = cast(Array, logits)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(logits.dtype, jnp.float32)
    weights = jax.nn.softmax(logits.astype(dtype)).astype(logits.dtype)
    return weights


def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Bool[Array, "q_seq kv_seq"] | None = None,
    dropout = None,
    *,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "q_seq v_size"]:
    query, key_, value = promote_dtype(query, key_, value)
    weights = dot_product_attention_weights(query, key_, mask)
    if dropout is not None:
        weights = dropout(weights, key=key)
    attn = jnp.einsum("sS,Sd->sd", weights, value)
    return attn


def make_2D_attention_mask(seq_attention_mask: Int[Array, " seq_len"], attention_heads: int) -> Int[Array, " seq_len attention_heads seq_len"]:
    attn_mask = (seq_attention_mask[:, None] * seq_attention_mask[None, :]).astype(jnp.int32)
    attn_mask = jnp.broadcast_to(attn_mask[:, None, :], (attn_mask.shape[0], attention_heads, attn_mask.shape[1]))
    return attn_mask
