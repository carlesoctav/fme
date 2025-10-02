from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ._utils import promote_dtype


@dataclass(frozen=True)
class AttentionConfig:
    type: str = "eager"
    is_causal: bool = True


def make_attention_mask_from_segment(
    *,
    segment_ids: Int[Array, "*B T"],
    nheads: int,
    is_causal: bool,
) -> Bool[Array, "*B T N T"]:
    if segment_ids.ndim < 1:
        raise ValueError("segment_ids must have at least one dimension for the sequence axis")

    segment_ids = jnp.asarray(segment_ids)
    valid = segment_ids != 0

    same_segment = segment_ids[..., :, None] == segment_ids[..., None, :]
    mask = same_segment & valid[..., :, None] & valid[..., None, :]
    if is_causal:
        mask = jnp.tril(mask)

    batch_shape = segment_ids.shape[:-1]
    seq_len = segment_ids.shape[-1]
    mask = jnp.expand_dims(mask, axis=-2)
    mask = jnp.broadcast_to(mask, (*batch_shape, seq_len, nheads, seq_len))
    return mask.astype(jnp.bool_)


def dot_product_attention(
    q: Float[Array, "*B T N H "],
    k: Float[Array, "*B S K H"],
    v: Float[Array, "*B S K H"],
    mask: Bool[Array, "*B T N S"] | None = None,
    dropout: Callable[[Array, PRNGKeyArray | None], Array] | None = None,
    *,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "... q_size q_heads head_size"]:

    q, k, v = promote_dtype(q, k, v)
    q = q / jnp.sqrt(q.shape[-1])

    *_, T, N, H = q.shape
    *_, S, K, _ = k.shape


    if k.shape[-1] != H or v.shape[-1] != H:
        raise ValueError("Query, key, and value must share the same head dimension")

    if v.shape[-2] != K:
        raise ValueError("Value tensor must share the key head axis for attention")

    if K != N:
        if K <= 0 or N % K != 0:
            raise ValueError(
                "Number of query heads must be a positive multiple of key/value heads"
            )
        repeat_factor = N // K
        k = jnp.repeat(k, repeat_factor, axis=-2)
        v = jnp.repeat(v, repeat_factor, axis=-2)
        K = N

    scores = jnp.einsum("...tnh, ...snh -> ...tns", q, k)

    if mask is not None:
        mask_bool = jnp.asarray(mask, dtype=jnp.bool_)
        if mask_bool.shape != scores.shape:
            raise ValueError(
                f"Mask shape {mask_bool.shape} must match attention scores shape {scores.shape}"
            )
        neg_inf = jnp.array(jnp.finfo(scores.dtype).min, dtype=scores.dtype)
        scores = jnp.where(mask_bool, scores, neg_inf)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(scores.dtype, jnp.float32)
    weights = jax.nn.softmax(scores.astype(dtype), axis=-1).astype(scores.dtype)

    if dropout is not None:
        weights = dropout(weights, key=key)

    attn = jnp.einsum("...tns,...snh -> ...tnh", weights, v)
    return attn


class AttentionModule(eqx.Module):
    def __call__(
        self,
        q: Float[Array, "*B T N H"],
        k: Float[Array, "*B S K H"],
        v: Float[Array, "*B S K H"],
        dropout: Callable[[Array, PRNGKeyArray | None], Array],
        attention_mask: Int[Array, "*B T N S"] | Bool[Array, "*B T"] | None = None,
        segment_ids: Int[Array, "*B T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "*B T N H"]:
        raise NotImplementedError


class SDPA(AttentionModule):
    """Standard scaled dot-product attention."""

    config: AttentionConfig
    dtype: jnp.dtype = eqx.field(static=True, default=jnp.float32)

    def __call__(
        self,
        q: Float[Array, "*B T N H"],
        k: Float[Array, "*B S K H"],
        v: Float[Array, "*B S K H"],
        dropout: Callable[[Array, PRNGKeyArray | None], Array],
        attention_mask: Int[Array, "*B T N S"] | Bool[Array, "*B T"] | None = None,
        segment_ids: Int[Array, "*B T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "*B T N H"]:

        *B, T, N, _ = q.shape
        *_, S, K, _ = k.shape
        B = tuple(B)


        if k.shape[:-3] != B or v.shape[:-3] != B:
            raise ValueError("Query, key, and value must share batch dimensions")

        if v.shape[-3] != S or v.shape[-2] != K:
            raise ValueError("Key and value must share their sequence and head dimensions")

        if k.shape[-1] != q.shape[-1] or v.shape[-1] != q.shape[-1]:
            raise ValueError("Query, key, and value must share the same head dimension")

        if K <= 0 or N % K != 0:
            raise ValueError(
                "Number of query heads must be a positive multiple of key/value heads"
            )

        group_size = N // K

        target_shape = B + (T, N, S)
        kv_target_shape = B + (T, K, S)

        mask: Bool[Array, "... q_size q_heads kv_size"] | None = None

        if attention_mask is not None:
            attn_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
            expected_query_shape = B + (T,)
            expected_dense_shape = B + (T, S)

            if attn_mask.shape == target_shape:
                mask = attn_mask
            elif group_size > 1 and attn_mask.shape == kv_target_shape:
                mask = jnp.repeat(attn_mask, group_size, axis=-2)
            elif attn_mask.shape == expected_dense_shape:
                mask = jnp.broadcast_to(attn_mask[..., :, None, :], target_shape)
            elif attn_mask.shape == expected_query_shape:
                mask = jnp.broadcast_to(attn_mask[..., :, None, None], target_shape)
            else:
                raise ValueError(
                    "attention_mask shape "
                    f"{attn_mask.shape} is incompatible with expected "
                    f"shapes {expected_query_shape} or {expected_dense_shape}"
                )

        elif segment_ids is not None:
            mask = self.make_attention_mask_from_segment(
                segment_ids=segment_ids,
                nheads=N,
                is_causal=self.config.is_causal,
            )

        else:
            base = jnp.ones(B + (T, S), dtype=jnp.bool_)
            if self.config.is_causal:
                base = jnp.tril(base)
            
            mask = jnp.broadcast_to(base[..., :, None, :], target_shape)

        return dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout=dropout,
            key=key,
        )

    def make_attention_mask_from_segment(
        self,
        *,
        segment_ids: Int[Array, "*B T"],
        nheads: int,
        is_causal: bool,
    ) -> Bool[Array, "*B T N T"]:
        if segment_ids.ndim < 1:
            raise ValueError("segment_ids must have at least one dimension for the sequence axis")

        segment_ids = jnp.asarray(segment_ids)
        valid = segment_ids != 0

        same_segment = segment_ids[..., :, None] == segment_ids[..., None, :]
        mask = same_segment & valid[..., :, None] & valid[..., None, :]
        if is_causal:
            mask = jnp.tril(mask)

        batch_shape = segment_ids.shape[:-1]
        seq_len = segment_ids.shape[-1]
        mask = jnp.expand_dims(mask, axis=-2)
        mask = jnp.broadcast_to(mask, (*batch_shape, seq_len, nheads, seq_len))
        return mask.astype(jnp.bool_)


def make_attention_module(
    config: AttentionConfig | None = None,
    dtype: jnp.dtype = jnp.float32
) -> AttentionModule:
    if config.type == "eager":
        return SDPA(config=config,dtype = dtype)
    raise ValueError(f"Unsupported attention type: {config.type}")

