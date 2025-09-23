from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from ._utils import promote_dtype


class AttentionModule(eqx.Module):
    dtype: jnp.dtype = eqx.field(static=True, default=jnp.float32)

    def __call__(
        self,
        query: Float[Array, "... q_size nheads head_size"],
        key_tensor: Float[Array, "... kv_size nheads head_size"],
        value: Float[Array, "... kv_size nheads head_size"],
        *,
        attention_mask: Int[Array, "... q_size"] | Bool[Array, "... q_size"] | None = None,
        segment_ids: Int[Array, "... q_size"] | None = None,
        dropout: Callable[[Array, PRNGKeyArray | None], Array],
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "... q_size nheads head_size"]:
        raise NotImplementedError


def make_attention_mask_from_segment(
    *,
    segment_ids: Int[Array, "... seq_len"],
    nheads: int,
    is_causal: bool,
) -> Bool[Array, "... seq_len nheads seq_len"]:
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
    q: Float[Array, "... q_size nheads head_size"],
    k: Float[Array, "... kv_size nheads head_size"],
    v: Float[Array, "... kv_size nheads head_size"],
    mask: Bool[Array, "... q_size nheads kv_size"] | None = None,
    dropout: Callable[[Array, PRNGKeyArray | None], Array] | None = None,
    *,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "... q_size nheads head_size"]:
    q, k, v = promote_dtype(q, k, v)
    q = q / jnp.sqrt(q.shape[-1])

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


def _default_attention_factory(**kwargs: Any) -> AttentionModule:
    values = dict(kwargs)
    values.setdefault("dtype", jnp.float32)
    return SDPA(**values)


_DEF_FACTORY: ContextVar[Callable[..., AttentionModule]] = ContextVar(
    "attention_factory",
    default=_default_attention_factory,
)


def _replace_attr(module: AttentionModule, name: str, value: Any) -> AttentionModule:
    if not hasattr(module, name):
        raise TypeError(f"Unexpected attention override parameter: {name}")
    try:
        return replace(module, **{name: value})
    except TypeError as exc:  # pragma: no cover - defensive: dataclass signature mismatch
        raise TypeError(f"Cannot override attribute '{name}' on {type(module).__name__}") from exc


@contextmanager
def attention_op(module: AttentionModule):
    base_module = module

    def factory(**module_kwargs: Any) -> AttentionModule:
        result = base_module
        for name, value in module_kwargs.items():
            result = _replace_attr(result, name, value)

        return result

    token = _DEF_FACTORY.set(factory)
    try:
        yield module
    finally:
        _DEF_FACTORY.reset(token)

def build_attention_module(**kwargs: Any) -> AttentionModule:
    factory = _DEF_FACTORY.get()
    return factory(**kwargs)


class SDPA(AttentionModule):
    """Standard scaled dot-product attention."""

    is_causal: bool = eqx.field(static=True, default=True)

    def __call__(
        self,
        q: Float[Array, "... q_size nheads head_size"],
        k: Float[Array, "... kv_size nheads head_size"],
        v: Float[Array, "... kv_size nheads head_size"],
        *,
        attention_mask: Int[Array, "... q_size"] | Bool[Array, "... q_size"] | None = None,
        segment_ids: Int[Array, "... q_size"] | None = None,
        dropout: Callable[[Array, PRNGKeyArray | None], Array],
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "... q_size nheads head_size"]:
        batch_shape = q.shape[:-3]
        q_len = q.shape[-3]
        nheads = q.shape[-2]
        kv_len = k.shape[-3]
        target_shape = batch_shape + (q_len, nheads, kv_len)

        mask: Bool[Array, "... q_size nheads kv_size"] | None = None

        if attention_mask is not None:
            attn_mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
            expected_query_shape = batch_shape + (q_len,)
            expected_dense_shape = batch_shape + (q_len, kv_len)

            if attn_mask.shape == target_shape:
                mask = attn_mask
            elif attn_mask.shape == expected_dense_shape:
                mask = jnp.expand_dims(attn_mask, axis=-2)
                mask = jnp.broadcast_to(mask, target_shape)
            elif attn_mask.shape == expected_query_shape:
                mask = jnp.expand_dims(attn_mask, axis=-1)
                mask = jnp.expand_dims(mask, axis=-1)
                mask = jnp.broadcast_to(mask, target_shape)
            else:
                raise ValueError(
                    "attention_mask shape "
                    f"{attn_mask.shape} is incompatible with expected "
                    f"shapes {expected_query_shape} or {expected_dense_shape}"
                )

        elif segment_ids is not None:
            mask = make_attention_mask_from_segment(
                segment_ids=segment_ids,
                nheads=nheads,
                is_causal=self.is_causal,
            )

        else:
            base = jnp.ones(batch_shape + (q_len, kv_len), dtype=jnp.bool_)
            if self.is_causal:
                base = jnp.tril(base)
            mask = jnp.expand_dims(base, axis=-2)
            mask = jnp.broadcast_to(mask, target_shape)

        return dot_product_attention(
            q,
            k,
            v,
            mask=mask,
            dropout=dropout,
            key=key,
        )

