from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
from equinox import field
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Float, PRNGKeyArray

from transformers import PretrainedConfig


def eager_dot_product_attention(
    query: Float[Array, "B T N H"],
    key: Float[Array, "B S K H"],
    value: Float[Array, "B S K H"],
    bias: Array | None = None,
    mask: Bool[Array, "B T N S"] | None = None,
    *,
    dropout_rate: float = 0.0,
    dropout_rng: PRNGKeyArray | None = None,
    broadcast_dropout: bool = True,
    **kwargs,
) -> Float[Array, "B T N H"]:
    query = query / jnp.sqrt(query.shape[-1])

    B, T, N, H = query.shape
    Bk, S, K, Hk = key.shape
    Bv, Sv, Kv, Hv = value.shape

    if Hk != H or Hv != H:
        raise ValueError("Query, key, and value must share the same head dimension")

    if Kv != K:
        raise ValueError("Value tensor must share the key head axis for attention")

    if K != N:
        if K <= 0 or N % K != 0:
            raise ValueError(
                "Number of query heads must be a positive multiple of key/value heads"
            )
        repeat_factor = N // K
        key = jnp.repeat(key, repeat_factor, axis=-2)
        value = jnp.repeat(value, repeat_factor, axis=-2)
        K = N

    scores = jnp.einsum("b t n h, b s n h -> b t n s", query, key)

    if mask is not None:
        mask_bool = jnp.asarray(mask, dtype=jnp.bool_)
        if mask_bool.ndim != scores.ndim:
            raise ValueError(
                f"Mask shape {mask_bool.shape} must match attention scores shape {scores.shape}"
            )
        neg_inf = jnp.array(jnp.finfo(scores.dtype).min, dtype=scores.dtype)
        scores = jnp.where(mask_bool, scores, neg_inf)

    with jax.numpy_dtype_promotion("standard"):
        dtype = jnp.result_type(scores.dtype, jnp.float32)
    weights = jax.nn.softmax(scores.astype(dtype), axis=-1).astype(scores.dtype)

    if dropout_rate > 0.0:
        if dropout_rng is None:
            raise TypeError("dropout_rate > 0 but no dropout_rng provided")
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            dropout_shape = list(weights.shape)
            if len(dropout_shape) >= 2:
                dropout_shape[1] = 1  # broadcast across query length
            keep = jax.random.bernoulli(dropout_rng, keep_prob, tuple(dropout_shape))
            keep = jnp.broadcast_to(keep, weights.shape)
        else:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, weights.shape)
        multiplier = keep.astype(weights.dtype) / keep_prob
        weights = weights * multiplier

    attn = jnp.einsum("btns, bsnh -> btnh", weights, value)
    return attn


class AttentionModule(eqx.Module):
    attn_fn: Callable = eqx.field(static=True)
    _attn_implementation: str = eqx.field(static=True, default="eager")
    implementation: str = eqx.field(static=True, default="xla")
    inference: bool = eqx.field(static=True, default=False)

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        bias: Array | None = None,
        mask: Array | None = None,
        *,
        dropout_rate: float = 0.0,
        implementation: Literal["xla", "cudnn"] | None = None,
        dropout_key: PRNGKeyArray | None = None,
        broadcast_dropout: bool = True,
        **kwargs,
    ) -> Array:
        raise NotImplementedError


class JaxNNAttentionModule(AttentionModule):
    def __init__(
        self,
        config: PretrainedConfig | None = None,
        implementation: str | None = None,
    ):
        self.attn_fn = jax.nn.dot_product_attention
        self._attn_implementation = "sdpa"
        self.implementation = implementation
        self.inference = False

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        bias: Array | None = None,
        mask: Array | None = None,
        *,
        dropout_rate: float = 0.0,
        dropout_key: PRNGKeyArray | None = None,
        broadcast_dropout: bool = True,
        **kwargs,
    ) -> Array:
        if self.inference:
            dropout_rate = 0.0

        if dropout_rate > 0.0:
            raise NotImplementedError(
                "dropout on sdpa which uses jax.nn.dot_product_attention is not implemented yet"
            )

        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None, :, :]
            elif mask.ndim == 4:
                mask = mask.transpose(0, 2, 1, 3)

        return self.attn_fn(
            query,
            key,
            value,
            mask=mask,
            **kwargs,
            implementation=self.implementation,
        )


class EagerAttentionModule(AttentionModule):
    broadcast_dropout: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        config: PretrainedConfig | None = None,
        implementation: str | None = None,
        broadcast_dropout: bool = True,
    ):
        self.attn_fn = eager_dot_product_attention
        self._attn_implementation = "eager"
        self.implementation = implementation
        self.inference = False
        self.broadcast_dropout = broadcast_dropout

    def __call__(
        self,
        query: Array,
        key: Array,
        value: Array,
        bias: Array | None = None,
        mask: Array | None = None,
        *,
        dropout_rate: float = 0.0,
        dropout_key: PRNGKeyArray | None = None,
        broadcast_dropout: bool | None = None,
        **kwargs,
    ) -> Array:
        if mask.ndim == 3:
            mask = mask[..., None, :]

        assert mask.ndim == 4, "Mask should be 4D"

        if self.inference:
            dropout_rate = 0.0

        return self.attn_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rate=dropout_rate,
            dropout_rng=dropout_key,
            broadcast_dropout=self.broadcast_dropout
            if broadcast_dropout is None
            else broadcast_dropout,
            **kwargs,
        )


def make_attention_module(config: PretrainedConfig | None = None) -> AttentionModule:
    implementation = (
        config.implementation if hasattr(config, "implementation") else "xla"
    )
    if config._attn_implementation == "sdpa":
        return JaxNNAttentionModule(config=config, implementation=implementation)
    if config._attn_implementation == "eager":
        return EagerAttentionModule(config=config, implementation=implementation)

    raise ValueError(f"Unsupported attention type: {config._attn_implementation}")
