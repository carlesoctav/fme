from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from transformers import PretrainedConfig

from ._utils import promote_dtype

def eager_dot_product_attention(
    query: Float[Array, "*B T N H "],
    key: Float[Array, "*B S K H"],
    value: Float[Array, "*B S K H"],
    mask: Bool[Array, "*B T N S"] | None = None,
    inference: bool = False,
    dropout_rate: float | None 0.0 
    dropout_mask: Bool[Array, "*B T N S"] | None = None, 
    **kwargs,
    dropout_key: PRNGKeyArray | None = None,
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

    if dropout is not None and dropout_rate > 0.0 and not inference:
        if dropout_mask is not None:
            weights = jnp.where(dropout_mask, weights, 0.0) / (1.0 - dropout_rate)

    attn = jnp.einsum("...tns,...snh -> ...tnh", weights, v)
    return attn




class AttetionModule(eqx.Module):
    attn_fn: tp.Callable
    _attn_implementation: str = eqx.field(static=True, default="eager") # same as config._attn_implementaiton
    implementation: str = eqx.field(static=True, default="xla") # this one is for tokamax

    def __call__(
        query: ArrayLike,
        key: ArrayLike,
        value: ArrayLike,
        bias: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        **kwargs,
        inference: bool = True,
        dropout_rate: float  = 0.0, 
        dropout_mask: ArrayLike | None = None,
        implementation: Literal['xla', 'cudnn'] | None = None,
        dropout_key: PRNGKeyArray | None = None,
    ) -> Array:
        ...


class SDPA(AttetionModule):
    def __init__(self, implementation: str | None = None):
        self.attn_fn = jax.nn.dot_product_attention
        self._attn_implementation = "sdpa" 
        self.implmentation = implementation

    def __call__(
        query: Array,
        key: Array,
        value: Array,
        bias: Array | None = None,
        mask: Array | None = None,
        **kwargs,
        inference: bool = False,
        dropout_rate: float  = 0.0, 
        dropout_mask: Array | None = None,
        dropout_key: PRNGKeyArray | None = None,
    ) -> Array:
        if dropout_rate > 0.0:
            raise NotImplementedError(" dropout on sdpa which use jax.nn.dot_product_attention is not implemented yet")

        return self.attn_fn(
            query,
            key,
            value,
            mask=mask,
            **kwargs,
            implementation=self.implementation,
        )



class EagerAttentionModule(AttetionModule):
    def __init__(self, implementation: str | None = None):
        self.attn_fn = eager_dot_product_attention
        self._attn_implementation = "eager"
        self.implmentation = implementation

    def __call__(
        query: Array,
        key: Array,
        value: Array,
        bias: Array | None = None,
        mask: Array | None = None,
        **kwargs,
        inference: bool = False,
        dropout_rate: float  = 0.0, 
        dropout_mask: Array | None = None,
        dropout_key: PRNGKeyArray | None = None,
    ) -> Array:
        return self.attn_fn(
            query,
            key,
            value,
            mask=mask,
            inference=inference,
            dropout_rate=dropout_rate,
            dropout_mask=dropout_mask,
            dropout_key=dropout_key,
            **kwargs,
        )

def make_attention_module(
    config: PretrainedConfig | None = None,
    implementation: str | None = None,
    dtype: jnp.dtype = jnp.float32
) -> AttentionModule:
    if config._attn_implementation == "sdpa":
        return SDPA(config=config, dtype = dtype, implementation=implementation)
    if config._attn_implementation == "eager":
        return EagerAttentionModule(config=config, dtype = dtype, implementation=implementation)

    raise ValueError(f"Unsupported attention type: {config.type}")

