from __future__ import annotations

import copy
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Float, Int, PRNGKeyArray
from transformers import ModernBertConfig

from ... import nn
from ..._utils import first_from
from ...nn import AttentionConfig, AttentionModule, make_rope_init_fn, make_attention_module


class ModernBertRotaryEmbedding(eqx.Module):
    rtheta: Array
    attention_scaling: Array
    rope_type: str = field(static=True)
    rope_init_fn: Any = field(static=True)
    dtype: jnp.dtype = field(static=True)
    param_dtype: jnp.dtype = field(static=True)

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32
    ):
        self.rope_type = "default"
        if isinstance(getattr(rope_config, "rope_scaling", None), dict):
            self.rope_type = rope_config.rope_scaling.get("rope_type", rope_config.rope_scaling.get("type", "default"))

        self.dtype = dtype
        self.rope_init_fn = make_rope_init_fn(self.rope_type)
        self.config = rope_config

        self.param_dtype = param_dtype
        self.dtype = dtype
        self.rtheta, self.attention_scaling = self.rope_init_fn(rope_config, seq_len=max_position_embeddings, dtype=self.dtype)


    def __call__(
        self,
        hidden_states: Float[Array, "*B T H D"],
        position_ids: Int[Array, "*B T"] | None = None,
    ) -> tuple[Float[Array, "*B T H D"], Float[Array, "*B T H D"]]:
        if hidden_states.shape != k.shape:
            raise ValueError("Query and key must share the same shape for rotary embeddings.")

        *B, T, N, H = hidden_states.shape

        expected_pos_shape = tuple(batch_shape) + (T,)

        if position_ids is None:
            position_ids = jnp.arange(T, dtype=jnp.int32)
            position_ids = position_ids[None, :]
        else:
            position_ids = jnp.asarray(position_ids, dtype=jnp.int32)

            if position_ids.shape != expected_pos_shape:
                raise ValueError(
                    "position_ids shape must match the leading batch dimensions of the inputs "
                    f"({expected_pos_shape}), got {position_ids.shape}."
                )

        rtheta = jnp.take(self.rtheta, position_ids, axis = 0) #(*B, T, halfdim)
        rtheta = rtheta[..., None, :]   #(*B, T, 1, halfdim)

        tensor_pairs = hidden_states.reshape(tuple(B) + (T, N, 2, H //2 )) #(*B, T, N, 2, halfdim)
        tensor_complex = tensor_pairs[..., 0] + 1j * tensor_pairs[..., 1]
        rotated = tensor_complex * freqs #(*B, T, N, halfdim)
        rotated = jnp.stack([jnp.real(rotated), jnp.imag(rotated)], axis=-1)
        rotated = rotated.reshape(hidden_states.shape) * self.attention_scaling

        return rotated 


class ModernBertAttention(eqx.Module):
    Wqkv: nn.Linear
    Wo: nn.Linear
    out_drop: nn.Dropout
    sdpa: AttentionModule
    rotary_embed: ModernBertRotaryEmbedding
    num_attention_heads: int = field(static=True)
    attention_head_size: int = field(static=True)
    all_head_size: int = field(static=True)
    local_attention: int = field(static=True)
    attention_config: AttentionConfig = field(static=True)
    config: ModernBertConfig | None = field(static=True, default=None)

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        attention_config: AttentionConfig | None = None,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        store_config: bool = True,
        key: PRNGKeyArray,
    ):
        qkv_key, out_key = jax.random.split(key, 2)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "Hidden size must be divisible by the number of attention heads "
                f"({config.hidden_size=} vs {config.num_attention_heads=})."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size


        self.Wqkv = nn.Linear(
            config.hidden_size,
            3 * config.hidden_size,
            use_bias=config.attention_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=qkv_key,
        )
        self.Wo = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            use_bias=config.attention_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=out_key,
        )

        self.out_drop = nn.Dropout(p=config.attention_dropout, dtype=dtype, params_dtype=params_dtype)

        attention_config = first_from(
            attention_config,
            AttentionConfig(type="eager", is_causal=False),
            error_msg="ModernBertAttention requires an attention_config",
        )

        self.attention_config = attention_config

        use_global_attention = (layer_id % max(config.global_attn_every_n_layers, 1)) == 0
        if use_global_attention:
            copy_config = copy.deepcopy(config)
            rope_theta = copy_config.global_rope_theta
            max_positions = copy_config.max_position_embeddings
            self.local_attention = (-1, -1)
            attention_config.is_causal = False
            attention_config.is_local_attention = False
        else:
            self.local_attention = (config.local_attention // 2, config.local_attention // 2)
            attention_config.is_lcoal_attention = False
            attention_config.window_size = (self.local_attention[0], self.local_attention[1])

        self.sdpa = make_attention_module(config=attention_config, dtype=dtype)
        self.rotary_embed = ModernBertRotaryEmbedding(
            config,
            dtype=dtype,
            param_dtype=params_dtype,
        )

    def _make_attention_mask(
        self,
        attention_mask: Int[Array, "*B T"] | Float[Array, "*B T"] | None,
        *,
        batch_size: int,
        seq_len: int,
    ) -> Array:
        if attention_mask is None:
            mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
        else:
            mask = jnp.asarray(attention_mask, dtype=jnp.bool_)
            if mask.ndim == 1:
                mask = jnp.broadcast_to(mask[None, :], (batch_size, seq_len))
            elif mask.ndim == 2:
                if mask.shape[0] == 1 and batch_size > 1:
                    mask = jnp.broadcast_to(mask, (batch_size, seq_len))
                elif mask.shape[0] != batch_size:
                    raise ValueError(
                        "attention_mask batch dimension mismatch: "
                        f"expected {batch_size}, got {mask.shape[0]}"
                    )
            else:
                mask = mask.reshape((-1, mask.shape[-1]))
                if mask.shape[0] != batch_size:
                    raise ValueError(
                        "attention_mask batch dimension mismatch: "
                        f"expected {batch_size}, got {mask.shape[0]}"
                    )

        base = jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
        if self.local_attention >= 0:
            positions = jnp.arange(seq_len)
            distance = jnp.abs(positions[:, None] - positions[None, :])
            window = distance <= self.local_attention
            base = base & window

        allowed = base[None, :, :] & mask[:, :, None] & mask[:, None, :]
        allowed = allowed[:, :, None, :]
        return jnp.broadcast_to(allowed, (batch_size, seq_len, self.num_attention_heads, seq_len))

    def __call__(
        self,
        hidden_states: Float[Array, "*B T D"],
        attention_mask: Int[Array, "*B T"] | Float[Array, "*B T"] | None = None,
        segment_ids: Int[Array, "*B T"] | None = None,
        position_ids: Int[Array, "*B T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "*B T D"]:
        attn_key, out_key = jax.random.split(key, 2) if key is not None else (None, None)

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        def _to_heads(x):
            return x.reshape(*x.shape[:-1], self.num_attention_heads, self.attention_head_size)

        q_heads = _to_heads(q)
        k_heads = _to_heads(k)
        v_heads = _to_heads(v)

        q_heads, k_heads = self.rotary_embed(q_heads, k_heads, position_ids=position_ids)

        batch_size = q_heads.shape[0]
        seq_len = q_heads.shape[-3]
        attn_mask = self._make_attention_mask(
            attention_mask,
            batch_size=batch_size,
            seq_len=seq_len,
        )

        attn_heads = self.sdpa(
            q_heads,
            k_heads,
            v_heads,
            dropout=self.out_drop,
            attention_mask=attn_mask,
            segment_ids=segment_ids,
            key=attn_key,
        )

        attn_output = attn_heads.reshape(*attn_heads.shape[:-2], self.all_head_size)
        attn_output = self.Wo(attn_output)
        attn_output = self.out_drop(attn_output, key=out_key)
        return attn_output
