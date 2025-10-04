from __future__ import annotations

import copy
import dataclasses
import warnings
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import transformers
from equinox import field
from jax.nn.initializers import normal, ones as ones_init, zeros as zeros_init
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from transformers import ModernBertConfig

from ... import nn
from ..._darray import DArray
from ..._huggingface import HuggingFaceCompatibleModule
from ..._utils import first_from
from ...nn import (
    AttentionConfig,
    AttentionModule,
    eager_dot_product_attention,
    make_attention_module,
    make_rope_init_fn,
)

class ModernBertRotaryEmbedding(eqx.Module):
    rtheta: Array
    attention_scaling: Array
    rope_type: str = field(static=True)
    rope_init_fn: Any = field(static=True)
    dtype: jnp.dtype = field(static=True)

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray = None,
    ):
        self.rope_type = "default"
        if isinstance(getattr(config, "rope_scaling", None), dict):
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type", "default")
            )

        self.dtype = dtype
        self.rope_init_fn = make_rope_init_fn(self.rope_type)
        self.config = config

        self.param_dtype = param_dtype
        self.dtype = dtype
        self.rtheta, self.attention_scaling = self.rope_init_fn(
            config, dtype=self.dtype
        )

    def __call__(
        self,
        hidden_states: Float[Array, "B T N H"],
        position_ids: Int[Array, "B T"] | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
    ) -> Float[Array, "B T N H"]:
        *B, T, N, H = hidden_states.shape
        B = tuple(B)

        if segment_ids is not None:
            raise NotImplementedError("Sequence packing with RoPE is not implemented.")

        expected_pos_shape = B + (T,)

        if position_ids is None:
            position_ids = jnp.arange(T, dtype=jnp.int32)
            position_ids = position_ids[None, :]  # (1, T)
            position_ids = jnp.broadcast_to(position_ids, expected_pos_shape)  # (*B, T)
        else:
            if position_ids.shape != expected_pos_shape:
                raise ValueError(
                    f"position_ids must have shape {expected_pos_shape}, but got {position_ids.shape}"
                )

        # rtheta (max_seq, halfdim)
        rtheta = jnp.take(self.rtheta, position_ids, axis=0)  # (*B, T, halfdim)
        rtheta = rtheta[..., None, :]  # (*B, T, 1, halfdim)

        tensor_pairs = hidden_states.reshape(
            (B, T, N, H // 2, 2)
        )  # (*B, T, N, halfdim, 2)
        tensor_complex = (
            tensor_pairs[..., 0] + 1j * tensor_pairs[..., 1]
        )  # (*B, T, N, halfdim) but complex
        rotated = (
            tensor_complex * rtheta
        )  # (*B, T, N, halfdim)  (*B, T, 1, halfdim) -> (*B, T, N, halfdim)
        ## (a+ ib) (c + id) = (ac - bd) + i(ad + bc) here c = cos(t*angle), d = sin(t*angle)

        rotated = jnp.stack(
            [jnp.real(rotated), jnp.imag(rotated)], axis=-1
        )  # (*B, T, N, halfdim, 2)
        rotated = (
            rotated.reshape(hidden_states.shape) * self.attention_scaling
        )  # (*B, T, N, H)
        return rotated


def _get_activation(name: str) -> Callable[[Array], Array]:
    name = name.lower()
    mapping = {
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
        "silu": jax.nn.silu,
        "swish": jax.nn.silu,
        "tanh": jnp.tanh,
    }
    try:
        return mapping[name]
    except KeyError as err:
        raise ValueError(f"Unsupported activation '{name}'.") from err


class Identity(eqx.Module):
    def __call__(self, x):
        return x


class ModernBertAttention(eqx.Module):
    Wqkv: nn.Linear
    Wo: nn.Linear
    attn_drop: nn.Dropout
    out_drop: nn.Dropout
    rotary_emb: ModernBertRotaryEmbedding
    attention_module: AttentionModule
    num_attention_heads: int = field(static=True)
    attention_head_size: int = field(static=True)
    all_head_size: int = field(static=True)

    def __init__(
        self,
        config: ModernBertConfig,
        attention_config: AttentionConfig | None = None,
        layer_id: int = 0,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
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
        self.all_head_size = config.hidden_size

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

        self.attn_drop = nn.Dropout(
            p=config.attention_dropout, dtype=dtype, params_dtype=params_dtype
        )
        self.out_drop = nn.Dropout(
            p=config.attention_dropout, dtype=dtype, params_dtype=params_dtype
        )

        attention_config = first_from(
            attention_config,
            error_msg="ModernBertAttention requires an attention_config",
        )

        use_global = (layer_id % max(config.global_attn_every_n_layers, 1)) == 0
        rope_config = copy.deepcopy(config)

        if use_global:
            rope_config.rope_theta = config.global_rope_theta
            attn_cfg = dataclasses.replace(
                attention_config,
                use_local_attention=False,
                window_size=None,
            )
            self.attention_module = make_attention_module(config=attn_cfg, dtype=dtype)
        else:
            half_window = int(config.local_attention) // 2
            local_window = (half_window, half_window)
            rope_theta = (
                config.local_rope_theta
                if getattr(config, "local_rope_theta", None) is not None
                else config.global_rope_theta
            )
            rope_config.rope_theta = rope_theta
            attn_cfg = dataclasses.replace(
                attention_config,
                use_local_attention=True,
                window_size=local_window,
            )
            self.attention_module = ModernBertLocalAttention(attention_config=attn_cfg)

        self.rotary_emb = ModernBertRotaryEmbedding(
            rope_config,
            dtype=dtype,
            param_dtype=params_dtype,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        attention_mask: Int[Array, "B T"]
        | Float[Array, "B T"]
        | Bool[Array, "B T"]
        | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        position_ids: Int[Array, "B T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        if hidden_states.ndim < 2:
            raise ValueError("hidden_states must be (..., seq_len, hidden_size)")

        attn_key, out_key = (
            jax.random.split(key, 2) if key is not None else (None, None)
        )

        batch_shape = tuple(hidden_states.shape[:-2])
        seq_len = hidden_states.shape[-2]

        qkv = self.Wqkv(hidden_states)
        qkv = qkv.reshape(
            *batch_shape, seq_len, 3, self.num_attention_heads, self.attention_head_size
        )
        q, k, v = jnp.split(qkv, 3, axis=-3)
        q = jnp.squeeze(q, axis=-3)
        k = jnp.squeeze(k, axis=-3)
        v = jnp.squeeze(v, axis=-3)

        if position_ids is None:
            base = jnp.arange(seq_len, dtype=jnp.int32)
            position_ids = jnp.broadcast_to(base, tuple(batch_shape) + (seq_len,))
        else:
            position_ids = jnp.asarray(position_ids, dtype=jnp.int32)

        q = self.rotary_emb(q, position_ids=position_ids)

        k = self.rotary_emb(k, position_ids=position_ids)

        attn_output = self.attention_module(
            q,
            k,
            v,
            dropout=self.attn_drop,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            key=attn_key,
        )

        attn_output = attn_output.reshape(*batch_shape, seq_len, self.all_head_size)
        attn_output = self.Wo(attn_output)
        attn_output = self.out_drop(attn_output, key=out_key)
        return attn_output


class ModernBertMLP(eqx.Module):
    Wi: nn.Linear
    Wo: nn.Linear
    act: Callable[[Array], Array]
    drop: nn.Dropout

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        wi_key, wo_key = jax.random.split(key, 2)
        self.Wi = nn.Linear(
            config.hidden_size,
            2 * int(config.intermediate_size),
            use_bias=config.mlp_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=wi_key,
        )
        self.Wo = nn.Linear(
            int(config.intermediate_size),
            config.hidden_size,
            use_bias=config.mlp_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=wo_key,
        )
        self.act = _get_activation(config.hidden_activation)
        self.drop = nn.Dropout(
            p=config.mlp_dropout, dtype=dtype, params_dtype=params_dtype
        )

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        wi_out = self.Wi(hidden_states)
        input_part, gate = jnp.split(wi_out, 2, axis=-1)
        activated = self.act(input_part)
        gated = activated * gate
        dropped = self.drop(gated, key=key)
        return self.Wo(dropped)


class ModernBertEncoderLayer(eqx.Module):
    attn_norm: eqx.Module
    attention: ModernBertAttention
    mlp_norm: nn.LayerNorm
    mlp: ModernBertMLP

    def __init__(
        self,
        config: ModernBertConfig,
        attention_config: AttentionConfig,
        layer_id: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        attn_key, attn_norm_key, mlp_key, mlp_norm_key = jax.random.split(key, 4)
        if layer_id == 0:
            self.attn_norm = Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size,
                eps=config.norm_eps,
                bias=config.norm_bias,
                dtype=dtype,
                params_dtype=params_dtype,
                key=attn_norm_key,
            )
        self.attention = ModernBertAttention(
            config,
            attention_config=attention_config,
            layer_id=layer_id,
            dtype=dtype,
            params_dtype=params_dtype,
            key=attn_key,
        )
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=mlp_norm_key,
        )
        self.mlp = ModernBertMLP(
            config,
            dtype=dtype,
            params_dtype=params_dtype,
            key=mlp_key,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        attention_mask: Int[Array, "B T"]
        | Float[Array, "B T"]
        | Bool[Array, "B T"]
        | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        position_ids: Int[Array, "B T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        attn_key, mlp_key = (
            jax.random.split(key, 2) if key is not None else (None, None)
        )

        residual = hidden_states
        normed = self.attn_norm(hidden_states)
        attn_out = self.attention(
            normed,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            key=attn_key,
        )
        hidden_states = residual + attn_out

        mlp_residual = hidden_states
        normed_mlp = self.mlp_norm(hidden_states)
        mlp_out = self.mlp(normed_mlp, key=mlp_key)
        hidden_states = mlp_residual + mlp_out

        return hidden_states


class ModernBertEncoder(eqx.Module):
    layers: tuple[ModernBertEncoderLayer, ...]
    attention_config: AttentionConfig = field(static=True)

    def __init__(
        self,
        config: ModernBertConfig,
        attention_config: AttentionConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        layer_keys = jax.random.split(key, config.num_hidden_layers)
        self.layers = tuple(
            ModernBertEncoderLayer(
                config,
                attention_config=attention_config,
                layer_id=layer_id,
                dtype=dtype,
                params_dtype=params_dtype,
                key=layer_keys[layer_id],
            )
            for layer_id in range(config.num_hidden_layers)
        )
        self.attention_config = attention_config

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        attention_mask: Int[Array, "B T"]
        | Float[Array, "B T"]
        | Bool[Array, "B T"]
        | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        position_ids: Int[Array, "B T"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        if key is not None:
            layer_keys = jax.random.split(key, len(self.layers))
        else:
            layer_keys = [None] * len(self.layers)

        output = hidden_states
        for layer, layer_key in zip(self.layers, layer_keys):
            output = layer(
                output,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                position_ids=position_ids,
                key=layer_key,
            )
        return output


class ModernBertEmbeddings(eqx.Module):
    tok_embeddings: nn.Embedding
    norm: nn.LayerNorm
    drop: nn.Dropout

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        tok_key, norm_key = jax.random.split(key, 2)
        self.tok_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=tok_key,
        )
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=norm_key,
        )
        self.drop = nn.Dropout(
            p=config.embedding_dropout, dtype=dtype, params_dtype=params_dtype
        )

    def __call__(
        self,
        input_ids: Int[Array, "B T"] | None = None,
        inputs_embeds: Float[Array, "B T H"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None:
            hidden_states = self.tok_embeddings(input_ids)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        hidden_states = self.norm(hidden_states)
        hidden_states = self.drop(hidden_states, key=key)
        return hidden_states


class ModernBertModelWeightPlanMixin:
    def init_weights_plan(self, module, key):
        cfg = getattr(self, "config", None)
        if cfg is None:
            return module

        std = getattr(cfg, "initializer_range", 0.02)

        if isinstance(module, nn.Linear):
            w_key, b_key = jax.random.split(key, 2)
            w_shape = (module.out_features, module.in_features)
            w_dtype = module.params_dtype
            new_w = normal(std)(w_key, w_shape, dtype=w_dtype)
            new_module = eqx.tree_at(
                lambda m: m.weight,
                module,
                DArray(value=new_w, pspec=module.weight.pspec),
            )
            if module.use_bias and module.bias is not None:
                b_shape = (module.out_features,)
                new_b = zeros_init(b_key, b_shape, dtype=w_dtype)
                new_module = eqx.tree_at(
                    lambda m: m.bias,
                    new_module,
                    DArray(value=new_b, pspec=module.bias.pspec),
                )
            return new_module

        if isinstance(module, nn.Embedding):
            new_w = normal(std)(
                key,
                (module.num_embeddings, module.embedding_dim),
                dtype=module.params_dtype,
            )
            return eqx.tree_at(
                lambda m: m.weight,
                module,
                DArray(value=new_w, pspec=module.weight.pspec),
            )

        if isinstance(module, nn.LayerNorm):
            w_key, b_key = jax.random.split(key, 2)
            w_shape = module.normalized_shape
            w_dtype = module.params_dtype
            new_w = ones_init(w_key, w_shape, dtype=w_dtype)
            updated = eqx.tree_at(
                lambda m: m.weight,
                module,
                DArray(
                    value=new_w,
                    pspec=module.weight.pspec if module.weight is not None else None,
                ),
            )
            if module.bias is not None:
                new_b = zeros_init(b_key, w_shape, dtype=w_dtype)
                updated = eqx.tree_at(
                    lambda m: m.bias,
                    updated,
                    DArray(value=new_b, pspec=module.bias.pspec),
                )
            return updated

        if isinstance(module, ModernBertRotaryEmbedding):
            rtheta, attention_scaling = module.rope_init_fn(cfg, dtype=module.dtype)
            updated = eqx.tree_at(
                lambda m: [m.rtheta, m.attention_scaling],
                module,
                [rtheta, attention_scaling],
            )
            return updated

        return module


class ModernBertModel(
    eqx.Module,
    ModernBertModelWeightPlanMixin,
    HuggingFaceCompatibleModule[transformers.ModernBertModel],
):
    embeddings: ModernBertEmbeddings
    encoder: ModernBertEncoder
    final_norm: nn.LayerNorm
    config: ModernBertConfig | None = field(static=True, default=None)
    attention_config: AttentionConfig = field(static=True)

    def __init__(
        self,
        config: ModernBertConfig,
        attention_config: AttentionConfig | None = None,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        store_config: bool = True,
        key: PRNGKeyArray,
    ):
        embed_key, encoder_key, norm_key = jax.random.split(key, 3)
        attn_cfg = first_from(
            attention_config,
            AttentionConfig(type="eager", is_causal=False),
            error_msg="ModernBertModel requires an attention_config",
        )

        self.embeddings = ModernBertEmbeddings(
            config,
            dtype=dtype,
            params_dtype=params_dtype,
            key=embed_key,
        )
        self.encoder = ModernBertEncoder(
            config,
            attention_config=attn_cfg,
            dtype=dtype,
            params_dtype=params_dtype,
            key=encoder_key,
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=norm_key,
        )

        if store_config:
            self.config = config
        else:
            self.config = None
        self.attention_config = attn_cfg

    def __call__(
        self,
        input_ids: Int[Array, "B T"] | None = None,
        attention_mask: Int[Array, "B T"]
        | Float[Array, "B T"]
        | Bool[Array, "B T"]
        | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        position_ids: Int[Array, "B T"] | None = None,
        inputs_embeds: Float[Array, "B T H"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        embed_key, encoder_key = (
            jax.random.split(key, 2) if key is not None else (None, None)
        )

        hidden_states = self.embeddings(
            input_ids,
            inputs_embeds=inputs_embeds,
            key=embed_key,
        )
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            key=encoder_key,
        )
        hidden_states = self.final_norm(hidden_states)
        return hidden_states


class ModernBertPredictionHead(eqx.Module):
    dense: nn.Linear
    act: Callable[[Array], Array]
    norm: nn.LayerNorm

    def __init__(
        self,
        config: ModernBertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        dense_key, norm_key = jax.random.split(key, 2)
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            use_bias=config.classifier_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=dense_key,
        )
        self.act = _get_activation(config.classifier_activation)
        self.norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            bias=config.norm_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=norm_key,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "B T H"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T H"]:
        out = self.dense(hidden_states)
        out = self.act(out)
        out = self.norm(out)
        return out


class ModernBertForMaskedLM(
    eqx.Module,
    ModernBertModelWeightPlanMixin,
    HuggingFaceCompatibleModule[transformers.ModernBertForMaskedLM],
):
    model: ModernBertModel
    head: ModernBertPredictionHead
    decoder: nn.Linear
    config: ModernBertConfig | None = field(static=True, default=None)

    def __init__(
        self,
        config: ModernBertConfig,
        attention_config: AttentionConfig | None = None,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        store_config: bool = True,
        key: PRNGKeyArray,
    ):
        model_key, head_key, decoder_key = jax.random.split(key, 3)
        self.model = ModernBertModel(
            config,
            attention_config=attention_config,
            dtype=dtype,
            params_dtype=params_dtype,
            store_config=False,
            key=model_key,
        )
        self.head = ModernBertPredictionHead(
            config,
            dtype=dtype,
            params_dtype=params_dtype,
            key=head_key,
        )
        self.decoder = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            use_bias=config.decoder_bias,
            dtype=dtype,
            params_dtype=params_dtype,
            key=decoder_key,
        )
        if store_config:
            self.config = config
        else:
            self.config = None

    def __call__(
        self,
        input_ids: Int[Array, "B T"] | None = None,
        attention_mask: Int[Array, "B T"]
        | Float[Array, "B T"]
        | Bool[Array, "B T"]
        | None = None,
        segment_ids: Int[Array, "B T"] | None = None,
        position_ids: Int[Array, "B T"] | None = None,
        inputs_embeds: Float[Array, "B T H"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "B T V"]:
        model_key, head_key = (
            jax.random.split(key, 2) if key is not None else (None, None)
        )
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            key=model_key,
        )
        hidden_states = self.head(hidden_states, key=head_key)
        logits = self.decoder(hidden_states)
        return logits


__all__ = [
    "ModernBertRotaryEmbedding",
    "ModernBertAttention",
    "ModernBertMLP",
    "ModernBertEncoder",
    "ModernBertEmbeddings",
    "ModernBertModel",
    "ModernBertPredictionHead",
    "ModernBertForMaskedLM",
]
