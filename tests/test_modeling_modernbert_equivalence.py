from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch


try:  # pragma: no cover - dependency availability gate
    from transformers import ModernBertConfig
    from transformers.models.modernbert.modeling_modernbert import (
        ModernBertForMaskedLM as TorchModernBertForMaskedLM,
        ModernBertModel as TorchModernBertModel,
    )
except Exception as err:  # pragma: no cover - allow running without transformers locally
    pytest.skip(f"transformers not available: {err}", allow_module_level=True)

from src import make_module_opt
from src.nn._attention import AttentionConfig as EqAttentionConfig
from src.models.modernbert.modeling_modernbert import (
    ModernBertForMaskedLM,
    ModernBertModel,
)
from tests.utils import (
    assert_close,
    has_shape_dtype_struct,
    update_embedding,
    update_layernorm,
    update_linear,
)


def _make_config(**overrides) -> ModernBertConfig:
    config = ModernBertConfig(
        hidden_size=48,
        num_attention_heads=6,
        intermediate_size=128,
        num_hidden_layers=2,
        vocab_size=50257,
        max_position_embeddings=96,
        global_attn_every_n_layers=2,
        local_attention=12,
        attention_dropout=0.0,
        mlp_dropout=0.0,
        embedding_dropout=0.0,
        attention_bias=False,
        mlp_bias=False,
        decoder_bias=False,
        classifier_bias=False,
        norm_bias=False,
        deterministic_flash_attn=False,
        rope_scaling=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    config._attn_implementation = "eager"
    return config


def _copy_embeddings(eq_emb, torch_emb):
    tok = update_embedding(eq_emb.tok_embeddings, torch_emb.tok_embeddings)
    eq_emb = eqx.tree_at(lambda m: m.tok_embeddings, eq_emb, tok)
    norm = update_layernorm(eq_emb.norm, torch_emb.norm)
    eq_emb = eqx.tree_at(lambda m: m.norm, eq_emb, norm)
    return eq_emb


def _copy_attention(eq_attn, torch_attn):
    wqkv = update_linear(eq_attn.Wqkv, torch_attn.Wqkv)
    eq_attn = eqx.tree_at(lambda m: m.Wqkv, eq_attn, wqkv)
    wo = update_linear(eq_attn.Wo, torch_attn.Wo)
    eq_attn = eqx.tree_at(lambda m: m.Wo, eq_attn, wo)
    return eq_attn


def _copy_mlp(eq_mlp, torch_mlp):
    wi = update_linear(eq_mlp.Wi, torch_mlp.Wi)
    eq_mlp = eqx.tree_at(lambda m: m.Wi, eq_mlp, wi)
    wo = update_linear(eq_mlp.Wo, torch_mlp.Wo)
    eq_mlp = eqx.tree_at(lambda m: m.Wo, eq_mlp, wo)
    return eq_mlp


def _copy_encoder_layer(eq_layer, torch_layer):
    if hasattr(torch_layer.attn_norm, "weight") and hasattr(eq_layer.attn_norm, "weight"):
        attn_norm = update_layernorm(eq_layer.attn_norm, torch_layer.attn_norm)
        eq_layer = eqx.tree_at(lambda m: m.attn_norm, eq_layer, attn_norm)

    attention = _copy_attention(eq_layer.attention, torch_layer.attn)
    eq_layer = eqx.tree_at(lambda m: m.attention, eq_layer, attention)

    mlp_norm = update_layernorm(eq_layer.mlp_norm, torch_layer.mlp_norm)
    eq_layer = eqx.tree_at(lambda m: m.mlp_norm, eq_layer, mlp_norm)

    mlp = _copy_mlp(eq_layer.mlp, torch_layer.mlp)
    eq_layer = eqx.tree_at(lambda m: m.mlp, eq_layer, mlp)
    return eq_layer


def _copy_encoder(eq_encoder, torch_layers):
    updated_layers = tuple(
        _copy_encoder_layer(eq_layer, torch_layer)
        for eq_layer, torch_layer in zip(eq_encoder.layers, torch_layers)
    )
    return eqx.tree_at(lambda m: m.layers, eq_encoder, updated_layers)


def _copy_model(eq_model: ModernBertModel, torch_model: TorchModernBertModel) -> ModernBertModel:
    embeddings = _copy_embeddings(eq_model.embeddings, torch_model.embeddings)
    eq_model = eqx.tree_at(lambda m: m.embeddings, eq_model, embeddings)

    encoder = _copy_encoder(eq_model.encoder, torch_model.layers)
    eq_model = eqx.tree_at(lambda m: m.encoder, eq_model, encoder)

    final_norm = update_layernorm(eq_model.final_norm, torch_model.final_norm)
    eq_model = eqx.tree_at(lambda m: m.final_norm, eq_model, final_norm)
    return eq_model


def _copy_mlm(eq_model: ModernBertForMaskedLM, torch_model: TorchModernBertForMaskedLM) -> ModernBertForMaskedLM:
    core = _copy_model(eq_model.model, torch_model.model)
    eq_model = eqx.tree_at(lambda m: m.model, eq_model, core)

    head_dense = update_linear(eq_model.head.dense, torch_model.head.dense)
    eq_model = eqx.tree_at(lambda m: m.head.dense, eq_model, head_dense)

    head_norm = update_layernorm(eq_model.head.norm, torch_model.head.norm)
    eq_model = eqx.tree_at(lambda m: m.head.norm, eq_model, head_norm)

    decoder = update_linear(eq_model.decoder, torch_model.decoder)
    eq_model = eqx.tree_at(lambda m: m.decoder, eq_model, decoder)
    return eq_model


def _prepare_inputs(config: ModernBertConfig, *, batch_size: int, seq_len: int):
    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    if seq_len > 4:
        attention_mask[:, -1] = 0
    return input_ids, attention_mask


def test_modernbert_model_matches_transformers():
    config = _make_config()

    torch_model = TorchModernBertModel(config).eval()
    eq_key = jax.random.PRNGKey(0)
    eq_model = ModernBertModel(
        config,
        attention_config=EqAttentionConfig(type="eager", is_causal=False),
        dtype=jnp.float32,
        params_dtype=jnp.float32,
        key=eq_key,
    )

    eq_model = _copy_model(eq_model, torch_model)

    input_ids, attention_mask = _prepare_inputs(config, batch_size=2, seq_len=16)

    with torch.no_grad():
        torch_out = torch_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    jax_inputs = jnp.asarray(input_ids.numpy())
    jax_mask = jnp.asarray(attention_mask.numpy().astype(np.int32))

    eq_out = eq_model(jax_inputs, attention_mask=jax_mask, key=jax.random.PRNGKey(1))

    assert_close(eq_out, torch_out.last_hidden_state.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)


def test_modernbert_for_maskedlm_matches_transformers():
    config = _make_config(vocab_size=128)

    torch_model = TorchModernBertForMaskedLM(config).eval()
    eq_key = jax.random.PRNGKey(123)
    eq_model = ModernBertForMaskedLM(
        config,
        attention_config=EqAttentionConfig(type="eager", is_causal=False),
        dtype=jnp.float32,
        params_dtype=jnp.float32,
        key=eq_key,
    )

    eq_model = _copy_mlm(eq_model, torch_model)

    input_ids, attention_mask = _prepare_inputs(config, batch_size=2, seq_len=12)

    with torch.no_grad():
        torch_out = torch_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

    jax_inputs = jnp.asarray(input_ids.numpy())
    jax_mask = jnp.asarray(attention_mask.numpy().astype(np.int32))

    eq_logits = eq_model(jax_inputs, attention_mask=jax_mask, key=jax.random.PRNGKey(2))

    assert_close(eq_logits, torch_out.logits.detach().cpu().numpy(), atol=1e-4, rtol=1e-4)


def test_make_module_opt_initializes_modernbert():
    config = _make_config()
    attn_cfg = EqAttentionConfig(type="eager", is_causal=False)

    def build_module(rng):
        return ModernBertModel(
            config,
            attention_config=attn_cfg,
            dtype=jnp.float32,
            params_dtype=jnp.float32,
            key=rng,
        )

    abstract_module = eqx.filter_eval_shape(build_module, jax.random.PRNGKey(0))
    assert has_shape_dtype_struct(abstract_module)

    initialized_module, optimizer = make_module_opt(
        abstract_module,
        grad_tx=optax.identity(),
        key=jax.random.PRNGKey(1),
    )

    assert not has_shape_dtype_struct(initialized_module)
    # Optimizer state should also be concrete
    assert not has_shape_dtype_struct(optimizer)
