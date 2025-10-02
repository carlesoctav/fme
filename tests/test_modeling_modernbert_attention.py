import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest


try:
    import torch
    from transformers import ModernBertConfig
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
    from transformers.models.modernbert.modeling_modernbert import (
        ModernBertAttention as TorchModernBertAttention,
    )
except Exception as e:  # pragma: no cover - allow skipping when transformers is unavailable
    pytest.skip(f"transformers not available: {e}", allow_module_level=True)

from src.nn import AttentionConfig
from src.models.modernbert.modeling_modernbert import ModernBertAttention


def make_config(**overrides) -> ModernBertConfig:
    config = ModernBertConfig(
        hidden_size=64,
        num_attention_heads=8,
        global_attn_every_n_layers=2,
        local_attention=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        attention_bias=False,
        deterministic_flash_attn=False,
        rope_scaling=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    config._attn_implementation = "eager"
    return config


def copy_linear_weights(eq_mod, attr: str, weight: np.ndarray, bias: np.ndarray | None):
    eq_mod = eqx.tree_at(
        lambda m: getattr(getattr(m, attr), "weight").value,
        eq_mod,
        jnp.asarray(weight),
    )
    linear = getattr(eq_mod, attr)
    if bias is not None and linear.bias is not None:
        eq_mod = eqx.tree_at(
            lambda m: getattr(getattr(m, attr), "bias").value,
            eq_mod,
            jnp.asarray(bias),
        )
    return eq_mod


def copy_attention_weights(eq_mod: ModernBertAttention, torch_mod: TorchModernBertAttention) -> ModernBertAttention:
    with torch.no_grad():
        w_qkv = torch_mod.Wqkv.weight.detach().cpu().numpy()
        b_qkv = (
            torch_mod.Wqkv.bias.detach().cpu().numpy()
            if torch_mod.Wqkv.bias is not None
            else None
        )

        w_q, w_k, w_v = np.split(w_qkv, 3, axis=0)
        b_q, b_k, b_v = (None, None, None)
        if b_qkv is not None:
            b_q, b_k, b_v = np.split(b_qkv, 3, axis=0)

        eq_mod = copy_linear_weights(eq_mod, "query", w_q, b_q)
        eq_mod = copy_linear_weights(eq_mod, "key", w_k, b_k)
        eq_mod = copy_linear_weights(eq_mod, "value", w_v, b_v)

        w_out = torch_mod.Wo.weight.detach().cpu().numpy()
        b_out = (
            torch_mod.Wo.bias.detach().cpu().numpy()
            if torch_mod.Wo.bias is not None
            else None
        )
        eq_mod = copy_linear_weights(eq_mod, "output", w_out, b_out)
    return eq_mod


def make_sliding_window_mask(attention_mask: torch.Tensor, local_attention: int) -> torch.Tensor:
    global_attention_mask = _prepare_4d_attention_mask(attention_mask, dtype=torch.float32)
    seq_len = global_attention_mask.shape[-1]
    rows = torch.arange(seq_len, device=attention_mask.device).unsqueeze(0)
    distance = torch.abs(rows - rows.T)
    window_mask = (distance <= (local_attention // 2)).unsqueeze(0).unsqueeze(0)
    sliding_window = global_attention_mask.masked_fill(~window_mask, torch.finfo(torch.float32).min)
    return sliding_window


@pytest.mark.parametrize("layer_id", [0, 1])
def test_modernbert_attention_matches_torch(layer_id: int):
    config = make_config()

    torch_attn = TorchModernBertAttention(config=config, layer_id=layer_id)
    key = jax.random.PRNGKey(0)
    eqx_attn = ModernBertAttention(
        config,
        attention_config=AttentionConfig(type="eager", is_causal=False),
        layer_id=layer_id,
        dtype=jnp.float32,
        params_dtype=jnp.float32,
        store_config=False,
        key=key,
    )
    eqx_attn = copy_attention_weights(eqx_attn, torch_attn)

    batch_size, seq_len = 2, 12
    torch_hidden = torch.randn(batch_size, seq_len, config.hidden_size, dtype=torch.float32)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    sliding_mask = None
    if layer_id % config.global_attn_every_n_layers != 0:
        sliding_mask = make_sliding_window_mask(attention_mask, config.local_attention)

    torch_outputs = torch_attn(
        hidden_states=torch_hidden,
        attention_mask=attention_mask,
        sliding_window_mask=sliding_mask,
        position_ids=position_ids,
        output_attentions=False,
    )[0]

    jax_hidden = jnp.asarray(torch_hidden.numpy())
    jax_mask = jnp.asarray(attention_mask.numpy().astype(np.int32))
    jax_pos = jnp.asarray(position_ids.numpy())

    jax_outputs = eqx_attn(
        jax_hidden,
        attention_mask=jax_mask,
        position_ids=jax_pos,
        key=None,
    )

    np.testing.assert_allclose(
        jax_outputs, torch_outputs.detach().cpu().numpy(), atol=1e-4, rtol=1e-4
    )
