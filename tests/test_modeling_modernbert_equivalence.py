import os

# Force JAX to prefer the CPU backend so tests do not try to grab a TPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

try:
    from transformers import (
        AutoTokenizer,
        ModernBertConfig,
        ModernBertModel as TorchModernBertModel,
    )
except Exception as e:
    pytest.skip(f"transformers not available: {e}", allow_module_level=True)

from src.models.modernbert.modeling_modernbert import ModernBertModel
from tests.utils import set_attr, t2np


def copy_modernbert_weights(jx_model: ModernBertModel, th_model: TorchModernBertModel):
    jx_model = set_attr(
        jx_model,
        "embeddings.tok_embeddings.weight",
        t2np(th_model.embeddings.tok_embeddings.weight),
    )
    jx_model = set_attr(
        jx_model,
        "embeddings.norm.weight",
        t2np(th_model.embeddings.norm.weight),
    )

    for i, th_layer in enumerate(th_model.layers):
        if hasattr(th_layer, "attn_norm") and hasattr(th_layer.attn_norm, "weight"):
            jx_model = set_attr(
                jx_model,
                f"encoder.layers.{i}.attn_norm.weight",
                t2np(th_layer.attn_norm.weight),
            )

        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.attention.Wqkv.weight",
            t2np(th_layer.attn.Wqkv.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.attention.Wo.weight",
            t2np(th_layer.attn.Wo.weight),
        )

        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.mlp_norm.weight",
            t2np(th_layer.mlp_norm.weight),
        )

        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.mlp.Wi.weight",
            t2np(th_layer.mlp.Wi.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.mlp.Wo.weight",
            t2np(th_layer.mlp.Wo.weight),
        )

    jx_model = set_attr(
        jx_model,
        "final_norm.weight",
        t2np(th_model.final_norm.weight),
    )

    return jx_model


def make_config():
    return ModernBertConfig(
        hidden_size=48,
        num_attention_heads=6,
        intermediate_size=128,
        num_hidden_layers=4,
        vocab_size=60000,
        max_position_embeddings=96,
        global_attn_every_n_layers=1,
        local_attention=5,
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


@pytest.mark.parametrize(
    "batch_size,seq_len,has_padding",
    [
        (2, 8, False),
        (2, 10, True),
    ],
)
def test_modeling_modernbert(batch_size, seq_len, has_padding):
    cfg = make_config()
    cfg._attn_implementation = "eager"
    key = jax.random.key(42)

    input_ids = torch.randint(
        0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long
    )

    if has_padding:
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        attention_mask[0, seq_len // 2 :] = 0
    else:
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    th_model = TorchModernBertModel(cfg)
    th_model.eval()

    with torch.no_grad():
        th_out = th_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    jx_model = ModernBertModel(cfg, key=key)
    jx_model = copy_modernbert_weights(jx_model, th_model)

    jx_out = jx_model(
        jnp.asarray(input_ids.numpy()),
        attention_mask=jnp.asarray(attention_mask.numpy()),
        key=key,
    )

    np.testing.assert_allclose(jx_out, th_out.numpy(), atol=1e-3, rtol=1e-3)


def test_modeling_modernbert_with_segment_ids():
    cfg = make_config()
    cfg._attn_implementation = "eager"
    key = jax.random.key(42)

    seq_len_1 = 6
    seq_len_2 = 8
    total_len = seq_len_1 + seq_len_2

    input_ids_1 = torch.randint(0, cfg.vocab_size, (1, seq_len_1), dtype=torch.long)
    input_ids_2 = torch.randint(0, cfg.vocab_size, (1, seq_len_2), dtype=torch.long)

    attention_mask_1 = torch.ones((1, seq_len_1), dtype=torch.long)
    attention_mask_2 = torch.ones((1, seq_len_2), dtype=torch.long)

    th_model = TorchModernBertModel(cfg)
    th_model.eval()

    with torch.no_grad():
        th_out_1 = th_model(
            input_ids=input_ids_1,
            attention_mask=attention_mask_1,
        ).last_hidden_state

        th_out_2 = th_model(
            input_ids=input_ids_2,
            attention_mask=attention_mask_2,
        ).last_hidden_state

    input_ids_packed = jnp.concatenate(
        [jnp.asarray(input_ids_1.numpy()), jnp.asarray(input_ids_2.numpy())], axis=1
    )

    segment_ids = jnp.concatenate(
        [
            jnp.zeros((1, seq_len_1), dtype=jnp.int32),
            jnp.ones((1, seq_len_2), dtype=jnp.int32),
        ],
        axis=1,
    )

    jx_model = ModernBertModel(cfg, key=key)
    jx_model = copy_modernbert_weights(jx_model, th_model)

    jx_out = jx_model(
        input_ids_packed,
        segment_ids=segment_ids,
        key=key,
    )

    np.testing.assert_allclose(
        jx_out[0, :seq_len_1, :], th_out_1.numpy()[0], atol=1e-3, rtol=1e-3
    )
    np.testing.assert_allclose(
        jx_out[0, seq_len_1:, :], th_out_2.numpy()[0], atol=1e-3, rtol=1e-3
    )


def test_modernbert_with_real_model():
    model_name = "answerdotai/ModernBERT-base"
    text1 = "hallo nama saya carles dan ini adalah text yang lebih panjang"
    text2 = "hallo dunia"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    th_model = TorchModernBertModel.from_pretrained(model_name)
    th_model.eval()

    encoded = tokenizer(
        [text1, text2],
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    with torch.no_grad():
        th_out = th_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    key = jax.random.key(42)
    jx_model = ModernBertModel(th_model.config, key=key)
    jx_model = copy_modernbert_weights(jx_model, th_model)
    jx_model = eqx.nn.inference_mode(jx_model)

    jx_out = jx_model(
        jnp.asarray(input_ids.numpy()),
        attention_mask=jnp.asarray(attention_mask.numpy(), dtype = np.bool),
        key=key,
    )

    np.testing.assert_allclose(jx_out, th_out.numpy(), atol=5e-2, rtol=5e-2)
