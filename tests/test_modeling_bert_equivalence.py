import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

try:
    from transformers import AutoTokenizer, BertConfig, BertModel as TorchBertModel
except Exception as e:
    pytest.skip(f"transformers not available: {e}", allow_module_level=True)

from src.models.bert.modeling_bert import BertModel
from tests.utils import set_attr, t2np


def copy_bert_weights(jx_model: BertModel, th_model: TorchBertModel):
    jx_model = set_attr(
        jx_model,
        "embeddings.word_embeddings.weight",
        t2np(th_model.embeddings.word_embeddings.weight),
    )
    jx_model = set_attr(
        jx_model,
        "embeddings.position_embeddings.weight",
        t2np(th_model.embeddings.position_embeddings.weight),
    )
    jx_model = set_attr(
        jx_model,
        "embeddings.token_type_embeddings.weight",
        t2np(th_model.embeddings.token_type_embeddings.weight),
    )
    jx_model = set_attr(
        jx_model,
        "embeddings.LayerNorm.weight",
        t2np(th_model.embeddings.LayerNorm.weight),
    )
    jx_model = set_attr(
        jx_model, "embeddings.LayerNorm.bias", t2np(th_model.embeddings.LayerNorm.bias)
    )

    for i, th_layer in enumerate(th_model.encoder.layer):
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.self.query.weight",
            t2np(th_layer.attention.self.query.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.self.query.bias",
            t2np(th_layer.attention.self.query.bias),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.self.key.weight",
            t2np(th_layer.attention.self.key.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.self.key.bias",
            t2np(th_layer.attention.self.key.bias),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.self.value.weight",
            t2np(th_layer.attention.self.value.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.self.value.bias",
            t2np(th_layer.attention.self.value.bias),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.output.dense.weight",
            t2np(th_layer.attention.output.dense.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.output.dense.bias",
            t2np(th_layer.attention.output.dense.bias),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.output.LayerNorm.weight",
            t2np(th_layer.attention.output.LayerNorm.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.attention.output.LayerNorm.bias",
            t2np(th_layer.attention.output.LayerNorm.bias),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.intermediate.dense.weight",
            t2np(th_layer.intermediate.dense.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.intermediate.dense.bias",
            t2np(th_layer.intermediate.dense.bias),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.output.dense.weight",
            t2np(th_layer.output.dense.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.output.dense.bias",
            t2np(th_layer.output.dense.bias),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.output.LayerNorm.weight",
            t2np(th_layer.output.LayerNorm.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layer.{i}.output.LayerNorm.bias",
            t2np(th_layer.output.LayerNorm.bias),
        )

    return jx_model


def make_config():
    return BertConfig(
        vocab_size=30522,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=128,
        type_vocab_size=2,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,has_padding",
    [
        (2, 8, False),
        (2, 10, True),
    ],
)
def test_modeling_bert(batch_size, seq_len, has_padding):
    cfg = make_config()
    cfg._attn_implementation = "eager"
    key = jax.random.key(42)

    input_ids = torch.randint(
        0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long
    )
    position_ids = (
        torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

    if has_padding:
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
        attention_mask[0, seq_len // 2 :] = 0
    else:
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    th_model = TorchBertModel(cfg)
    th_model.eval()

    with torch.no_grad():
        th_out = th_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    jx_model = BertModel(cfg, key=key)
    jx_model = copy_bert_weights(jx_model, th_model)

    jx_out = jx_model(
        jnp.asarray(input_ids.numpy()),
        jnp.asarray(position_ids.numpy()),
        jnp.asarray(token_type_ids.numpy()),
        jnp.asarray(attention_mask.numpy()),
        key=key,
    )

    np.testing.assert_allclose(jx_out, th_out.numpy(), atol=1e-3, rtol=1e-3)


def test_modeling_bert_with_segment_ids():
    cfg = make_config()
    cfg._attn_implementation = "eager"
    key = jax.random.key(42)

    seq_len_1 = 6
    seq_len_2 = 8
    total_len = seq_len_1 + seq_len_2

    input_ids_1 = torch.randint(0, cfg.vocab_size, (1, seq_len_1), dtype=torch.long)
    input_ids_2 = torch.randint(0, cfg.vocab_size, (1, seq_len_2), dtype=torch.long)

    position_ids_1 = torch.arange(0, seq_len_1, dtype=torch.long).unsqueeze(0)
    position_ids_2 = torch.arange(0, seq_len_2, dtype=torch.long).unsqueeze(0)

    token_type_ids_1 = torch.zeros((1, seq_len_1), dtype=torch.long)
    token_type_ids_2 = torch.zeros((1, seq_len_2), dtype=torch.long)

    attention_mask_1 = torch.ones((1, seq_len_1), dtype=torch.long)
    attention_mask_2 = torch.ones((1, seq_len_2), dtype=torch.long)

    th_model = TorchBertModel(cfg)
    th_model.eval()

    with torch.no_grad():
        th_out_1 = th_model(
            input_ids=input_ids_1,
            token_type_ids=token_type_ids_1,
            position_ids=position_ids_1,
            attention_mask=attention_mask_1,
        ).last_hidden_state

        th_out_2 = th_model(
            input_ids=input_ids_2,
            token_type_ids=token_type_ids_2,
            position_ids=position_ids_2,
            attention_mask=attention_mask_2,
        ).last_hidden_state

    input_ids_packed = jnp.concatenate(
        [jnp.asarray(input_ids_1.numpy()), jnp.asarray(input_ids_2.numpy())], axis=1
    )
    position_ids_packed = jnp.concatenate(
        [jnp.asarray(position_ids_1.numpy()), jnp.asarray(position_ids_2.numpy())],
        axis=1,
    )
    token_type_ids_packed = jnp.concatenate(
        [
            jnp.asarray(token_type_ids_1.numpy()),
            jnp.asarray(token_type_ids_2.numpy()),
        ],
        axis=1,
    )

    segment_ids = jnp.concatenate(
        [
            jnp.zeros((1, seq_len_1), dtype=jnp.int32),
            jnp.ones((1, seq_len_2), dtype=jnp.int32),
        ],
        axis=1,
    )

    jx_model = BertModel(cfg, key=key)
    jx_model = copy_bert_weights(jx_model, th_model)

    jx_out = jx_model(
        input_ids_packed,
        position_ids_packed,
        token_type_ids_packed,
        segment_ids=segment_ids,
        key=key,
    )

    np.testing.assert_allclose(
        jx_out[0, :seq_len_1, :], th_out_1.numpy()[0], atol=1e-3, rtol=1e-3
    )
    np.testing.assert_allclose(
        jx_out[0, seq_len_1:, :], th_out_2.numpy()[0], atol=1e-3, rtol=1e-3
    )


def test_bert_with_real_model():
    model_name = "google-bert/bert-base-uncased"
    text1 = "hallo nama saya carles dan ini adalah text yang lebih panjang"
    text2 = "hallo dunia"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    th_model = TorchBertModel.from_pretrained(model_name)
    th_model.eval()

    encoded = tokenizer(
        [text1, text2],
        padding=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids))

    batch_size, seq_len = input_ids.shape
    position_ids = (
        torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )

    with torch.no_grad():
        th_out = th_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        ).last_hidden_state

    key = jax.random.key(42)
    jx_model = BertModel(th_model.config, key=key)
    jx_model = copy_bert_weights(jx_model, th_model)
    jx_model = eqx.nn.inference_mode(jx_model)

    jx_out = jx_model(
        jnp.asarray(input_ids.numpy()),
        jnp.asarray(position_ids.numpy()),
        jnp.asarray(token_type_ids.numpy()),
        jnp.asarray(attention_mask.numpy()),
        key=key,
    )

    np.testing.assert_allclose(jx_out, th_out.numpy(), atol=5e-2, rtol=5e-2)
