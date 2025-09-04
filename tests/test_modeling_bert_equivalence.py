from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
import torch


# Guard entire module if transformers is unavailable (e.g., offline env)
try:
    from transformers import (
        AutoTokenizer,
        BertConfig,
        BertForMaskedLM as TorchBertForMaskedLM,
    )
    from transformers.models.bert.modeling_bert import (
        BertAttention as TorchBertAttention,
        BertEmbeddings as TorchBertEmbeddings,
        BertEncoder as TorchBertEncoder,
        BertIntermediate as TorchBertIntermediate,
        BertLayer as TorchBertLayer,
        BertOutput as TorchBertOutput,
        BertSelfAttention as TorchBertSelfAttention,
        BertSelfOutput as TorchBertSelfOutput,
    )
except Exception as e:  # pragma: no cover - skip gracefully when HF is missing
    pytest.skip(f"transformers not available: {e}", allow_module_level=True)

from src.models.bert.modeling_bert import (
    BertAttention,
    BertEmbeddings,
    BertEncoder,
    BertForMaskedLM,
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)


# --------------------------
# Utilities
# --------------------------


def t2np(t: torch.Tensor):
    return t.detach().cpu().numpy()


def arr(x):
    return jnp.asarray(x)


def set_attr(module, path, value):
    # Helper to set nested attributes via eqx.tree_at without importing custom wrappers
    return eqx.tree_at(lambda m: eval("m." + path), module, arr(value))


def copy_embedding_weights(jx_emb: BertEmbeddings, th_emb: TorchBertEmbeddings):
    jx_emb = set_attr(jx_emb, "word_embeddings.weight", t2np(th_emb.word_embeddings.weight))
    jx_emb = set_attr(jx_emb, "position_embeddings.weight", t2np(th_emb.position_embeddings.weight))
    jx_emb = set_attr(jx_emb, "token_type_embeddings.weight", t2np(th_emb.token_type_embeddings.weight))
    jx_emb = set_attr(jx_emb, "LayerNorm.weight", t2np(th_emb.LayerNorm.weight))
    jx_emb = set_attr(jx_emb, "LayerNorm.bias", t2np(th_emb.LayerNorm.bias))
    return jx_emb


def copy_linear(jx_mod, base: str, th_mod):
    # Copies .<base>.weight and .<base>.bias from th_mod to jx_mod
    jx_mod = set_attr(jx_mod, f"{base}.weight", t2np(getattr(th_mod, base).weight))
    if getattr(th_mod, base).bias is not None:
        jx_mod = set_attr(jx_mod, f"{base}.bias", t2np(getattr(th_mod, base).bias))
    return jx_mod


def copy_self_attn_weights(jx_sa: BertSelfAttention, th_sa: TorchBertSelfAttention):
    jx_sa = copy_linear(jx_sa, "query", th_sa)
    jx_sa = copy_linear(jx_sa, "key", th_sa)
    jx_sa = copy_linear(jx_sa, "value", th_sa)
    return jx_sa


def copy_self_output_weights(jx_so: BertSelfOutput, th_so: TorchBertSelfOutput):
    jx_so = copy_linear(jx_so, "dense", th_so)
    jx_so = set_attr(jx_so, "LayerNorm.weight", t2np(th_so.LayerNorm.weight))
    jx_so = set_attr(jx_so, "LayerNorm.bias", t2np(th_so.LayerNorm.bias))
    return jx_so


def copy_attention_weights(jx_attn: BertAttention, th_attn: TorchBertAttention):
    jx_attn_self = copy_self_attn_weights(jx_attn.self, th_attn.self)
    jx_attn_out = copy_self_output_weights(jx_attn.output, th_attn.output)
    jx_attn = eqx.tree_at(lambda m: m.self, jx_attn, jx_attn_self)
    jx_attn = eqx.tree_at(lambda m: m.output, jx_attn, jx_attn_out)
    return jx_attn


def copy_intermediate_weights(jx_inter: BertIntermediate, th_inter: TorchBertIntermediate):
    jx_inter = copy_linear(jx_inter, "dense", th_inter)
    return jx_inter


def copy_output_weights(jx_out: BertOutput, th_out: TorchBertOutput):
    jx_out = copy_linear(jx_out, "dense", th_out)
    jx_out = set_attr(jx_out, "LayerNorm.weight", t2np(th_out.LayerNorm.weight))
    jx_out = set_attr(jx_out, "LayerNorm.bias", t2np(th_out.LayerNorm.bias))
    return jx_out


def copy_layer_weights(jx_layer: BertLayer, th_layer: TorchBertLayer):
    jx_layer = eqx.tree_at(
        lambda m: m.attention,
        jx_layer,
        copy_attention_weights(jx_layer.attention, th_layer.attention),
    )
    jx_layer = eqx.tree_at(
        lambda m: m.intermediate,
        jx_layer,
        copy_intermediate_weights(jx_layer.intermediate, th_layer.intermediate),
    )
    jx_layer = eqx.tree_at(
        lambda m: m.output,
        jx_layer,
        copy_output_weights(jx_layer.output, th_layer.output),
    )
    return jx_layer


def copy_encoder_weights(jx_encoder: BertEncoder, th_encoder: TorchBertEncoder):
    for i, th_layer in enumerate(th_encoder.layer):
        jx_layer = copy_layer_weights(jx_encoder.layer[i], th_layer)
        jx_encoder = eqx.tree_at(lambda m: m.layer[i], jx_encoder, jx_layer)
    return jx_encoder


def make_position_ids(input_ids: torch.Tensor):
    # position_ids aligned with sequence length per sample (no offsets)
    bs, seqlen = input_ids.shape
    return torch.arange(0, seqlen, dtype=torch.long).unsqueeze(0).expand(bs, -1)


def extend_attention_mask(attn_mask_2d: torch.Tensor) -> torch.Tensor:
    # attn_mask_2d: (batch, seq_len) with 1 for tokens, 0 for pads
    if attn_mask_2d is None:
        return None
    # Create additive mask (batch, 1, 1, seq_len)
    extended = attn_mask_2d[:, None, None, :].to(dtype=torch.float32)
    extended = (1.0 - extended) * -10000.0
    return extended


def compare_close(a: jnp.ndarray, b: jnp.ndarray, atol=1e-4, rtol=1e-4):
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    assert jnp.allclose(a, b, atol=atol, rtol=rtol), (
        f"Outputs differ beyond tolerance. max_abs={float(jnp.max(jnp.abs(a - b)))}"
    )


@dataclass
class SmallConfig:
    hidden_size: int = 64
    num_attention_heads: int = 8
    intermediate_size: int = 256
    vocab_size: int = 30522
    max_position_embeddings: int = 96
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    num_hidden_layers: int = 2


def make_config(**overrides):
    base = SmallConfig()
    base.__dict__.update(overrides)
    return BertConfig(
        vocab_size=base.vocab_size,
        hidden_size=base.hidden_size,
        num_hidden_layers=base.num_hidden_layers,
        num_attention_heads=base.num_attention_heads,
        intermediate_size=base.intermediate_size,
        max_position_embeddings=base.max_position_embeddings,
        type_vocab_size=base.type_vocab_size,
        layer_norm_eps=base.layer_norm_eps,
        hidden_dropout_prob=base.hidden_dropout_prob,
        attention_probs_dropout_prob=base.attention_probs_dropout_prob,
    )


def vmap_embeddings(jx_emb: BertEmbeddings, input_ids: jnp.ndarray, position_ids: jnp.ndarray, token_type_ids: jnp.ndarray, *, key):
    # input tensors: (batch, seq_len)
    batch = input_ids.shape[0]
    keys = jax.random.split(key, batch)
    fn = lambda ids, pos, tt, k: jx_emb(ids, pos, tt, key=k)
    return jax.vmap(fn)(input_ids, position_ids, token_type_ids, keys)


def vmap_call(module, hidden_states: jnp.ndarray, attention_mask: jnp.ndarray | None, *, key):
    # module signature: (seq_len, hidden_size) and optional attention_mask (seq_len,), returns (seq_len, hidden_size)
    batch = hidden_states.shape[0]
    keys = jax.random.split(key, batch)
    if attention_mask is None:
        fn = lambda hs, k: module(hs, key=k)
        return jax.vmap(fn)(hidden_states, keys)
    else:
        fn = lambda hs, am, k: module(hs, am, key=k)
        return jax.vmap(fn)(hidden_states, attention_mask, keys)


# --------------------------
# Tests
# --------------------------


def test_bert_embeddings_equivalence():
    cfg = make_config()
    seq_len = 9
    bs = 1

    # Inputs
    torch_input_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
    torch_position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0)
    torch_token_type_ids = torch.zeros((bs, seq_len), dtype=torch.long)

    # Torch
    th_emb = TorchBertEmbeddings(cfg)
    th_emb.eval()
    with torch.no_grad():
        th_out = th_emb(
            input_ids=torch_input_ids,
            token_type_ids=torch_token_type_ids,
            position_ids=torch_position_ids,
        )

    # JAX
    key = jax.random.key(0)
    jx_emb = BertEmbeddings(cfg, key=key)
    jx_emb = copy_embedding_weights(jx_emb, th_emb)

    jx_out = vmap_embeddings(
        jx_emb,
        input_ids=arr(torch_input_ids.numpy()),
        position_ids=arr(torch_position_ids.numpy()),
        token_type_ids=arr(torch_token_type_ids.numpy()),
        key=key,
    )

    compare_close(jx_out, th_out.numpy(), atol=1e-4, rtol=1e-4)


def test_self_attention_equivalence_no_mask_random():
    # No attention mask, random hidden states
    cfg = make_config(hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    cfg._attn_implementation = "eager"
    bs, seq_len = 2, 8
    key = jax.random.key(42)

    # Inputs
    hidden_states_jx = jax.random.normal(jax.random.key(1), (bs, seq_len, cfg.hidden_size))
    hidden_states_th = torch.tensor(arr(hidden_states_jx), dtype=torch.float32)

    # Torch
    th_sa = TorchBertSelfAttention(cfg)
    th_sa.eval()
    with torch.no_grad():
        th_out = th_sa(hidden_states_th, attention_mask=None)[0]  # (bs, seq, hidden)

    # JAX
    jx_sa = BertSelfAttention(cfg, key=key)
    jx_sa = copy_self_attn_weights(jx_sa, th_sa)
    jx_out = vmap_call(jx_sa, hidden_states_jx, attention_mask=None, key=key)

    compare_close(jx_out, th_out.numpy(), atol=1e-3, rtol=1e-3)


def _tokenizer_or_skip():
    try:
        tok = AutoTokenizer.from_pretrained("bert-base-uncased")
        return tok
    except Exception as e:  # offline or not available
        pytest.skip(f"Skipping tokenizer-based test: {e}")


def _prep_tokenized_inputs(cfg: BertConfig, texts: list[str], padding: bool):
    tok = _tokenizer_or_skip()
    enc = tok(texts, padding=padding, return_tensors="pt")
    input_ids = enc["input_ids"]  # (bs, max_len)
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    attention_mask = enc["attention_mask"]
    position_ids = make_position_ids(input_ids)
    return input_ids, token_type_ids, attention_mask, position_ids


def _mask_nonpad_rows_and_compare(jx_out: jnp.ndarray, th_out: torch.Tensor, attn_mask: torch.Tensor, atol=1e-3, rtol=1e-3):
    # Compare only valid tokens where attn_mask==1
    th_np = th_out.detach().cpu().numpy()
    bs, seqlen = attn_mask.shape
    for b in range(bs):
        valid = attn_mask[b].nonzero(as_tuple=False).squeeze(-1).tolist()
        if not valid:
            continue
        j_slice = arr(jx_out[b, valid, :])
        t_slice = th_np[b, valid, :]
        compare_close(j_slice, t_slice, atol=atol, rtol=rtol)


@pytest.mark.parametrize("texts", [["hello world", "this is a much longer input sequence"]])
def test_self_attention_equivalence_with_padding_tokenizer(texts):
    cfg = make_config()
    cfg._attn_implementation = "eager"
    key = jax.random.key(0)

    # Inputs via tokenizer (padding=True)
    input_ids, token_type_ids, attention_mask, position_ids = _prep_tokenized_inputs(cfg, texts, padding=True)

    # Torch embeddings -> hidden_states
    th_emb = TorchBertEmbeddings(cfg)
    th_emb.eval()
    with torch.no_grad():
        th_hidden = th_emb(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

    # JAX embeddings -> hidden_states
    jx_emb = BertEmbeddings(cfg, key=key)
    jx_emb = copy_embedding_weights(jx_emb, th_emb)
    jx_hidden = vmap_embeddings(
        jx_emb,
        arr(input_ids.numpy()),
        arr(position_ids.numpy()),
        arr(token_type_ids.numpy()),
        key=key,
    )

    # Torch SelfAttention
    th_sa = TorchBertSelfAttention(cfg)
    th_sa.eval()
    with torch.no_grad():
        th_out = th_sa(th_hidden, attention_mask=extend_attention_mask(attention_mask))[0]

    # JAX SelfAttention (per sample, pass 1D mask)
    jx_sa = BertSelfAttention(cfg, key=key)
    jx_sa = copy_self_attn_weights(jx_sa, th_sa)
    jx_out = vmap_call(jx_sa, jx_hidden, arr(attention_mask.numpy()), key=key)

    # Compare only non-pad rows to avoid masked-row undefined behavior
    _mask_nonpad_rows_and_compare(jx_out, th_out, attention_mask)


def test_bert_attention_equivalence_no_mask_random():
    cfg = make_config()
    cfg._attn_implementation = "eager"
    bs, seq_len = 2, 10
    key = jax.random.key(123)

    hidden_states_jx = jax.random.normal(jax.random.key(321), (bs, seq_len, cfg.hidden_size))
    hidden_states_th = torch.tensor(arr(hidden_states_jx), dtype=torch.float32)

    th_attn = TorchBertAttention(cfg)
    th_attn.eval()
    with torch.no_grad():
        th_out = th_attn(hidden_states_th, attention_mask=None)[0]

    jx_attn = BertAttention(cfg, key=key)
    jx_attn = copy_attention_weights(jx_attn, th_attn)
    jx_out = vmap_call(jx_attn, hidden_states_jx, None, key=key)

    compare_close(jx_out, th_out.numpy(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("texts", [["short text", "this is a considerably longer piece of text"]])
def test_bert_attention_equivalence_with_padding_tokenizer(texts):
    cfg = make_config()
    cfg._attn_implementation = "eager"
    key = jax.random.key(7)

    input_ids, token_type_ids, attention_mask, position_ids = _prep_tokenized_inputs(cfg, texts, padding=True)

    th_emb = TorchBertEmbeddings(cfg)
    th_emb.eval()
    with torch.no_grad():
        th_hidden = th_emb(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

    jx_emb = BertEmbeddings(cfg, key=key)
    jx_emb = copy_embedding_weights(jx_emb, th_emb)
    jx_hidden = vmap_embeddings(
        jx_emb,
        arr(input_ids.numpy()),
        arr(position_ids.numpy()),
        arr(token_type_ids.numpy()),
        key=key,
    )

    th_attn = TorchBertAttention(cfg)
    th_attn.eval()
    with torch.no_grad():
        th_out = th_attn(th_hidden, attention_mask=extend_attention_mask(attention_mask))[0]

    jx_attn = BertAttention(cfg, key=key)
    jx_attn = copy_attention_weights(jx_attn, th_attn)
    jx_out = vmap_call(jx_attn, jx_hidden, arr(attention_mask.numpy()), key=key)

    _mask_nonpad_rows_and_compare(jx_out, th_out, attention_mask)


def test_bert_self_output_equivalence_random():
    cfg = make_config()
    bs, seq_len = 2, 5
    key = jax.random.key(555)

    hidden_states_jx = jax.random.normal(jax.random.key(556), (bs, seq_len, cfg.hidden_size))
    input_tensor_jx = jax.random.normal(jax.random.key(557), (bs, seq_len, cfg.hidden_size))

    hidden_states_th = torch.tensor(arr(hidden_states_jx), dtype=torch.float32)
    input_tensor_th = torch.tensor(arr(input_tensor_jx), dtype=torch.float32)

    th_so = TorchBertSelfOutput(cfg)
    th_so.eval()
    with torch.no_grad():
        th_out = th_so(hidden_states_th, input_tensor_th)

    jx_so = BertSelfOutput(cfg, key=key)
    jx_so = copy_self_output_weights(jx_so, th_so)

    # vmap over batch
    def call_so(hs, it, k):
        return jx_so(hs, it, key=k)

    keys = jax.random.split(key, bs)
    jx_out = jax.vmap(call_so)(hidden_states_jx, input_tensor_jx, keys)

    compare_close(jx_out, th_out.numpy(), atol=1e-3, rtol=1e-3)


def test_bert_intermediate_equivalence_random():
    cfg = make_config()
    bs, seq_len = 2, 7
    key = jax.random.key(600)

    hidden_states_jx = jax.random.normal(jax.random.key(601), (bs, seq_len, cfg.hidden_size))
    hidden_states_th = torch.tensor(arr(hidden_states_jx), dtype=torch.float32)

    th_inter = TorchBertIntermediate(cfg)
    th_inter.eval()
    with torch.no_grad():
        th_out = th_inter(hidden_states_th)

    jx_inter = BertIntermediate(cfg, key=key)
    jx_inter = copy_intermediate_weights(jx_inter, th_inter)
    jx_out = vmap_call(jx_inter, hidden_states_jx, None, key=key)

    compare_close(jx_out, th_out.numpy(), atol=1e-3, rtol=1e-3)


def test_bert_output_equivalence_random():
    cfg = make_config()
    bs, seq_len = 2, 7
    key = jax.random.key(700)

    # intermediate -> hidden transition
    hidden_states_jx = jax.random.normal(jax.random.key(701), (bs, seq_len, cfg.intermediate_size))
    input_tensor_jx = jax.random.normal(jax.random.key(702), (bs, seq_len, cfg.hidden_size))
    hidden_states_th = torch.tensor(arr(hidden_states_jx), dtype=torch.float32)
    input_tensor_th = torch.tensor(arr(input_tensor_jx), dtype=torch.float32)

    th_out_mod = TorchBertOutput(cfg)
    th_out_mod.eval()
    with torch.no_grad():
        th_out = th_out_mod(hidden_states_th, input_tensor_th)

    jx_out_mod = BertOutput(cfg, key=key)
    jx_out_mod = copy_output_weights(jx_out_mod, th_out_mod)

    def call_out(hs, it, k):
        return jx_out_mod(hs, it, key=k)

    keys = jax.random.split(key, bs)
    jx_out = jax.vmap(call_out)(hidden_states_jx, input_tensor_jx, keys)

    compare_close(jx_out, th_out.numpy(), atol=1e-3, rtol=1e-3)


def test_bert_layer_equivalence_no_mask_random():
    cfg = make_config()
    cfg._attn_implementation = "eager"
    bs, seq_len = 2, 7
    key = jax.random.key(99)

    hidden_states_jx = jax.random.normal(jax.random.key(100), (bs, seq_len, cfg.hidden_size))
    hidden_states_th = torch.tensor(arr(hidden_states_jx), dtype=torch.float32)

    th_layer = TorchBertLayer(cfg)
    th_layer.eval()
    with torch.no_grad():
        th_out = th_layer(hidden_states_th)[0]

    jx_layer = BertLayer(cfg, rngs=key)
    jx_layer = copy_layer_weights(jx_layer, th_layer)
    # BertLayer requires an attention_mask positional arg; supply ones for no padding
    attn_ones = jnp.ones((bs, seq_len), dtype=jnp.int32)
    jx_out = vmap_call(jx_layer, hidden_states_jx, attn_ones, key=key)

    compare_close(jx_out, th_out.numpy(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("texts", [["alpha bravo", "charlie delta echo foxtrot golf"]])
def test_bert_layer_equivalence_with_padding_tokenizer(texts):
    cfg = make_config()
    cfg._attn_implementation = "eager"
    key = jax.random.key(1234)

    input_ids, token_type_ids, attention_mask, position_ids = _prep_tokenized_inputs(cfg, texts, padding=True)

    th_emb = TorchBertEmbeddings(cfg)
    th_emb.eval()
    with torch.no_grad():
        th_hidden = th_emb(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

    jx_emb = BertEmbeddings(cfg, key=key)
    jx_emb = copy_embedding_weights(jx_emb, th_emb)
    jx_hidden = vmap_embeddings(
        jx_emb,
        arr(input_ids.numpy()),
        arr(position_ids.numpy()),
        arr(token_type_ids.numpy()),
        key=key,
    )

    th_layer = TorchBertLayer(cfg)
    th_layer.eval()
    with torch.no_grad():
        th_out = th_layer(th_hidden, attention_mask=extend_attention_mask(attention_mask))[0]

    jx_layer = BertLayer(cfg, rngs=key)
    jx_layer = copy_layer_weights(jx_layer, th_layer)
    jx_out = vmap_call(jx_layer, jx_hidden, arr(attention_mask.numpy()), key=key)

    _mask_nonpad_rows_and_compare(jx_out, th_out, attention_mask)


def test_bert_encoder_equivalence_no_mask_random():
    # Multiple layers
    cfg = make_config(num_hidden_layers=2)
    cfg._attn_implementation = "eager"
    bs, seq_len = 2, 6
    key = jax.random.key(2024)

    hidden_states_jx = jax.random.normal(jax.random.key(777), (bs, seq_len, cfg.hidden_size))
    hidden_states_th = torch.tensor(arr(hidden_states_jx), dtype=torch.float32)

    th_encoder = TorchBertEncoder(cfg)
    th_encoder.eval()
    with torch.no_grad():
        th_out = th_encoder(hidden_states_th, attention_mask=None)[0]

    jx_encoder = BertEncoder(cfg, key=key)
    jx_encoder = copy_encoder_weights(jx_encoder, th_encoder)
    jx_out = vmap_call(jx_encoder, hidden_states_jx, None, key=key)

    compare_close(jx_out, th_out.numpy(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("texts", [["one two", "one two three four five six seven"]])
def test_bert_encoder_equivalence_with_padding_tokenizer(texts):
    cfg = make_config(num_hidden_layers=2)
    cfg._attn_implementation = "eager"
    key = jax.random.key(888)

    input_ids, token_type_ids, attention_mask, position_ids = _prep_tokenized_inputs(cfg, texts, padding=True)

    th_emb = TorchBertEmbeddings(cfg)
    th_emb.eval()
    with torch.no_grad():
        th_hidden = th_emb(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)

    jx_emb = BertEmbeddings(cfg, key=key)
    jx_emb = copy_embedding_weights(jx_emb, th_emb)
    jx_hidden = vmap_embeddings(
        jx_emb,
        arr(input_ids.numpy()),
        arr(position_ids.numpy()),
        arr(token_type_ids.numpy()),
        key=key,
    )

    th_encoder = TorchBertEncoder(cfg)
    th_encoder.eval()
    with torch.no_grad():
        th_out = th_encoder(th_hidden, attention_mask=extend_attention_mask(attention_mask))[0]

    jx_encoder = BertEncoder(cfg, key=key)
    jx_encoder = copy_encoder_weights(jx_encoder, th_encoder)
    jx_out = vmap_call(jx_encoder, jx_hidden, arr(attention_mask.numpy()), key=key)

    _mask_nonpad_rows_and_compare(jx_out, th_out, attention_mask)


def copy_mlm_head_weights(jx_mlm: BertForMaskedLM, th_mlm: TorchBertForMaskedLM):
    th_head = th_mlm.cls.predictions
    jx_mlm = set_attr(jx_mlm, "cls.predictions.transform.dense.weight", t2np(th_head.transform.dense.weight))
    jx_mlm = set_attr(jx_mlm, "cls.predictions.transform.dense.bias", t2np(th_head.transform.dense.bias))
    jx_mlm = set_attr(jx_mlm, "cls.predictions.transform.LayerNorm.weight", t2np(th_head.transform.LayerNorm.weight))
    jx_mlm = set_attr(jx_mlm, "cls.predictions.transform.LayerNorm.bias", t2np(th_head.transform.LayerNorm.bias))
    jx_mlm = set_attr(jx_mlm, "cls.predictions.bias", t2np(th_head.bias))
    return jx_mlm


def copy_full_bert_for_mlm(jx_mlm: BertForMaskedLM, th_mlm: TorchBertForMaskedLM):
    jx_mlm = eqx.tree_at(
        lambda m: m.bert.embeddings,
        jx_mlm,
        copy_embedding_weights(jx_mlm.bert.embeddings, th_mlm.bert.embeddings),
    )
    jx_mlm = eqx.tree_at(
        lambda m: m.bert.encoder,
        jx_mlm,
        copy_encoder_weights(jx_mlm.bert.encoder, th_mlm.bert.encoder),
    )
    jx_mlm = copy_mlm_head_weights(jx_mlm, th_mlm)
    return jx_mlm


def test_bert_mlm_equivalence_single_no_padding():
    cfg = make_config(num_hidden_layers=2)
    cfg._attn_implementation = "eager"
    key = jax.random.key(4242)
    texts = ["masked language modeling small check"]

    tok = _tokenizer_or_skip()
    enc = tok(texts, padding=False, return_tensors="pt")
    input_ids = enc["input_ids"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    attention_mask = enc["attention_mask"]
    position_ids = make_position_ids(input_ids)

    th_mlm = TorchBertForMaskedLM(cfg)
    th_mlm.eval()
    with torch.no_grad():
        th_logits = th_mlm(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ).logits

    jx_mlm = BertForMaskedLM(cfg, key=key)
    jx_mlm = copy_full_bert_for_mlm(jx_mlm, th_mlm)

    # vmap over batch (bs=1 here is fine)
    def call_mlm(ids, pos, tt, am, k):
        return jx_mlm(ids, pos, tt, am, key=k)

    bs = input_ids.shape[0]
    keys = jax.random.split(key, bs)
    jx_logits = jax.vmap(call_mlm)(
        arr(input_ids.numpy()), arr(position_ids.numpy()), arr(token_type_ids.numpy()), arr(attention_mask.numpy()), keys
    )

    compare_close(jx_logits, th_logits.numpy(), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("texts", [["short example", "this example is a lot longer and padded"]])
def test_bert_mlm_equivalence_with_padding(texts):
    cfg = make_config(num_hidden_layers=2)
    cfg._attn_implementation = "eager"
    key = jax.random.key(4243)

    tok = _tokenizer_or_skip()
    enc = tok(texts, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    attention_mask = enc["attention_mask"]
    position_ids = make_position_ids(input_ids)

    th_mlm = TorchBertForMaskedLM(cfg)
    th_mlm.eval()
    with torch.no_grad():
        th_logits = th_mlm(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ).logits

    jx_mlm = BertForMaskedLM(cfg, key=key)
    jx_mlm = copy_full_bert_for_mlm(jx_mlm, th_mlm)

    def call_mlm(ids, pos, tt, am, k):
        return jx_mlm(ids, pos, tt, am, key=k)

    bs = input_ids.shape[0]
    keys = jax.random.split(key, bs)
    jx_logits = jax.vmap(call_mlm)(
        arr(input_ids.numpy()), arr(position_ids.numpy()), arr(token_type_ids.numpy()), arr(attention_mask.numpy()), keys
    )

    # Compare only non-pad rows to avoid query-masking differences
    bs, seqlen = attention_mask.shape
    for b in range(bs):
        valid = attention_mask[b].nonzero(as_tuple=False).squeeze(-1).tolist()
        if not valid:
            continue
        compare_close(jx_logits[b, valid, :], th_logits.numpy()[b, valid, :], atol=1e-3, rtol=1e-3)
