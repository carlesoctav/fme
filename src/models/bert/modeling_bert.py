from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import field
from jaxtyping import Array, Float, Int, PRNGKeyArray
import transformers
from transformers.models.bert.configuration_bert import BertConfig
from jax import P
from jax.interpreters.pxla import thread_resources
from src import HuggingFaceCompatibleModule, nn
from src.nn import functional as F


Pytree = Any


class BertEmbeddings(eqx.Module):
    word_embeddings: nn.Embedding
    position_embeddings: nn.Embedding
    token_type_embeddings: nn.Embedding
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        word_key, position_key, token_type_key, layer_norm_key, dropout_key = (
            jax.random.split(key, 5)
        )

        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=word_key,
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=config.max_position_embeddings,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=position_key,
        )

        self.token_type_embeddings = nn.Embedding(
            num_embeddings=config.type_vocab_size,
            embedding_dim=config.hidden_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=token_type_key,
        )

        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=dtype,
            params_dtype=params_dtype,
            key=layer_norm_key,
        )
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob, dtype=dtype, params_dtype=params_dtype)

    def __call__(
        self,
        input_ids: Int[Array, " ..."],
        position_ids: Int[Array, " ..."],
        token_type_ids: Int[Array, " ..."],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " ... hidden_size"]:
        d_key = jax.random.split(key, 1)[0] if key is not None else None

        inputs_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeddings + token_type_embeddings + position_embeddings

        embeddings = self.LayerNorm(embeddings)

        embeddings = self.dropout(embeddings, key=d_key)

        return embeddings


class BertSelfAttention(eqx.Module):
    query: nn.Linear
    value: nn.Linear
    key: nn.Linear
    dropout: nn.Dropout
    num_attention_heads: int = field(static=True)
    attention_head_size: int = field(static=True)
    all_head_size: int = field(static=True)
    btnh_spec: Any = field(static=True, default=None)
    bsnh_spec: Any = field(static=True, default=None)
    bnts_spec: Any = field(static=True, default=None)
    self_attn_out_spec: Any = field(static=True, default=None)

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
        btnh_spec: Any | None = None,
        bsnh_spec: Any | None = None,
        bnts_spec: Any | None = None,
        self_attn_out_spec: Any | None = None, 
    ):
        q_key, v_key, k_key = jax.random.split(key, 3)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size})"
                "is not a multiple of the number of attention"
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob, dtype=dtype, params_dtype=params_dtype)

        self.query = nn.Linear(
            config.hidden_size,
            self.all_head_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=q_key,
        )
        self.key = nn.Linear(
            config.hidden_size,
            self.all_head_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=k_key,
        )
        self.value = nn.Linear(
            config.hidden_size,
            self.all_head_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=v_key,
        )

        self.btnh_spec = btnh_spec
        self.bsnh_spec = bsnh_spec
        self.bnts_spec = bnts_spec
        self.self_attn_out_spec = self_attn_out_spec

    def __call__(
        self,
        hidden_states: Float[Array, " ... seq_len hidden_size"],
        attention_mask: Int[Array, " ... seq_len hidden_size"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, " ... seq_len hidden_size"]: 

        dropout_key = jax.random.split(key, 1)[0] if key is not None else None

        if hidden_states.ndim ==1:
            raise ValueError("hidden_states must have (..., seq_len, hidden_size) shape")

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        def _to_heads(x):
            return x.reshape(*x.shape[:-1], self.num_attention_heads, self.attention_head_size)

        q_heads = _to_heads(q)
        k_heads = _to_heads(k)
        v_heads = _to_heads(v)


        if attention_mask is not None:
            attn_mask = F.make_4D_attention_mask(attention_mask, self.num_attention_heads)
            attn_heads = F.dot_product_attention(
                q_heads, k_heads, v_heads, mask=attn_mask, dropout=self.dropout, key=dropout_key
            )
        else:
            attn_heads = F.dot_product_attention(q_heads, k_heads, v_heads, dropout=self.dropout, key=dropout_key) 

        attn = attn_heads.reshape(*attn_heads.shape[:-2], self.all_head_size)
        return attn


class BertSelfOutput(eqx.Module):
    dense: nn.Linear
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        dense_key, layer_key = jax.random.split(key, 2)
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=dense_key,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=dtype,
            params_dtype=params_dtype,
            key=layer_key,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob, dtype=dtype, params_dtype=params_dtype)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        input_tensor: Float[Array, "seq_len hidden_size"],
        *,
        key: PRNGKeyArray | None = None,
    ):
        d_key = jax.random.split(key, 1)[0] if key is not None else None

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, key=d_key)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(eqx.Module):
    self: BertSelfAttention
    output: BertSelfOutput

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        self_key, output_key = jax.random.split(key, 2)
        self.self = BertSelfAttention(
            config, dtype=dtype, params_dtype=params_dtype, key=self_key
        )

        self.output = BertSelfOutput(
            config, dtype=dtype, params_dtype=params_dtype, key=output_key
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        attention_mask: Int[Array, " seq_len"] | None = None,
        /,
        *,
        key: PRNGKeyArray | None = None,
    ):
        self_key, output_key = (
            jax.random.split(key, 2) if key is not None else (None, None)
        )
        self_output = self.self(hidden_states, attention_mask, key=self_key) 
        attention_output = self.output(self_output, hidden_states, key=output_key)
        return attention_output


class BertIntermediate(eqx.Module):
    dense: nn.Linear
    # intermediate_act_fn: Callable  todo: think about this later

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        dense_key = jax.random.split(key, 1)[0]
        self.dense = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=dense_key,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len intermediate_size"]:
        hidden_states = self.dense(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)

        return hidden_states


class BertOutput(eqx.Module):
    dense: nn.Linear
    LayerNorm: nn.LayerNorm
    dropout: nn.Dropout

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        dense_key, layer_norm_key, dropout_key = jax.random.split(key, 3)
        self.dense = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=dense_key,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=dtype,
            params_dtype=params_dtype,
            key=layer_norm_key,
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob, dtype=dtype, params_dtype=params_dtype)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len intermediate_size"],
        input_tensor: Float[Array, "seq_len hidden_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        d_key = jax.random.split(key, 1)[0] if key is not None else None
        hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states, key=d_key)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# thinking about layer
class BertLayer(eqx.Module):
    attention: BertAttention
    intermediate: BertIntermediate
    output: BertOutput

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        rngs: PRNGKeyArray,
    ):
        attention_key, intermediate_key, output_key = jax.random.split(rngs, 3)

        self.attention = BertAttention(
            config, dtype=dtype, params_dtype=params_dtype, key=attention_key
        )
        self.intermediate = BertIntermediate(
            config, dtype=dtype, params_dtype=params_dtype, key=intermediate_key
        )
        self.output = BertOutput(
            config, dtype=dtype, params_dtype=params_dtype, key=output_key
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        attention_mask: Float[Array, " seq_len"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ):
        atention_key, intermediate_key, output_key = (
            jax.random.split(key, 3) if key is not None else (None, None, None)
        )

        attention_output = self.attention(
            hidden_states, attention_mask, key=atention_key
        )
        intermediate_output = self.intermediate(attention_output, key=intermediate_key)
        layer_output = self.output(
            intermediate_output, attention_output, key=output_key
        )
        return layer_output


class BertEncoder(eqx.Module):
    layer: list[BertLayer]

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        self.layer = []
        encoder_keys = jax.random.split(key, config.num_hidden_layers)
        for i in range(config.num_hidden_layers):
            self.layer.append(
                BertLayer(
                    config,
                    dtype=dtype,
                    params_dtype=params_dtype,
                    rngs=encoder_keys[i],
                )
            )

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        attention_mask: Float[Array, " seq_len"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        layer_key = jax.random.split(key, len(self.layer)) if key is not None else None
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                key=layer_key[i] if layer_key is not None else None,
            )
        return hidden_states


class BertModel(eqx.Module, HuggingFaceCompatibleModule[transformers.BertModel]):
    embeddings: BertEmbeddings
    encoder: BertEncoder
    config: BertConfig | None = field(static=True)

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        store_config=True,
        key: PRNGKeyArray,
    ):
        embedding_key, encoder_key = jax.random.split(key, 2)
        self.embeddings = BertEmbeddings(
            config, dtype=dtype, params_dtype=params_dtype, key=embedding_key
        )
        self.encoder = BertEncoder(
            config, dtype=dtype, params_dtype=params_dtype, key=encoder_key
        )

        if store_config:
            self.config = config
        else:
            self.config = None

    def __call__(
        self,
        input_ids: Int[Array, " seq_len"],
        position_ids: Int[Array, " seq_len"],
        token_type_ids: Int[Array, " seq_len"],
        attention_mask: Int[Array, " seq_len"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ):
        embed_key, encoder_key = (
            jax.random.split(key, 2) if key is not None else (None, None)
        )
        hidden_states = self.embeddings(
            input_ids, position_ids, token_type_ids, key=embed_key
        )
        hidden_states = self.encoder(
            hidden_states, attention_mask=attention_mask, key=encoder_key
        )

        return hidden_states


class BertPredictionHeadTransform(eqx.Module):
    dense: nn.Linear
    LayerNorm: nn.LayerNorm

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        dense_key, ln_key = jax.random.split(key, 2)
        self.dense = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            dtype=dtype,
            params_dtype=params_dtype,
            key=dense_key,
        )
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            dtype=dtype,
            params_dtype=params_dtype,
            key=ln_key,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        hidden_states = self.dense(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(eqx.Module):
    transform: BertPredictionHeadTransform
    bias: Array

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        self.transform = BertPredictionHeadTransform(
            config, dtype=dtype, params_dtype=params_dtype, key=key
        )
        self.bias = jnp.zeros((config.vocab_size,), dtype=params_dtype)

    def __call__(
        self,
        hidden_states: Float[Array, "seq_len hidden_size"],
        embedding_weight: Float[Array, "vocab_size hidden_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len vocab_size"]:
        hs = self.transform(hidden_states, key=key)
        logits = jnp.einsum("...d, vd->...v", hs, embedding_weight)
        logits = logits + self.bias
        return logits


class BertOnlyMLMHead(eqx.Module):
    predictions: BertLMPredictionHead

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        key: PRNGKeyArray,
    ):
        self.predictions = BertLMPredictionHead(
            config, dtype=dtype, params_dtype=params_dtype, key=key
        )

    def __call__(
        self,
        sequence_output: Float[Array, "seq_len hidden_size"],
        embedding_weight: Float[Array, "vocab_size hidden_size"],
        *,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "seq_len vocab_size"]:
        return self.predictions(sequence_output, embedding_weight, key=key)


class BertForMaskedLM(eqx.Module, HuggingFaceCompatibleModule):
    bert: BertModel
    cls: BertOnlyMLMHead
    config: BertConfig | None = eqx.field(static=True)

    def __init__(
        self,
        config: BertConfig,
        *,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        store_config: bool = True,
        key: PRNGKeyArray,
    ):
        bert_key, cls_key = jax.random.split(key, 2)
        self.bert = BertModel(
            config, dtype=dtype, params_dtype=params_dtype, store_config=False, key=bert_key
        )
        self.cls = BertOnlyMLMHead(
            config, dtype=dtype, params_dtype=params_dtype, key=cls_key
        )

        if store_config:
            self.config = config
        else:
            self.config = None

    def __call__(
        self,
        input_ids: Int[Array, " ... seq_len"],
        position_ids: Int[Array, "... seq_len"],
        token_type_ids: Int[Array, "... seq_len"],
        attention_mask: Int[Array, "... seq_len"] | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ):
        bert_key, cls_key = (
            jax.random.split(key, 2) if key is not None else (None, None)
        )

        sequence_output = self.bert(
            input_ids, position_ids, token_type_ids, attention_mask, key=bert_key
        )  # (seq len, hidden_size)

        # decoder tied weights: use embedding matrix for projection
        w = self.bert.embeddings.word_embeddings.weight  # (vocab_size, hidden_size)
        logits = self.cls(sequence_output, w, key=cls_key)
        return logits

    @classmethod
    def normalize_hf_key_for_eqx(cls, key: str) -> str | None: 
        if key.startswith("cls.predictions.decoder.weight"):
            return None
        return key
