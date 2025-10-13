import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    ModernBertModel as TorchModernBertModel,
)

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


model_name = "answerdotai/ModernBERT-base"
text1 = "hallo nama saya carles"
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

print(f"Input shape: {input_ids.shape}")
print(f"Attention mask shape: {attention_mask.shape}")
print(f"Input IDs:\n{input_ids}")
print(f"Attention mask:\n{attention_mask}")

key = jax.random.key(42)
jx_model = ModernBertModel(th_model.config, key=key)
jx_model = copy_modernbert_weights(jx_model, th_model)
jx_model = eqx.nn.inference_mode(jx_model)

print("\n=== EMBEDDINGS ===")
with torch.no_grad():
    th_emb = th_model.embeddings(input_ids)
    print(f"PyTorch embeddings shape: {th_emb.shape}")
    print(f"PyTorch embeddings sample: {th_emb[0, 0, :5]}")

jx_emb = jx_model.embeddings(jnp.asarray(input_ids.numpy()))
print(f"JAX embeddings shape: {jx_emb.shape}")
print(f"JAX embeddings sample: {jx_emb[0, 0, :5]}")

diff = jnp.max(jnp.abs(jx_emb - th_emb.numpy()))
print(f"Embeddings max diff: {diff}")

print("\n=== LAYER BY LAYER ===")
th_hidden = th_emb
jx_hidden = jx_emb

jx_attention_mask = jnp.asarray(attention_mask.numpy(), dtype=np.bool_)

for i in range(len(th_model.layers)):
    print(f"\n--- Layer {i} ---")
    th_layer = th_model.layers[i]
    jx_layer = jx_model.encoder.layers[i]
    
    print(f"Layer {i} uses global attention: {jx_layer.use_global}")
    
    with torch.no_grad():
        th_hidden = th_layer(
            th_hidden,
            attention_mask=attention_mask,
        )
    
    jx_hidden = jx_layer(
        jx_hidden,
        attention_mask=jx_attention_mask,
        key=key,
    )
    
    diff = jnp.max(jnp.abs(jx_hidden - th_hidden.numpy()))
    print(f"Layer {i} output max diff: {diff}")
    print(f"PyTorch layer {i} output sample: {th_hidden[0, 0, :5]}")
    print(f"JAX layer {i} output sample: {jx_hidden[0, 0, :5]}")
    
    if diff > 0.1:
        print(f"!!! DIVERGENCE DETECTED AT LAYER {i} !!!")
        break

print("\n=== FINAL NORM ===")
with torch.no_grad():
    th_final = th_model.final_norm(th_hidden)
    
jx_final = jx_model.final_norm(jx_hidden)

diff = jnp.max(jnp.abs(jx_final - th_final.numpy()))
print(f"Final norm max diff: {diff}")
