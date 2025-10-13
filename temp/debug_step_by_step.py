import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, ModernBertModel as TorchModernBertModel

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
text1 = "hello"

tokenizer = AutoTokenizer.from_pretrained(model_name)
th_model = TorchModernBertModel.from_pretrained(model_name)
th_model.eval()

encoded = tokenizer([text1], return_tensors="pt")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

print("Input IDs:", input_ids)
print("Tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))

# PyTorch embeddings
with torch.no_grad():
    th_embeds = th_model.embeddings(input_ids)
    print(f"\nPyTorch embeddings: shape={th_embeds.shape}, mean={th_embeds.mean():.6f}")
    print(f"PyTorch embeds[0, 0, :5]: {th_embeds[0, 0, :5].numpy()}")

# JAX embeddings
key = jax.random.key(42)
jx_model = ModernBertModel(th_model.config, key=key)
jx_model = copy_modernbert_weights(jx_model, th_model)
jx_model = eqx.nn.inference_mode(jx_model)

input_ids_jax = jnp.asarray(input_ids.numpy())
jx_embeds = jx_model.embeddings(input_ids_jax)
print(f"\nJAX embeddings: shape={jx_embeds.shape}, mean={jx_embeds.mean():.6f}")
print(f"JAX embeds[0, 0, :5]: {jx_embeds[0, 0, :5]}")

embed_diff = np.abs(jx_embeds - th_embeds.numpy())
print(f"\nEmbedding diff: max={embed_diff.max():.6f}, mean={embed_diff.mean():.6f}")

if embed_diff.max() > 1e-5:
    print("⚠️ EMBEDDINGS DIFFER!")
    # Check individual components
    with torch.no_grad():
        th_tok_embeds = th_model.embeddings.tok_embeddings(input_ids)
        print(f"\nPyTorch tok_embeds[0, 0, :5]: {th_tok_embeds[0, 0, :5].numpy()}")
    
    jx_tok_embeds = jx_model.embeddings.tok_embeddings(input_ids_jax)
    print(f"JAX tok_embeds[0, 0, :5]: {jx_tok_embeds[0, 0, :5]}")
    
    tok_diff = np.abs(jx_tok_embeds - th_tok_embeds.numpy())
    print(f"Tok embedding diff: max={tok_diff.max():.10f}")
else:
    print("✓ Embeddings match!")
