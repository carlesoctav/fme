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
text = "Hello world"

print(f"Loading tokenizer and model from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
th_model = TorchModernBertModel.from_pretrained(model_name)
th_model.eval()

encoded = tokenizer(
    [text],
    padding=False,
    return_tensors="pt",
)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

print(f"Input: {text}")
print(f"Input IDs: {input_ids}")
print(f"Input shape: {input_ids.shape}")

# Get PyTorch embeddings
with torch.no_grad():
    th_embeds = th_model.embeddings(input_ids).numpy()

print(f"\nPyTorch embeddings shape: {th_embeds.shape}")
print(f"PyTorch embeddings stats: mean={th_embeds.mean():.6f}, std={th_embeds.std():.6f}")

# Create JAX model and copy weights
key = jax.random.key(42)
jx_model = ModernBertModel(th_model.config, key=key)
jx_model = copy_modernbert_weights(jx_model, th_model)

# Get JAX embeddings
jx_embeds = jx_model.embeddings(
    jnp.asarray(input_ids.numpy()),
    key=jax.random.key(0),
)

print(f"\nJAX embeddings shape: {jx_embeds.shape}")
print(f"JAX embeddings stats: mean={jx_embeds.mean():.6f}, std={jx_embeds.std():.6f}")

# Compare embeddings
embed_diff = np.abs(jx_embeds - th_embeds)
print(f"\nEmbedding difference: mean={embed_diff.mean():.6f}, max={embed_diff.max():.6f}")

try:
    np.testing.assert_allclose(jx_embeds, th_embeds, atol=1e-5, rtol=1e-5)
    print("✅ Embeddings match!")
except AssertionError as e:
    print(f"❌ Embeddings don't match: {e}")
