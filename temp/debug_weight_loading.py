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
th_model = TorchModernBertModel.from_pretrained(model_name)
th_model.eval()

key = jax.random.key(42)
jx_model = ModernBertModel(th_model.config, key=key)

# Copy weights
jx_model = copy_modernbert_weights(jx_model, th_model)

# Check weights AFTER copying
print("=== WEIGHT CHECK ===")
w_jax = jx_model.encoder.layers[0].attention.Wqkv.weight
w_torch = th_model.layers[0].attn.Wqkv.weight.detach().numpy()

print(f"JAX Layer 0 Wqkv type: {type(w_jax)}")
print(f"JAX Layer 0 Wqkv shape: {w_jax.shape if hasattr(w_jax, 'shape') else 'N/A'}")
print(f"PyTorch Layer 0 Wqkv shape: {w_torch.shape}")

# If it's a DArray, get the value
if hasattr(w_jax, 'value'):
    w_jax = w_jax.value
elif hasattr(w_jax, '_value'):
    w_jax = w_jax._value
    
print(f"\nAfter extraction:")
print(f"JAX Layer 0 Wqkv shape: {w_jax.shape}")
print(f"JAX Layer 0 Wqkv[0, :5]: {w_jax[0, :5]}")
print(f"PyTorch Layer 0 Wqkv[0, :5]: {w_torch[0, :5]}")

diff = np.abs(w_jax - w_torch)
print(f"\nMax diff: {diff.max():.10f}")
print(f"Weights match: {diff.max() < 1e-6}")

# Check if weights are transposed
print(f"\nTrying transpose:")
diff_T = np.abs(w_jax - w_torch.T)
print(f"Max diff (transposed): {diff_T.max():.10f}")
print(f"Weights match (transposed): {diff_T.max() < 1e-6}")
