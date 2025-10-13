import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from transformers import ModernBertModel as TorchModernBertModel
from src.models.modernbert.modeling_modernbert import ModernBertModel

model_name = "answerdotai/ModernBERT-base"
th_model = TorchModernBertModel.from_pretrained(model_name)

key = jax.random.key(42)
jx_model = ModernBertModel(th_model.config, key=key)

rope0 = jx_model.encoder.layers[0].attention.rotary_emb.rtheta
rope1 = jx_model.encoder.layers[1].attention.rotary_emb.rtheta

print("Layer 0 (GLOBAL) rtheta[1, :5]:", rope0[1, :5])
print("Layer 1 (LOCAL) rtheta[1, :5]:", rope1[1, :5])
print("\nDifference:", jnp.abs(rope0[1, :5] - rope1[1, :5]))

# Check position 10
print("\nLayer 0 (GLOBAL) rtheta[10, :5]:", rope0[10, :5])
print("Layer 1 (LOCAL) rtheta[10, :5]:", rope1[10, :5])
print("Difference:", jnp.abs(rope0[10, :5] - rope1[10, :5]))
