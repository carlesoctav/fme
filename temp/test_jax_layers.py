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

print(f"global_attn_every_n_layers = {jx_model.config.global_attn_every_n_layers}")
print(f"global_rope_theta = {jx_model.config.global_rope_theta}")
print(f"local_rope_theta = {jx_model.config.local_rope_theta}")
print()

for i in range(min(10, len(jx_model.encoder.layers))):
    layer = jx_model.encoder.layers[i]
    print(f"Layer {i}: use_global={layer.use_global}")
    # Check first element of rtheta to see if it's different
    print(f"  RoPE rtheta[0, 0]: {layer.attention.rotary_emb.rtheta[0, 0]}")

print("\nComparing layer 0 (global) vs layer 1 (local) RoPE:")
rope0 = jx_model.encoder.layers[0].attention.rotary_emb.rtheta
rope1 = jx_model.encoder.layers[1].attention.rotary_emb.rtheta
print(f"Layer 0 rtheta shape: {rope0.shape}, [0,0]={rope0[0, 0]}")
print(f"Layer 1 rtheta shape: {rope1.shape}, [0,0]={rope1[0, 0]}")
print(f"Are they the same? {jnp.array_equal(rope0, rope1)}")
