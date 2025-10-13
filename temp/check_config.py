import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from transformers import AutoTokenizer, ModernBertModel as TorchModernBertModel

model_name = "answerdotai/ModernBERT-base"
th_model = TorchModernBertModel.from_pretrained(model_name)

print("Config:")
print(f"  _attn_implementation: {th_model.config._attn_implementation}")
print(f"  global_attn_every_n_layers: {th_model.config.global_attn_every_n_layers}")
print(f"  local_attention: {th_model.config.local_attention}")
print(f"  num_hidden_layers: {th_model.config.num_hidden_layers}")
print(f"  hidden_size: {th_model.config.hidden_size}")
print(f"  num_attention_heads: {th_model.config.num_attention_heads}")
print(f"  rope_theta: {th_model.config.rope_theta}")

# Check which layers are global vs local
for i in range(th_model.config.num_hidden_layers):
    is_global = (i + 1) % th_model.config.global_attn_every_n_layers == 0
    print(f"  Layer {i}: {'GLOBAL' if is_global else 'LOCAL'}")
