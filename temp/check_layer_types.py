import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from transformers import ModernBertModel as TorchModernBertModel

model_name = "answerdotai/ModernBERT-base"
th_model = TorchModernBertModel.from_pretrained(model_name)

print(f"global_attn_every_n_layers = {th_model.config.global_attn_every_n_layers}")
print(f"global_rope_theta = {th_model.config.global_rope_theta}")
print(f"local_rope_theta = {getattr(th_model.config, 'local_rope_theta', 'NOT SET')}")
print()

for i in range(min(10, th_model.config.num_hidden_layers)):
    layer = th_model.layers[i]
    # Check with the formulas
    formula1_global = (i + 1) % th_model.config.global_attn_every_n_layers == 0
    formula2_global = i % max(th_model.config.global_attn_every_n_layers, 1) == 0
    
    print(f"Layer {i}:")
    print(f"  (i+1) % n == 0: {formula1_global} -> {'GLOBAL' if formula1_global else 'LOCAL'}")
    print(f"  i % n == 0: {formula2_global} -> {'GLOBAL' if formula2_global else 'LOCAL'}")

# Let's check the source code
print("\nChecking actual implementation:")
print(f"Layer 0 rope: {th_model.layers[0].attn.rotary_emb}")
print(f"Layer 2 rope: {th_model.layers[2].attn.rotary_emb}")
