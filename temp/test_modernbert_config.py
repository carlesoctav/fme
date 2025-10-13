from transformers import ModernBertModel as TorchModernBertModel
import jax.tree_util as jtu

model_name = "answerdotai/ModernBERT-base"
th_model = TorchModernBertModel.from_pretrained(model_name)

print(f"Config type: {type(th_model.config)}")
print(f"Config._attn_implementation: {th_model.config._attn_implementation}")
print(f"Type: {type(th_model.config._attn_implementation)}")

# Check what attributes exist on the config
print("\nAll config attributes:")
for attr in dir(th_model.config):
    if not attr.startswith('_'):
        val = getattr(th_model.config, attr)
        if isinstance(val, bool):
            print(f"  {attr}: {val} (bool)")
