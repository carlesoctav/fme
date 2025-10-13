import equinox as eqx
import jax
import jax.numpy as jnp
from transformers import ModernBertModel as TorchModernBertModel

from src.models.modernbert.modeling_modernbert import ModernBertModel

model_name = "answerdotai/ModernBERT-base"
th_model = TorchModernBertModel.from_pretrained(model_name)

print("Creating JAX model...")
key = jax.random.key(42)
jx_model = ModernBertModel(th_model.config, key=key)

print("Attempting to set inference mode...")
try:
    jx_model_inf = eqx.nn.inference_mode(jx_model)
    print("✅ Success!")
except Exception as e:
    print(f"❌ Failed: {type(e).__name__}: {e}")
    
    # Try to debug what's wrong
    import traceback
    traceback.print_exc()
