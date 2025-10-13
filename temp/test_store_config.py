import equinox as eqx
import jax
from transformers import ModernBertConfig

from src.models.modernbert.modeling_modernbert import ModernBertModel

config = ModernBertConfig(
    hidden_size=48,
    num_attention_heads=6,
    intermediate_size=128,
    num_hidden_layers=2,
    vocab_size=1000,
    max_position_embeddings=96,
    global_attn_every_n_layers=1,
    local_attention=5,
    attention_dropout=0.0,
    mlp_dropout=0.0,
    embedding_dropout=0.0,
)
config._attn_implementation = "eager"

key = jax.random.key(42)

# Test 1: With config stored
print("Test 1: With config stored (store_config=True)")
jx_model_with_config = ModernBertModel(config, key=key, store_config=True)

try:
    jx_model_inf = eqx.nn.inference_mode(jx_model_with_config)
    print("✅ Works with config stored!")
except Exception as e:
    print(f"❌ Failed with config stored: {e}")

# Test 2: Without config stored
print("\nTest 2: Without config stored (store_config=False)")
jx_model_without_config = ModernBertModel(config, key=key, store_config=False)

try:
    jx_model_inf2 = eqx.nn.inference_mode(jx_model_without_config)
    print("✅ Works without config stored!")
except Exception as e:
    print(f"❌ Failed without config stored: {e}")
