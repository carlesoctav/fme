import equinox as eqx
import jax
from transformers import ModernBertConfig

from src.models.modernbert.modeling_modernbert import ModernBertModel

# Test 1: Fresh config (like in synthetic tests)
print("Test 1: Fresh config (synthetic)")
fresh_config = ModernBertConfig(
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
fresh_config._attn_implementation = "eager"

key = jax.random.key(42)
jx_model_fresh = ModernBertModel(fresh_config, key=key)

try:
    jx_model_fresh_inf = eqx.nn.inference_mode(jx_model_fresh)
    print("✅ Fresh config works!")
except Exception as e:
    print(f"❌ Fresh config failed: {e}")

# Test 2: Pretrained config
print("\nTest 2: Pretrained config")
from transformers import ModernBertModel as TorchModernBertModel

model_name = "answerdotai/ModernBERT-base"
th_model = TorchModernBertModel.from_pretrained(model_name)
pretrained_config = th_model.config

key2 = jax.random.key(43)
jx_model_pretrained = ModernBertModel(pretrained_config, key=key2)

try:
    jx_model_pretrained_inf = eqx.nn.inference_mode(jx_model_pretrained)
    print("✅ Pretrained config works!")
except Exception as e:
    print(f"❌ Pretrained config failed: {e}")
