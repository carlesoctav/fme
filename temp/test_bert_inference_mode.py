import equinox as eqx
import jax
from transformers import BertModel as TorchBertModel

from src.models.bert.modeling_bert import BertModel

model_name = "google-bert/bert-base-uncased"
th_model = TorchBertModel.from_pretrained(model_name)

print("Creating JAX BERT model...")
key = jax.random.key(42)
jx_model = BertModel(th_model.config, key=key)

print("Attempting to set inference mode on BERT...")
try:
    jx_model_inf = eqx.nn.inference_mode(jx_model)
    print("✅ BERT inference mode works!")
except Exception as e:
    print(f"❌ BERT failed: {type(e).__name__}: {e}")
