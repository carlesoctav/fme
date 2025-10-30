"""Test filter_jit with correct usage."""

import jax
import equinox as eqx
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert import BertModel

# Small model
config = BertConfig(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=128,
    vocab_size=10000,
    type_vocab_size=2,
    _attn_implementation="sdpa",
)

key = jax.random.PRNGKey(42)

print("Test 1: Just calling filter_jit on constructor")
@eqx.filter_jit
def create_model(key):
    return BertModel(config, key=key)

model1 = create_model(key)
sample1 = model1.encoder.layer[0].attention.self.query.weight
print(f"Type: {type(sample1)}")
print(f"Is concrete: {not isinstance(sample1, jax.core.Tracer)}")
print(f"Shape: {sample1.shape if hasattr(sample1, 'shape') else 'no shape'}")

if isinstance(sample1, jax.ShapeDtypeStruct):
    print("ERROR: Got ShapeDtypeStruct!")
else:
    print(f"SUCCESS: Got concrete array with dtype {sample1.dtype}")
