"""Test filter_jit with sharding."""

import jax
import equinox as eqx
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert import BertModel
from functools import partial

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

devices = jax.devices()
mesh = Mesh(np.array(devices[:2]), axis_names=("tp",))
key = jax.random.PRNGKey(42)

print("Getting abstract model for structure...")
abstract_model = eqx.filter_eval_shape(BertModel, config, key=key)

def get_replicated_sharding(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return NamedSharding(mesh, P())
    return None

out_shardings = jax.tree.map(get_replicated_sharding, abstract_model)

print("Creating model with filter_jit + out_shardings...")
@partial(eqx.filter_jit, out=out_shardings)
def create_model_sharded(key):
    return BertModel(config, key=key)

model = create_model_sharded(key)
sample = model.encoder.layer[0].attention.self.query.weight
print(f"Type: {type(sample)}")
print(f"Is concrete: {not isinstance(sample, jax.core.Tracer)}")
print(f"Sharding: {sample.sharding}")
