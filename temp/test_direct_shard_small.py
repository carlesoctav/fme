"""Test direct device_put with sharding on model init."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert import BertModel

# Small model that fits in memory
config = BertConfig(
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=128,
    vocab_size=10000,
    type_vocab_size=2,
    _attn_implementation="sdpa",
)

devices = jax.devices()
mesh = Mesh(np.array(devices[:2]), axis_names=("tp",))
key = jax.random.PRNGKey(42)

print("Creating model normally...")
model = BertModel(config, key=key)

print("Now moving to sharded devices with device_put...")
sharding = NamedSharding(mesh, P())

def shard_array(x):
    if eqx.is_array(x):
        return jax.device_put(x, sharding)
    return x

sharded_model = jax.tree.map(shard_array, model)
print("Done!")

sample = sharded_model.encoder.layer[0].attention.self.query.weight
print(f"Sample sharding: {sample.sharding}")
