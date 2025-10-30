"""Test distributed init on 1B BERT."""

import time
from functools import partial
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert import BertModel

config = BertConfig(
    hidden_size=1536,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=6144,
    max_position_embeddings=512,
    vocab_size=30522,
    type_vocab_size=2,
    _attn_implementation="sdpa",
)

devices = jax.devices()
mesh = Mesh(np.array(devices[:4]).reshape((2, 2)), axis_names=("tp", "fsdp"))
key = jax.random.PRNGKey(42)

print("Creating abstract model...")
abstract_model = eqx.filter_eval_shape(BertModel, config, key=key)

print("Defining shardings...")
def get_replicated_sharding(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return NamedSharding(mesh, P())
    return None

out_shardings = jax.tree.map(get_replicated_sharding, abstract_model)
num_arrays = sum(1 for x in jax.tree.leaves(out_shardings) if x is not None)
print(f"Will shard {num_arrays} arrays")

print("Defining init function...")
@partial(jax.jit, out_shardings=out_shardings)
def init_replicated(key):
    return BertModel(config, key=key)

print("Calling init (this compiles)...")
start = time.perf_counter()
model = init_replicated(key)
print("Blocking...")
jax.block_until_ready(jax.tree.leaves(model))
end = time.perf_counter()

print(f"Time: {end - start:.2f}s")
print(f"Sample sharding: {model.encoder.layer[0].attention.self.query.weight.sharding}")
