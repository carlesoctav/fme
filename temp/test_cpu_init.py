"""Test CPU initialization then TPU sharding."""

import time
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert import BertModel

# 1B model
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

# Estimate params
embed_params = (config.vocab_size + config.max_position_embeddings + config.type_vocab_size) * config.hidden_size
layer_params = (4 * config.hidden_size**2 + 2 * config.hidden_size * config.intermediate_size) * config.num_hidden_layers
total_params = embed_params + layer_params
print(f"Estimated params: {total_params / 1e9:.2f}B")
print(f"Estimated size (fp32): {total_params * 4 / 1e9:.2f}GB")

# Force init on CPU
print("\nForcing CPU devices...")
jax.config.update('jax_platforms', 'cpu')
cpu_devices = jax.devices()
print(f"Devices: {cpu_devices}")

key = jax.random.PRNGKey(42)

print("\nInitializing on CPU...")
start = time.perf_counter()
model = BertModel(config, key=key)
end = time.perf_counter()
print(f"CPU init time: {end - start:.2f}s")

print("Model created successfully on CPU!")
