"""Measure jit compilation time for different model sizes."""

import time
import jax
import equinox as eqx
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert import BertModel
from functools import partial

devices = jax.devices()
mesh = Mesh(np.array(devices[:2]), axis_names=("tp",))

def test_size(hidden, layers, name):
    print(f"\n{name}:")
    config = BertConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=min(hidden // 64, 16),
        intermediate_size=hidden * 4,
        max_position_embeddings=128,
        vocab_size=10000,
        type_vocab_size=2,
        _attn_implementation="sdpa",
    )
    
    key = jax.random.PRNGKey(42)
    abstract_model = eqx.filter_eval_shape(BertModel, config, key=key)
    
    def get_replicated_sharding(x):
        if isinstance(x, jax.ShapeDtypeStruct):
            return NamedSharding(mesh, P())
        return None
    
    out_shardings = jax.tree.map(get_replicated_sharding, abstract_model)
    
    @partial(jax.jit, out_shardings=out_shardings)
    def init_model(key):
        return BertModel(config, key=key)
    
    # Trigger compilation by calling .lower()
    print("  Lowering (traces without compiling)...")
    start = time.perf_counter()
    lowered = init_model.lower(key)
    end = time.perf_counter()
    print(f"    Time: {end - start:.2f}s")
    
    # Don't actually compile large models
    if hidden < 512:
        print("  Compiling...")
        start = time.perf_counter()
        compiled = lowered.compile()
        end = time.perf_counter()
        print(f"    Time: {end - start:.2f}s")
    else:
        print("  Skipping compilation (too large)")

test_size(256, 2, "Tiny (256h x 2L)")
test_size(512, 4, "Small (512h x 4L)")
test_size(1024, 8, "Medium (1024h x 8L)")
test_size(1536, 12, "Large (1536h x 12L) - will skip compile")
