#!/usr/bin/env python3
"""
Benchmark make_module_opt with 100GB model using all 4 TPUs with FSDP.
"""

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import optax
from jax.sharding import Mesh
from transformers import BertConfig

from src.models.bert import BertForMaskedLM
from src._training import make_module_opt
from src.distributed import column_parallel, row_parallel, fully_shard


def tp_only_plan(mesh: Mesh):
    """TP-only plan without FSDP."""
    plan = {
        # TP for attention and MLP
        "*.intermediate.dense": lambda m: column_parallel(m, "tp", mesh),
        "*.output.dense": lambda m: row_parallel(m, "tp", mesh),
        "*.attention.self.query": lambda m: column_parallel(m, "tp", mesh),
        "*.attention.self.key": lambda m: column_parallel(m, "tp", mesh),
        "*.attention.self.value": lambda m: column_parallel(m, "tp", mesh),
        "*.attention.output.dense": lambda m: row_parallel(m, "tp", mesh),
    }
    return plan


def benchmark_large_model():
    """Benchmark make_module_opt with large model using TP."""
    print("\n" + "=" * 80)
    print("BENCHMARK: make_module_opt with Large Model (TP only)")
    print("=" * 80)
    
    key = jr.PRNGKey(42)
    
    # Use all 4 TPU devices with 1D mesh for TP only
    devices = jax.devices()
    if len(devices) < 4:
        print(f"ERROR: Need 4 TPUs, only have {len(devices)}")
        return
    
    mesh = Mesh(devices[:4], ("tp",))
    print(f"\nUsing 1D mesh: {mesh}")
    print(f"Devices: {devices[:4]}")
    
    # BERT-36L-2048H ≈ 1.8B params ≈ 6.7GB fp32
    config = BertConfig(
        vocab_size=30522,
        hidden_size=2048,
        num_hidden_layers=36,
        num_attention_heads=16,
        intermediate_size=8192,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        _attn_implementation="sdpa",
    )
    
    print(f"\nModel config: BERT-{config.num_hidden_layers}L-{config.hidden_size}H")
    
    # Calculate model size
    params_per_layer = (
        4 * config.hidden_size * config.hidden_size +
        config.hidden_size * config.intermediate_size +
        config.intermediate_size * config.hidden_size +
        2 * 2 * config.hidden_size
    )
    embedding_params = (
        config.vocab_size * config.hidden_size +
        config.max_position_embeddings * config.hidden_size +
        config.type_vocab_size * config.hidden_size
    )
    total_params = params_per_layer * config.num_hidden_layers + embedding_params
    size_gb = total_params * 4 / (1024**3)
    
    print(f"Estimated params: {total_params / 1e9:.1f}B")
    print(f"Estimated size: {size_gb:.1f}GB (fp32)")
    print(f"Per-device after sharding: ~{size_gb / 4:.1f}GB")
    
    # Create model on CPU
    print("\nCreating model on CPU...")
    cpu_device = jax.devices("cpu")[0]
    
    with jax.default_device(cpu_device):
        start_model_create = time.monotonic()
        model = BertForMaskedLM(config, key=key)
        model_create_time = time.monotonic() - start_model_create
        print(f"Model creation time: {model_create_time:.2f}s")
    
    # Count transformations
    from src._filter import iter_module, _path_to_str
    import fnmatch
    
    tp_plan = tp_only_plan(mesh)
    
    matches = []
    for path, sub_module in iter_module(model):
        path_str = _path_to_str(path)
        for pattern in tp_plan.keys():
            if fnmatch.fnmatchcase(path_str, pattern):
                matches.append(path_str)
                break
    
    print(f"\nFound {len(matches)} submodules matching patterns")
    print(f"This means {len(matches)} transformations will be batched into 1 tree_at call")
    
    grad_tx = optax.adamw(learning_rate=1e-4)
    
    # FIRST RUN
    print("\n" + "=" * 80)
    print("FIRST RUN (JIT COMPILATION + EXECUTION)")
    print("=" * 80)
    
    with mesh:
        start = time.monotonic()
        sharded_model, opt = make_module_opt(
            model,
            grad_tx,
            mesh=mesh,
            parallelism_plans=tp_plan,
            key=key,
        )
        # block_until_ready
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            sharded_model,
        )
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            opt,
        )
        first_run_time = time.monotonic() - start
    
    print(f"\n{'=' * 80}")
    print(f"FIRST RUN TIME: {first_run_time:.4f}s ({first_run_time*1000:.2f}ms)")
    print(f"{'=' * 80}")
    
    print(f"\nModel size: {size_gb:.1f}GB")
    print(f"Transformations: {len(matches)}")
    print(f"Time: {first_run_time:.4f}s")


def main():
    benchmark_large_model()


if __name__ == "__main__":
    main()
