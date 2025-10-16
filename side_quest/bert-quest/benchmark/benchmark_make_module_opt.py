#!/usr/bin/env python3
"""
Benchmark make_module_opt with proper timing (block_until_ready) and JAX profiler trace.
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
from src.distributed import column_parallel, row_parallel


def simple_tp_plan(mesh: Mesh, axis_name: str = "tp"):
    """Simple tensor parallelism plan for BERT."""
    plan = {
        # MLP
        "*.intermediate.dense": lambda m: column_parallel(m, axis_name, mesh),
        "*.output.dense": lambda m: row_parallel(m, axis_name, mesh),
        
        # Attention Q/K/V
        "*.attention.self.query": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.self.key": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.self.value": lambda m: column_parallel(m, axis_name, mesh),
        
        # Attention output
        "*.attention.output.dense": lambda m: row_parallel(m, axis_name, mesh),
    }
    return plan


def benchmark_with_trace():
    """Benchmark make_module_opt with JAX profiler trace and proper timing."""
    print("\n" + "=" * 80)
    print("BENCHMARK: make_module_opt WITH Tensor Parallelism + Trace")
    print("=" * 80)
    
    key = jr.PRNGKey(42)
    
    # 2 devices for TP
    devices = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
    mesh = Mesh(devices, ("tp",))
    print(f"\nUsing mesh: {mesh}")
    print(f"Devices: {devices}")
    
    # BERT-4L config
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=4,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        _attn_implementation="sdpa",
    )
    
    print(f"\nModel config: BERT-{config.num_hidden_layers}L-{config.hidden_size}H")
    
    model = BertForMaskedLM(config, key=key)
    grad_tx = optax.adamw(learning_rate=1e-4)
    tp_plan = simple_tp_plan(mesh, axis_name="tp")
    
    # Count matches
    from src._filter import iter_module, _path_to_str
    import fnmatch
    
    matches = []
    for path, sub_module in iter_module(model):
        path_str = _path_to_str(path)
        for pattern in tp_plan.keys():
            if fnmatch.fnmatchcase(path_str, pattern):
                matches.append(path_str)
                break
    
    print(f"\nFound {len(matches)} submodules matching TP patterns")
    print(f"This means {len(matches)} transformations will be batched into 1 tree_at call")
    
    # Warmup (JIT compilation)
    print("\n" + "-" * 80)
    print("WARMUP RUN (JIT compilation)")
    print("-" * 80)
    
    with mesh:
        start = time.monotonic()
        sharded_model, opt = make_module_opt(
            model,
            grad_tx,
            mesh=mesh,
            parallelism_plans=tp_plan,
            key=key,
        )
        # IMPORTANT: block_until_ready to ensure computation is done
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            sharded_model,
        )
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            opt,
        )
        warmup_time = time.monotonic() - start
    
    print(f"Warmup time (with block_until_ready): {warmup_time:.4f}s")
    
    # Actual timing runs
    print("\n" + "-" * 80)
    print("TIMED RUNS (5 iterations)")
    print("-" * 80)
    
    times = []
    for i in range(5):
        key_i = jr.PRNGKey(42 + i)
        model_i = BertForMaskedLM(config, key=key_i)
        
        with mesh:
            start = time.monotonic()
            sharded_model, opt = make_module_opt(
                model_i,
                grad_tx,
                mesh=mesh,
                parallelism_plans=tp_plan,
                key=key_i,
            )
            # IMPORTANT: block_until_ready
            jtu.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                sharded_model,
            )
            jtu.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                opt,
            )
            elapsed = time.monotonic() - start
        
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms")
    
    import numpy as np
    times = np.array(times)
    print(f"\nResults:")
    print(f"  Mean:   {times.mean()*1000:.2f}ms")
    print(f"  Median: {np.median(times)*1000:.2f}ms")
    print(f"  Std:    {times.std()*1000:.2f}ms")
    
    # Generate JAX profiler trace
    print("\n" + "-" * 80)
    print("GENERATING JAX PROFILER TRACE")
    print("-" * 80)
    print("Trace will be saved to: ./trace_make_module_opt_batched/")
    
    key_trace = jr.PRNGKey(999)
    model_trace = BertForMaskedLM(config, key=key_trace)
    
    with mesh:
        jax.profiler.start_trace("./trace_make_module_opt_batched")
        
        sharded_model, opt = make_module_opt(
            model_trace,
            grad_tx,
            mesh=mesh,
            parallelism_plans=tp_plan,
            key=key_trace,
        )
        
        # Ensure computation completes before stopping trace
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            sharded_model,
        )
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            opt,
        )
        
        jax.profiler.stop_trace()
    
    print("\nTrace saved! Analyze with:")
    print("  tensorboard --logdir=./trace_make_module_opt_batched")
    
    return times


def main():
    print("=" * 80)
    print("Profiling make_module_opt with batched tree_at optimization")
    print("=" * 80)
    
    times = benchmark_with_trace()
    
    import numpy as np
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nmake_module_opt timing (with TP, batched tree_at):")
    print(f"  Mean:   {np.mean(times)*1000:.2f}ms")
    print(f"  Median: {np.median(times)*1000:.2f}ms")
    print(f"  Std:    {np.std(times)*1000:.2f}ms")
    print(f"\nJAX profiler trace: ./trace_make_module_opt_batched/")


if __name__ == "__main__":
    main()
