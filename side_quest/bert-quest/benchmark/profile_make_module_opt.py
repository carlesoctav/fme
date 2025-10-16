#!/usr/bin/env python3
"""
Profile make_module_opt to understand the overhead of node swapping with eqx.tree_at
when applying tensor parallelism plans.

This script:
1. Creates a simple model
2. Defines a simple TP plan
3. Times make_module_opt with JAX profiling
4. Analyzes what happens during eqx.tree_at node swapping
"""

import time
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jax.sharding import Mesh
from transformers import BertConfig

from src.models.bert import BertForMaskedLM
from src._training import make_module_opt
from src.distributed import column_parallel, row_parallel
from src import nn


def simple_tp_plan(mesh: Mesh, axis_name: str = "tp"):
    """
    Simple tensor parallelism plan for BERT:
    - MLP layer 1 (intermediate): column parallel
    - MLP layer 2 (output): row parallel
    - Attention Q/K/V: column parallel
    - Attention output: row parallel
    """
    plan = {
        # MLP (BERT structure)
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


def benchmark_make_module_opt_no_tp():
    """Benchmark make_module_opt WITHOUT tensor parallelism."""
    print("\n" + "=" * 80)
    print("BENCHMARK: make_module_opt WITHOUT Tensor Parallelism")
    print("=" * 80)
    
    key = jr.PRNGKey(42)
    
    # Single device mesh (no TP)
    devices = jax.devices()[:1]
    mesh = Mesh(devices, ("dp",))
    
    # Small BERT config
    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=4,  # Smaller for faster testing
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        _attn_implementation="sdpa",
    )
    
    model = BertForMaskedLM(config, key=key)
    
    grad_tx = optax.adamw(learning_rate=1e-4)
    
    # Warmup (JIT compilation)
    print("\nWarming up (JIT compilation)...")
    start = time.perf_counter()
    with mesh:
        _, _ = make_module_opt(
            model,
            grad_tx,
            mesh=mesh,
            parallelism_plans=None,  # No TP
            key=key,
        )
    warmup_time = time.perf_counter() - start
    print(f"Warmup time: {warmup_time:.4f}s")
    
    # Actual timing (5 runs)
    print("\nRunning 5 timed iterations...")
    times = []
    for i in range(5):
        # Recreate model each time
        key = jr.PRNGKey(42 + i)
        model = BertForMaskedLM(config, key=key)
        
        start = time.perf_counter()
        with mesh:
            _, _ = make_module_opt(
                model,
                grad_tx,
                mesh=mesh,
                parallelism_plans=None,
                key=key,
            )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    import numpy as np
    times = np.array(times)
    print(f"\nResults (no TP):")
    print(f"  Mean:   {times.mean():.4f}s")
    print(f"  Median: {np.median(times):.4f}s")
    print(f"  Std:    {times.std():.4f}s")
    
    return times


def benchmark_make_module_opt_with_tp():
    """Benchmark make_module_opt WITH tensor parallelism."""
    print("\n" + "=" * 80)
    print("BENCHMARK: make_module_opt WITH Tensor Parallelism")
    print("=" * 80)
    
    key = jr.PRNGKey(42)
    
    # 2 devices for TP
    devices = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
    mesh = Mesh(devices, ("tp",))
    print(f"\nUsing mesh: {mesh}")
    print(f"Devices: {devices}")
    
    # Small BERT config
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
    
    model = BertForMaskedLM(config, key=key)
    
    grad_tx = optax.adamw(learning_rate=1e-4)
    
    # Create TP plan
    tp_plan = simple_tp_plan(mesh, axis_name="tp")
    print(f"\nTP plan patterns: {list(tp_plan.keys())}")
    
    # Warmup
    print("\nWarming up (JIT compilation)...")
    start = time.perf_counter()
    with mesh:
        _, _ = make_module_opt(
            model,
            grad_tx,
            mesh=mesh,
            parallelism_plans=tp_plan,
            key=key,
        )
    warmup_time = time.perf_counter() - start
    print(f"Warmup time: {warmup_time:.4f}s")
    
    # Actual timing
    print("\nRunning 5 timed iterations...")
    times = []
    for i in range(5):
        key = jr.PRNGKey(42 + i)
        model = BertForMaskedLM(config, key=key)
        
        start = time.perf_counter()
        with mesh:
            _, _ = make_module_opt(
                model,
                grad_tx,
                mesh=mesh,
                parallelism_plans=tp_plan,
                key=key,
            )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f}s")
    
    import numpy as np
    times = np.array(times)
    print(f"\nResults (with TP):")
    print(f"  Mean:   {times.mean():.4f}s")
    print(f"  Median: {np.median(times):.4f}s")
    print(f"  Std:    {times.std():.4f}s")
    
    return times


def profile_tree_at_operations():
    """Profile eqx.tree_at operations during apply_transforms."""
    print("\n" + "=" * 80)
    print("PROFILING: eqx.tree_at operations in apply_transforms")
    print("=" * 80)
    
    key = jr.PRNGKey(42)
    
    devices = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
    mesh = Mesh(devices, ("tp",))
    
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
    
    model = BertForMaskedLM(config, key=key)
    grad_tx = optax.adamw(learning_rate=1e-4)
    tp_plan = simple_tp_plan(mesh, axis_name="tp")
    
    print("\nStarting JAX profiler trace...")
    print("Trace will be saved to: ./trace_make_module_opt/")
    
    with mesh:
        jax.profiler.start_trace("./trace_make_module_opt")
        
        # Time just the apply_transforms part
        from src._filter import apply_transforms
        
        print("\nApplying transforms...")
        start = time.perf_counter()
        transformed_model = apply_transforms(model, tp_plan)
        transform_time = time.perf_counter() - start
        print(f"apply_transforms took: {transform_time:.4f}s")
        
        # Continue with the rest of make_module_opt
        print("\nRunning full make_module_opt...")
        start = time.perf_counter()
        sharded_model, opt = make_module_opt(
            model,
            grad_tx,
            mesh=mesh,
            parallelism_plans=tp_plan,
            key=key,
        )
        total_time = time.perf_counter() - start
        print(f"Full make_module_opt took: {total_time:.4f}s")
        
        jax.profiler.stop_trace()
    
    print("\nProfiler trace saved!")
    print(f"  Transform overhead: {transform_time:.4f}s ({transform_time/total_time*100:.1f}% of total)")
    print(f"  Other overhead:     {total_time - transform_time:.4f}s ({(total_time-transform_time)/total_time*100:.1f}% of total)")


def count_tree_at_calls():
    """Count how many eqx.tree_at calls happen during apply_transforms."""
    print("\n" + "=" * 80)
    print("ANALYSIS: Counting eqx.tree_at calls")
    print("=" * 80)
    
    key = jr.PRNGKey(42)
    
    devices = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
    mesh = Mesh(devices, ("tp",))
    
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
    
    model = BertForMaskedLM(config, key=key)
    tp_plan = simple_tp_plan(mesh, axis_name="tp")
    
    # Count matches
    from src._filter import iter_module, _path_to_str
    import fnmatch
    
    matches = []
    for path, sub_module in iter_module(model):
        path_str = _path_to_str(path)
        for pattern, transform in tp_plan.items():
            if fnmatch.fnmatchcase(path_str, pattern):
                matches.append((path_str, pattern, type(sub_module).__name__))
                break
    
    print(f"\nFound {len(matches)} submodules matching TP patterns:")
    for path_str, pattern, module_type in matches:
        print(f"  {path_str:60s} <- {pattern:40s} ({module_type})")
    
    print(f"\nThis means eqx.tree_at will be called {len(matches)} times during apply_transforms!")
    
    # Each tree_at call needs to:
    # 1. Traverse the entire PyTree to find the target node
    # 2. Create a new PyTree with the replaced node
    # 3. Reconstruct all parent nodes along the path
    
    print("\nFor each eqx.tree_at call:")
    print("  1. Traverse entire model PyTree to find target")
    print("  2. Replace target node")
    print("  3. Reconstruct all parent nodes in path")
    print("  4. Return new model PyTree")
    
    # Estimate PyTree size
    import jax.tree_util as jtu
    num_leaves = len(jtu.tree_leaves(model))
    print(f"\nModel PyTree has ~{num_leaves} leaves")
    print(f"With {len(matches)} tree_at calls, total operations:")
    print(f"  ~{len(matches) * num_leaves} node traversals")


def main():
    print("=" * 80)
    print("Profiling make_module_opt with Tensor Parallelism")
    print("=" * 80)
    
    # 1. Count tree_at operations
    count_tree_at_calls()
    
    # 2. Benchmark without TP
    no_tp_times = benchmark_make_module_opt_no_tp()
    
    # 3. Benchmark with TP
    with_tp_times = benchmark_make_module_opt_with_tp()
    
    # 4. Profile with JAX tracer
    profile_tree_at_operations()
    
    # Summary
    import numpy as np
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nWithout TP: {np.median(no_tp_times):.4f}s")
    print(f"With TP:    {np.median(with_tp_times):.4f}s")
    print(f"Overhead:   {np.median(with_tp_times) - np.median(no_tp_times):.4f}s")
    print(f"            ({(np.median(with_tp_times) / np.median(no_tp_times) - 1) * 100:.1f}% slower)")
    
    print("\nJAX profiler trace saved to: ./trace_make_module_opt/")
    print("Analyze with:")
    print("  tensorboard --logdir=./trace_make_module_opt")


if __name__ == "__main__":
    main()
