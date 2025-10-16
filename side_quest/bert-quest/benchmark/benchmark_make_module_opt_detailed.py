#!/usr/bin/env python3
"""
Detailed timing breakdown of make_module_opt using manual instrumentation.
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
from src.distributed import column_parallel, row_parallel


def simple_tp_plan(mesh: Mesh, axis_name: str = "tp"):
    """Simple tensor parallelism plan for BERT."""
    plan = {
        "*.intermediate.dense": lambda m: column_parallel(m, axis_name, mesh),
        "*.output.dense": lambda m: row_parallel(m, axis_name, mesh),
        "*.attention.self.query": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.self.key": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.self.value": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.output.dense": lambda m: row_parallel(m, axis_name, mesh),
    }
    return plan


def manual_timing_breakdown():
    """Manually time each step of make_module_opt."""
    print("\n" + "=" * 80)
    print("MANUAL TIMING BREAKDOWN OF make_module_opt")
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
    
    print(f"\nModel: BERT-{config.num_hidden_layers}L-{config.hidden_size}H")
    print(f"Mesh: {mesh}")
    
    # Create model
    print("\n" + "-" * 80)
    print("Step 1: Create model")
    start = time.monotonic()
    model = BertForMaskedLM(config, key=key)
    model_time = time.monotonic() - start
    print(f"  Time: {model_time*1000:.2f}ms")
    
    grad_tx = optax.adamw(learning_rate=1e-4)
    tp_plan = simple_tp_plan(mesh, axis_name="tp")
    
    # Apply transforms (outside JIT to measure Python overhead)
    print("\n" + "-" * 80)
    print("Step 2: Apply transforms (outside JIT)")
    from src._filter import apply_transforms
    
    start = time.monotonic()
    transformed_model = apply_transforms(model, tp_plan)
    transform_time = time.monotonic() - start
    print(f"  Time: {transform_time*1000:.2f}ms")
    
    # Get partition spec
    print("\n" + "-" * 80)
    print("Step 3: Get partition spec")
    from src.distributed import get_partition_spec
    
    start = time.monotonic()
    pspec_tree = get_partition_spec(transformed_model)
    pspec_time = time.monotonic() - start
    print(f"  Time: {pspec_time*1000:.2f}ms")
    
    # Filter shard
    print("\n" + "-" * 80)
    print("Step 4: eqx.filter_shard")
    
    with mesh:
        start = time.monotonic()
        m_sharded = eqx.filter_shard(transformed_model, pspec_tree)
        # Block until ready
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            m_sharded,
        )
        shard_time = time.monotonic() - start
    print(f"  Time: {shard_time*1000:.2f}ms")
    
    # Create optimizer
    print("\n" + "-" * 80)
    print("Step 5: Optimizer.create")
    from src._training import Optimizer
    
    with mesh:
        start = time.monotonic()
        opt = Optimizer.create(grad_tx, m_sharded, wrt=eqx.is_inexact_array)
        # Block until ready
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            opt,
        )
        opt_time = time.monotonic() - start
    print(f"  Time: {opt_time*1000:.2f}ms")
    
    # Now test with JIT
    print("\n" + "=" * 80)
    print("TESTING WITH JIT (as in actual make_module_opt)")
    print("=" * 80)
    
    def _build(m, rng):
        m = apply_transforms(m, tp_plan)
        pspec_tree = get_partition_spec(m)
        m_sharded = eqx.filter_shard(m, pspec_tree)
        opt = Optimizer.create(grad_tx, m_sharded, wrt=eqx.is_inexact_array)
        return m_sharded, opt
    
    build_jit = eqx.filter_jit(_build)
    
    # Warmup
    print("\nWarmup (JIT compilation)...")
    with mesh:
        model_fresh = BertForMaskedLM(config, key=key)
        start = time.monotonic()
        m_out, opt_out = build_jit(model_fresh, key)
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            m_out,
        )
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            opt_out,
        )
        warmup_time = time.monotonic() - start
    print(f"  Warmup time: {warmup_time*1000:.2f}ms")
    
    # Timed runs
    print("\nTimed runs (5 iterations)...")
    times = []
    for i in range(5):
        key_i = jr.PRNGKey(42 + i)
        model_i = BertForMaskedLM(config, key=key_i)
        
        with mesh:
            start = time.monotonic()
            m_out, opt_out = build_jit(model_i, key_i)
            jtu.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                m_out,
            )
            jtu.tree_map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                opt_out,
            )
            elapsed = time.monotonic() - start
        
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f}ms")
    
    import numpy as np
    times = np.array(times)
    jit_time_avg = times.mean()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nManual step-by-step (outside JIT):")
    print(f"  Apply transforms:   {transform_time*1000:>8.2f}ms")
    print(f"  Get partition spec: {pspec_time*1000:>8.2f}ms")
    print(f"  Filter shard:       {shard_time*1000:>8.2f}ms")
    print(f"  Optimizer create:   {opt_time*1000:>8.2f}ms")
    total_manual = (transform_time + pspec_time + shard_time + opt_time) * 1000
    print(f"  TOTAL:              {total_manual:>8.2f}ms")
    
    print(f"\nWith JIT (actual make_module_opt):")
    print(f"  Average:            {jit_time_avg*1000:>8.2f}ms")
    print(f"  Median:             {np.median(times)*1000:>8.2f}ms")
    
    print(f"\nJIT overhead:")
    jit_overhead = jit_time_avg*1000 - total_manual
    print(f"  {jit_overhead:>8.2f}ms ({jit_overhead/total_manual*100:.1f}% of manual)")
    
    print(f"\nBreakdown of manual time:")
    print(f"  Apply transforms:   {transform_time/total_manual*1000*100:>6.1f}%")
    print(f"  Get partition spec: {pspec_time/total_manual*1000*100:>6.1f}%")
    print(f"  Filter shard:       {shard_time/total_manual*1000*100:>6.1f}%")
    print(f"  Optimizer create:   {opt_time/total_manual*1000*100:>6.1f}%")


def main():
    manual_timing_breakdown()


if __name__ == "__main__":
    main()
