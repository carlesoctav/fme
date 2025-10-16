#!/usr/bin/env python3
"""
Proof of concept: Batch all eqx.tree_at calls into a single operation.

This should reduce overhead from O(N * L) to O(L) where:
- N = number of transformations
- L = number of leaves in model PyTree
"""

import time
import jax
import jax.random as jr
import equinox as eqx
from jax.sharding import Mesh
from transformers import BertConfig
import numpy as np

from src.models.bert import BertForMaskedLM
from src.distributed import column_parallel, row_parallel
from src._filter import iter_module, _path_to_str
import fnmatch


def simple_tp_plan(mesh: Mesh, axis_name: str = "tp"):
    """BERT tensor parallelism plan."""
    plan = {
        "*.intermediate.dense": lambda m: column_parallel(m, axis_name, mesh),
        "*.output.dense": lambda m: row_parallel(m, axis_name, mesh),
        "*.attention.self.query": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.self.key": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.self.value": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.output.dense": lambda m: row_parallel(m, axis_name, mesh),
    }
    return plan


def apply_transforms_current(module, parallelism_plans):
    """Current implementation: N separate tree_at calls."""
    if parallelism_plans is None:
        return module
    
    for path, sub_module in iter_module(module):
        path_str = _path_to_str(path)
        
        for pattern, transform in parallelism_plans.items():
            if fnmatch.fnmatchcase(path_str, pattern):
                replacement = transform(sub_module)
                
                def getter(m, path=path):
                    for attr in path:
                        if isinstance(attr, int):
                            m = m[attr]
                        else:
                            m = getattr(m, attr)
                    return m
                
                module = eqx.tree_at(getter, module, replacement)
                break
    
    return module


def apply_transforms_batched(module, parallelism_plans):
    """Optimized implementation: Batch all tree_at calls."""
    if parallelism_plans is None:
        return module
    
    # Collect all transformations
    getters = []
    replacements = []
    
    for path, sub_module in iter_module(module):
        path_str = _path_to_str(path)
        
        for pattern, transform in parallelism_plans.items():
            if fnmatch.fnmatchcase(path_str, pattern):
                replacement = transform(sub_module)
                
                # Create getter with proper closure
                def make_getter(p):
                    def getter(m):
                        for attr in p:
                            if isinstance(attr, int):
                                m = m[attr]
                            else:
                                m = getattr(m, attr)
                        return m
                    return getter
                
                getters.append(make_getter(path))
                replacements.append(replacement)
                break
    
    # Single batched tree_at call
    if getters:
        module = eqx.tree_at(lambda m: [g(m) for g in getters], module, replacements)
    
    return module


def benchmark_both_approaches():
    """Compare current vs batched approach."""
    print("=" * 80)
    print("Comparing current vs batched tree_at approach")
    print("=" * 80)
    
    key = jr.PRNGKey(42)
    devices = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
    mesh = Mesh(devices, ("tp",))
    tp_plan = simple_tp_plan(mesh, axis_name="tp")
    
    # Create model
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
    
    # Benchmark current approach
    print("\n" + "=" * 80)
    print("CURRENT APPROACH (N separate tree_at calls)")
    print("=" * 80)
    
    times_current = []
    for i in range(5):
        with mesh:
            model = BertForMaskedLM(config, key=jr.PRNGKey(42 + i))
            start = time.perf_counter()
            transformed = apply_transforms_current(model, tp_plan)
            elapsed = time.perf_counter() - start
            times_current.append(elapsed * 1000)
            print(f"  Run {i+1}: {elapsed*1000:.2f}ms")
    
    print(f"\nCurrent approach:")
    print(f"  Mean:   {np.mean(times_current):.2f}ms")
    print(f"  Median: {np.median(times_current):.2f}ms")
    print(f"  Std:    {np.std(times_current):.2f}ms")
    
    # Benchmark batched approach
    print("\n" + "=" * 80)
    print("BATCHED APPROACH (1 tree_at call with all replacements)")
    print("=" * 80)
    
    times_batched = []
    for i in range(5):
        with mesh:
            model = BertForMaskedLM(config, key=jr.PRNGKey(42 + i))
            start = time.perf_counter()
            transformed = apply_transforms_batched(model, tp_plan)
            elapsed = time.perf_counter() - start
            times_batched.append(elapsed * 1000)
            print(f"  Run {i+1}: {elapsed*1000:.2f}ms")
    
    print(f"\nBatched approach:")
    print(f"  Mean:   {np.mean(times_batched):.2f}ms")
    print(f"  Median: {np.median(times_batched):.2f}ms")
    print(f"  Std:    {np.std(times_batched):.2f}ms")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    
    speedup = np.median(times_current) / np.median(times_batched)
    saved = np.median(times_current) - np.median(times_batched)
    
    print(f"\nCurrent:  {np.median(times_current):.2f}ms")
    print(f"Batched:  {np.median(times_batched):.2f}ms")
    print(f"Speedup:  {speedup:.1f}×")
    print(f"Saved:    {saved:.2f}ms ({saved/np.median(times_current)*100:.1f}%)")
    
    # Verify correctness
    print("\n" + "=" * 80)
    print("CORRECTNESS CHECK")
    print("=" * 80)
    
    with mesh:
        model = BertForMaskedLM(config, key=jr.PRNGKey(999))
        current_result = apply_transforms_current(model, tp_plan)
        
        model = BertForMaskedLM(config, key=jr.PRNGKey(999))
        batched_result = apply_transforms_batched(model, tp_plan)
        
        # Compare PyTree structures
        current_leaves = jax.tree.leaves(current_result)
        batched_leaves = jax.tree.leaves(batched_result)
        
        print(f"Current result has {len(current_leaves)} leaves")
        print(f"Batched result has {len(batched_leaves)} leaves")
        
        # Check if arrays are equal
        all_equal = all(
            jax.numpy.array_equal(a, b) 
            for a, b in zip(current_leaves, batched_leaves)
        )
        
        if all_equal:
            print("✅ Results are identical!")
        else:
            print("❌ Results differ!")
            # Find differences
            for i, (a, b) in enumerate(zip(current_leaves, batched_leaves)):
                if not jax.numpy.array_equal(a, b):
                    print(f"  Difference at leaf {i}: shapes {a.shape} vs {b.shape}")


if __name__ == "__main__":
    benchmark_both_approaches()
