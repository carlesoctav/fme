#!/usr/bin/env python3
"""
Directly measure eqx.tree_at overhead by instrumenting apply_transforms.
"""

import time
import jax
import jax.random as jr
import equinox as eqx
from jax.sharding import Mesh
from transformers import BertConfig

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


def apply_transforms_instrumented(module, parallelism_plans):
    """
    Instrumented version of apply_transforms that measures each tree_at call.
    """
    if parallelism_plans is None:
        return module
    
    timings = []
    
    for path, sub_module in iter_module(module):
        path_str = _path_to_str(path)
        
        for pattern, transform in parallelism_plans.items():
            if fnmatch.fnmatchcase(path_str, pattern):
                print(f"  Transforming: {path_str:60s} <- {pattern}")
                
                # Time the transformation
                start = time.perf_counter()
                
                # Get the replacement
                replacement = transform(sub_module)
                
                # Create getter function
                def getter(m, path=path):
                    for attr in path:
                        if isinstance(attr, int):
                            m = m[attr]
                        else:
                            m = getattr(m, attr)
                    return m
                
                # Time the tree_at call specifically
                tree_at_start = time.perf_counter()
                module = eqx.tree_at(getter, module, replacement)
                tree_at_time = time.perf_counter() - tree_at_start
                
                total_time = time.perf_counter() - start
                transform_time = total_time - tree_at_time
                
                timings.append({
                    'path': path_str,
                    'pattern': pattern,
                    'transform_time': transform_time * 1000,  # ms
                    'tree_at_time': tree_at_time * 1000,  # ms
                    'total_time': total_time * 1000,  # ms
                })
                
                break  # Only apply first matching pattern
    
    return module, timings


def main():
    print("=" * 80)
    print("Measuring eqx.tree_at overhead in apply_transforms")
    print("=" * 80)
    
    key = jr.PRNGKey(42)
    
    # Create mesh
    devices = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
    mesh = Mesh(devices, ("tp",))
    print(f"\nMesh: {mesh}")
    
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
    
    model = BertForMaskedLM(config, key=key)
    tp_plan = simple_tp_plan(mesh, axis_name="tp")
    
    print(f"\nModel has {len(jax.tree.leaves(model))} leaves")
    print(f"\nApplying {len(tp_plan)} transformation patterns...")
    
    # Run instrumented apply_transforms
    with mesh:
        start = time.perf_counter()
        transformed_model, timings = apply_transforms_instrumented(model, tp_plan)
        total_time = time.perf_counter() - start
    
    # Analyze timings
    print("\n" + "=" * 80)
    print("PER-TRANSFORMATION BREAKDOWN")
    print("=" * 80)
    print(f"{'Path':<60s} {'Transform':<10s} {'tree_at':<10s} {'Total':<10s}")
    print("-" * 80)
    
    for t in timings:
        print(f"{t['path']:<60s} {t['transform_time']:>8.2f}ms {t['tree_at_time']:>8.2f}ms {t['total_time']:>8.2f}ms")
    
    # Summary statistics
    import numpy as np
    transform_times = [t['transform_time'] for t in timings]
    tree_at_times = [t['tree_at_time'] for t in timings]
    total_times = [t['total_time'] for t in timings]
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Number of transformations: {len(timings)}")
    print(f"\nTransform (get replacement) time:")
    print(f"  Total:  {np.sum(transform_times):.2f}ms")
    print(f"  Mean:   {np.mean(transform_times):.2f}ms")
    print(f"  Median: {np.median(transform_times):.2f}ms")
    print(f"  Min:    {np.min(transform_times):.2f}ms")
    print(f"  Max:    {np.max(transform_times):.2f}ms")
    
    print(f"\neqx.tree_at time:")
    print(f"  Total:  {np.sum(tree_at_times):.2f}ms")
    print(f"  Mean:   {np.mean(tree_at_times):.2f}ms")
    print(f"  Median: {np.median(tree_at_times):.2f}ms")
    print(f"  Min:    {np.min(tree_at_times):.2f}ms")
    print(f"  Max:    {np.max(tree_at_times):.2f}ms")
    
    print(f"\nTotal transformation time:")
    print(f"  Total:  {np.sum(total_times):.2f}ms ({total_time*1000:.2f}ms wall clock)")
    print(f"  Mean:   {np.mean(total_times):.2f}ms")
    
    print(f"\nBreakdown:")
    print(f"  Transform logic: {np.sum(transform_times):.2f}ms ({np.sum(transform_times)/np.sum(total_times)*100:.1f}%)")
    print(f"  tree_at calls:   {np.sum(tree_at_times):.2f}ms ({np.sum(tree_at_times)/np.sum(total_times)*100:.1f}%)")


if __name__ == "__main__":
    main()
