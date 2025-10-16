#!/usr/bin/env python3
"""
Test how eqx.tree_at overhead scales with model size.
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


def measure_single_tree_at(module, pattern, transform):
    """Measure a single tree_at operation."""
    # Find first match
    for path, sub_module in iter_module(module):
        path_str = _path_to_str(path)
        if fnmatch.fnmatchcase(path_str, pattern):
            # Get replacement
            replacement = transform(sub_module)
            
            # Create getter
            def getter(m, path=path):
                for attr in path:
                    if isinstance(attr, int):
                        m = m[attr]
                    else:
                        m = getattr(m, attr)
                return m
            
            # Measure tree_at
            start = time.perf_counter()
            new_module = eqx.tree_at(getter, module, replacement)
            elapsed = time.perf_counter() - start
            
            return elapsed * 1000, path_str
    
    return None, None


def test_scaling():
    """Test how tree_at scales with different model sizes."""
    print("=" * 80)
    print("Testing eqx.tree_at scaling with model size")
    print("=" * 80)
    
    key = jr.PRNGKey(42)
    devices = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
    mesh = Mesh(devices, ("tp",))
    tp_plan = simple_tp_plan(mesh, axis_name="tp")
    
    # Test different model sizes
    configs = [
        ("2 layers, 256 hidden", 2, 256, 4),
        ("4 layers, 512 hidden", 4, 512, 8),
        ("4 layers, 768 hidden", 4, 768, 12),
        ("8 layers, 768 hidden", 8, 768, 12),
        ("12 layers, 768 hidden", 12, 768, 12),
    ]
    
    results = []
    
    for name, num_layers, hidden_size, num_heads in configs:
        config = BertConfig(
            vocab_size=30522,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )
        
        with mesh:
            model = BertForMaskedLM(config, key=key)
        
        num_leaves = len(jax.tree.leaves(model))
        
        # Measure a single tree_at on first query layer
        pattern = "*.attention.self.query"
        transform = tp_plan[pattern]
        
        # Warmup
        measure_single_tree_at(model, pattern, transform)
        
        # Measure 5 times
        times = []
        for _ in range(5):
            elapsed, path = measure_single_tree_at(model, pattern, transform)
            if elapsed:
                times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results.append({
            'name': name,
            'num_layers': num_layers,
            'hidden_size': hidden_size,
            'num_leaves': num_leaves,
            'avg_time': avg_time,
            'std_time': std_time,
        })
        
        print(f"\n{name}:")
        print(f"  Leaves:         {num_leaves}")
        print(f"  tree_at time:   {avg_time:.2f} ± {std_time:.2f}ms")
        print(f"  Time per leaf:  {avg_time/num_leaves*1000:.2f}µs")
    
    # Summary
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    print(f"{'Config':<30s} {'Leaves':<10s} {'tree_at (ms)':<15s} {'µs/leaf':<10s}")
    print("-" * 80)
    
    for r in results:
        us_per_leaf = r['avg_time'] / r['num_leaves'] * 1000
        print(f"{r['name']:<30s} {r['num_leaves']:<10d} {r['avg_time']:>8.2f} ± {r['std_time']:>4.2f} {us_per_leaf:>10.2f}")
    
    # Check if linear scaling
    print("\nConclusion:")
    leaves = np.array([r['num_leaves'] for r in results])
    times = np.array([r['avg_time'] for r in results])
    
    # Linear regression
    from numpy.polynomial import polynomial as P
    coefs = P.polyfit(leaves, times, 1)
    print(f"  Linear fit: time = {coefs[1]:.4f} * leaves + {coefs[0]:.2f}")
    print(f"  Approximately {coefs[1]*1000:.2f}µs per leaf")
    
    # R² to check linearity
    predicted = P.polyval(leaves, coefs)
    ss_res = np.sum((times - predicted) ** 2)
    ss_tot = np.sum((times - np.mean(times)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"  R² = {r_squared:.4f} (1.0 = perfect linear scaling)")


if __name__ == "__main__":
    test_scaling()
