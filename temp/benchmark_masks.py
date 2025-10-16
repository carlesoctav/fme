import jax
import jax.numpy as jnp
import time
import numpy as np

from src.masking_utils import make_causal_mask


def naive_causal_mask_tril(dummy_input):
    """Naive implementation using jnp.tril"""
    B, T, _ = dummy_input.shape
    return jnp.tril(jnp.ones((B, T, T), dtype=jnp.bool_))


def naive_causal_mask_comparison(dummy_input):
    """Naive implementation using comparison"""
    B, T, _ = dummy_input.shape
    q_idx = jnp.arange(T)[:, None]
    kv_idx = jnp.arange(T)[None, :]
    mask = q_idx >= kv_idx
    return jnp.broadcast_to(mask[None, :, :], (B, T, T))


def naive_causal_mask_with_padding_tril(dummy_input, padding_mask):
    """Naive with padding using tril"""
    B, T, _ = dummy_input.shape
    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))
    causal = jnp.broadcast_to(causal[None, :, :], (B, T, T))
    padding = padding_mask[:, None, :]
    return causal & padding


def naive_causal_mask_with_padding_comparison(dummy_input, padding_mask):
    """Naive with padding using comparison"""
    B, T, _ = dummy_input.shape
    q_idx = jnp.arange(T)[:, None]
    kv_idx = jnp.arange(T)[None, :]
    causal = q_idx >= kv_idx
    causal = jnp.broadcast_to(causal[None, :, :], (B, T, T))
    padding = padding_mask[:, None, :]
    return causal & padding


def benchmark_fn(fn, *args, warmup=3, iterations=10, use_jit=True):
    """Benchmark a function with JIT compilation"""
    if use_jit:
        fn_jit = jax.jit(fn)
    else:
        fn_jit = fn
    
    # Warmup
    for _ in range(warmup):
        result = fn_jit(*args)
        result.block_until_ready()
    
    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn_jit(*args)
        result.block_until_ready()
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times), result


# Test configurations
configs = [
    (2, 128),
    (4, 512),
    (8, 1024),
    (16, 2048),
]

print("=" * 80)
print("CAUSAL MASK BENCHMARK (without padding)")
print("=" * 80)

for B, T in configs:
    print(f"\nBatch={B}, SeqLen={T}")
    print("-" * 80)
    
    # Dummy input for make_causal_mask
    dummy_input = jnp.ones((B, T, 768))
    
    # Current implementation (no JIT)
    mean_time, std_time, result1 = benchmark_fn(
        lambda x: make_causal_mask("eager", x),
        dummy_input,
        use_jit=False
    )
    print(f"make_causal_mask:     {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    
    # Naive tril (no JIT)
    mean_time, std_time, result2 = benchmark_fn(
        naive_causal_mask_tril, dummy_input, use_jit=False
    )
    print(f"naive_tril:           {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    
    # Naive comparison (no JIT)
    mean_time, std_time, result3 = benchmark_fn(
        naive_causal_mask_comparison, dummy_input, use_jit=False
    )
    print(f"naive_comparison:     {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    
    # Verify all produce same result
    assert jnp.array_equal(result1, result2), "Results don't match: make_causal_mask vs naive_tril"
    assert jnp.array_equal(result1, result3), "Results don't match: make_causal_mask vs naive_comparison"
    print("✓ All implementations produce same result")


print("\n" + "=" * 80)
print("CAUSAL MASK BENCHMARK (with padding)")
print("=" * 80)

for B, T in configs:
    print(f"\nBatch={B}, SeqLen={T}")
    print("-" * 80)
    
    # Create padding mask (50% of tokens are padding in half the batch)
    padding_mask = jnp.ones((B, T), dtype=jnp.bool_)
    padding_mask = padding_mask.at[B//2:, T//2:].set(False)
    
    # Dummy input for make_causal_mask
    dummy_input = jnp.ones((B, T, 768))
    
    # Current implementation (no JIT)
    mean_time, std_time, result1 = benchmark_fn(
        lambda x, p: make_causal_mask("eager", x, attention_mask=p),
        dummy_input, padding_mask,
        use_jit=False
    )
    print(f"make_causal_mask:     {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    
    # Naive tril with padding (no JIT)
    mean_time, std_time, result2 = benchmark_fn(
        naive_causal_mask_with_padding_tril, dummy_input, padding_mask, use_jit=False
    )
    print(f"naive_tril+padding:   {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    
    # Naive comparison with padding (no JIT)
    mean_time, std_time, result3 = benchmark_fn(
        naive_causal_mask_with_padding_comparison, dummy_input, padding_mask, use_jit=False
    )
    print(f"naive_comp+padding:   {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")
    
    # Verify all produce same result
    assert jnp.array_equal(result1, result2), "Results don't match: make_causal_mask vs naive_tril"
    assert jnp.array_equal(result1, result3), "Results don't match: make_causal_mask vs naive_comparison"
    print("✓ All implementations produce same result")
