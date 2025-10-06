import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from src._masking_utils import make_causal_mask


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
    """Benchmark a function with optional JIT compilation"""
    if use_jit:
        fn_jit = jax.jit(fn)
    else:
        fn_jit = fn

    for _ in range(warmup):
        result = fn_jit(*args)
        result.block_until_ready()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn_jit(*args)
        result.block_until_ready()
        times.append(time.perf_counter() - start)

    return np.mean(times), np.std(times), result


def run_benchmark(use_jit=True, warmup=3, iterations=10):
    configs = [
        (2, 128),
        (4, 512),
        (8, 1024),
        (16, 2048),
    ]

    jit_status = "WITH JIT" if use_jit else "WITHOUT JIT"
    print("=" * 80)
    print(f"CAUSAL MASK BENCHMARK ({jit_status}) - without padding")
    print("=" * 80)

    for B, T in configs:
        print(f"\nBatch={B}, SeqLen={T}")
        print("-" * 80)

        dummy_input = jnp.ones((B, T, 768))

        mean_time, std_time, result1 = benchmark_fn(
            lambda x: make_causal_mask("eager", x), dummy_input, warmup=warmup, iterations=iterations, use_jit=use_jit
        )
        print(f"make_causal_mask:     {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")

        mean_time, std_time, result2 = benchmark_fn(
            naive_causal_mask_tril, dummy_input, warmup=warmup, iterations=iterations, use_jit=use_jit
        )
        print(f"naive_tril:           {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")

        mean_time, std_time, result3 = benchmark_fn(
            naive_causal_mask_comparison, dummy_input, warmup=warmup, iterations=iterations, use_jit=use_jit
        )
        print(f"naive_comparison:     {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")

        assert jnp.array_equal(
            result1, result2
        ), "Results don't match: make_causal_mask vs naive_tril"
        assert jnp.array_equal(
            result1, result3
        ), "Results don't match: make_causal_mask vs naive_comparison"
        print("✓ All implementations produce same result")

    print("\n" + "=" * 80)
    print(f"CAUSAL MASK BENCHMARK ({jit_status}) - with padding")
    print("=" * 80)

    for B, T in configs:
        print(f"\nBatch={B}, SeqLen={T}")
        print("-" * 80)

        padding_mask = jnp.ones((B, T), dtype=jnp.bool_)
        padding_mask = padding_mask.at[B // 2 :, T // 2 :].set(False)

        dummy_input = jnp.ones((B, T, 768))

        mean_time, std_time, result1 = benchmark_fn(
            lambda x, p: make_causal_mask("eager", x, attention_mask=p),
            dummy_input,
            padding_mask,
            warmup=warmup,
            iterations=iterations,
            use_jit=use_jit,
        )
        print(f"make_causal_mask:     {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")

        mean_time, std_time, result2 = benchmark_fn(
            naive_causal_mask_with_padding_tril, dummy_input, padding_mask, warmup=warmup, iterations=iterations, use_jit=use_jit
        )
        print(f"naive_tril+padding:   {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")

        mean_time, std_time, result3 = benchmark_fn(
            naive_causal_mask_with_padding_comparison, dummy_input, padding_mask, warmup=warmup, iterations=iterations, use_jit=use_jit
        )
        print(f"naive_comp+padding:   {mean_time*1000:.3f} ± {std_time*1000:.3f} ms")

        assert jnp.array_equal(
            result1, result2
        ), "Results don't match: make_causal_mask vs naive_tril"
        assert jnp.array_equal(
            result1, result3
        ), "Results don't match: make_causal_mask vs naive_comparison"
        print("✓ All implementations produce same result")


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention mask implementations")
    parser.add_argument(
        "--jit",
        action="store_true",
        default=False,
        help="Enable JIT compilation (default: False)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark iterations (default: 10)",
    )

    args = parser.parse_args()

    run_benchmark(use_jit=args.jit, warmup=args.warmup, iterations=args.iterations)


if __name__ == "__main__":
    main()
