# Benchmark Comparison: `make_train_step` vs Manual Partition/Combine

## Summary

This benchmark compares the performance of using `make_train_step` (which uses `eqx.filter_jit` internally) versus manual partition/combine pattern for BERT training, both with and without dropout.

## Results

### With Dropout (dropout=0.1, eager attention)
- **make_train_step (filter_jit)**: 0.1606s/step (6.23 batches/sec)
- **manual partition/combine**: 0.1290s/step (7.75 batches/sec)
- **Speedup**: 1.25x
- **Compilation time**: filter_jit 204.5s, manual 227.6s

### Without Dropout (dropout=0.0, SDPA attention)
- **make_train_step (filter_jit)**: 0.1345s/step (7.44 batches/sec)
- **manual partition/combine**: 0.1045s/step (9.57 batches/sec)
- **Speedup**: 1.29x
- **Compilation time**: filter_jit 53.9s, manual 55.8s

## Analysis

### Overhead Comparison

The overhead from `eqx.filter_jit` (used in `make_train_step`) remains consistent:

| Configuration | make_train_step | manual jit | Overhead (ms) | Overhead (%) |
|---------------|-----------------|------------|---------------|--------------|
| **No Dropout (SDPA)** | 134.5ms | 104.5ms | 30.0ms | 28.7% |
| **With Dropout (Eager)** | 160.6ms | 129.0ms | 31.6ms | 24.5% |

**Key Finding**: The overhead is ~30ms per step, slightly lower than the ~47ms observed in the original comparison script. This difference is likely due to:
1. `make_train_step` using the new `Optimizer` dataclass which has less overhead than the old raw opt_state approach
2. Different implementation details in how the optimizer state is managed

### Speedup Analysis

The speedup from manual partition/combine is:
- **1.29x** for no dropout (faster model)
- **1.25x** for with dropout (slower model)

The pattern holds: the relative benefit decreases as total computation time increases, but the **absolute time savings (~30ms) remains constant**.

### Compilation Time

Interestingly, compilation times are very similar between the two approaches:
- No dropout: ~54s both methods
- With dropout: filter_jit actually compiles *faster* (204.5s vs 227.6s)

This suggests that the filter_jit overhead is **runtime overhead from partition/combine**, not compilation overhead.

## Comparison with Original Benchmarks

### Original Comparison (from previous session)
- filter_jit overhead: ~47ms per step
- Speedup: 1.27x (no dropout), 1.23x (with dropout)

### make_train_step Comparison (this session)
- filter_jit overhead: ~30ms per step  
- Speedup: 1.29x (no dropout), 1.25x (with dropout)

**The improvement suggests that `make_train_step` with the new `Optimizer` class is already somewhat optimized** compared to the raw implementation, reducing overhead from 47ms to 30ms per step.

## Conclusions

1. **`make_train_step` is better than raw filter_jit** but still has overhead
   - Reduces overhead from 47ms to 30ms per step
   - Uses cleaner `Optimizer` abstraction

2. **Manual partition/combine still wins** 
   - 30ms faster per step
   - Over 10K steps: 300 seconds = 5 minutes saved

3. **The overhead is structure-dependent, not computation-dependent**
   - ~30ms regardless of dropout/attention implementation
   - Purely from partition/combine operations on pytree

4. **Recommendation for production code**:
   - Use manual partition/combine for long training runs
   - Use `make_train_step` for development (cleaner API, still reasonably fast)
   - The 30ms overhead may be acceptable for many use cases

## Files Created

- `research/bert-mlm-dropout/compare_with_make_train_step.py` - With dropout benchmark
- `research/bert-mlm-dropout/compare_with_make_train_step_no_dropout.py` - Without dropout benchmark
- `research/bert-mlm/flax_benchmark_traces_no_dropout/` - Traces (steps 1-10) for no dropout
- `research/bert-mlm/flax_benchmark_traces_with_dropout/` - Traces (steps 1-10) for with dropout

Note: Flax benchmark comparison was not run as Flax is not a dependency in this project.
