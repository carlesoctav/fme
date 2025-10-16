# tree_at Batching Optimization - Final Results

## Executive Summary

Batching `eqx.tree_at` operations in `apply_transforms` provides a **7.6× speedup** for tensor parallelism initialization.

- **Non-batched:** 130ms
- **Batched:** 17ms  
- **Speedup:** 7.6×
- **Time saved:** 113ms (87% reduction)

## Problem Identified

In the original implementation (`src/_filter.py`), each transformation pattern match resulted in a separate `eqx.tree_at` call:

```python
for _, getter, replacement in sorted(matches, key=lambda item: len(item[0])):
    module = eqx.tree_at(getter, module, replacement)
```

For a BERT-4L model with 6 TP patterns across 4 layers:
- **24 transformations** (6 patterns × 4 layers)
- **24 separate `tree_at` calls**
- **Each `tree_at` traverses entire PyTree** (~74 leaves)
- **Total cost:** ~130ms

## Solution Implemented

Batch all `tree_at` operations into a single call:

```python
if matches:
    sorted_matches = sorted(matches, key=lambda item: len(item[0]))
    getters = [getter for _, getter, _ in sorted_matches]
    replacements = [replacement for _, _, replacement in sorted_matches]
    module = eqx.tree_at(
        lambda m: [getter(m) for getter in getters], module, replacements
    )
```

This reduces overhead from **O(N × L)** to **O(L)** where:
- N = number of matches (24 for BERT-4L)
- L = number of leaves in PyTree (~74 for BERT-4L)

## Benchmark Results

### Test Configuration
- **Model:** BERT-4L-768H (4 layers, 768 hidden size)
- **Hardware:** 2 JAX devices
- **Mesh:** `Mesh('tp': 2)`
- **TP patterns:** 6 (query, key, value, intermediate, 2× output)
- **Expected transformations:** 24

### Performance Measurements

Test script: `test_direct_benchmark.py` (reuses same model instance across runs)

#### Non-batched Approach
```
Results (20 runs):
  Mean:   137.96ms
  Median: 130.29ms
  Std:    22.03ms
  Min:    129.34ms
  Max:    207.41ms
```

#### Batched Approach
```
Results (20 runs):
  Mean:   17.25ms
  Median: 17.17ms
  Std:    0.22ms
  Min:    17.01ms
  Max:    17.85ms
```

### Analysis

**Speedup:** 130.29ms / 17.17ms = **7.6×**

**Variance reduction:** Batched approach shows much lower standard deviation (0.22ms vs 22.03ms), indicating more consistent performance.

**Per-transformation cost:**
- Non-batched: 130ms / 24 = **5.4ms per transformation**
- Batched: 17ms / 24 = **0.7ms per transformation**

## Impact

For models with more layers, the speedup scales linearly:

| Model | Layers | Transformations | Non-batched | Batched | Speedup |
|-------|--------|----------------|-------------|---------|---------|
| BERT-4L | 4 | 24 | 130ms | 17ms | 7.6× |
| BERT-12L | 12 | 72 | ~390ms* | ~51ms* | 7.6× |
| BERT-24L | 24 | 144 | ~780ms* | ~102ms* | 7.6× |

*Extrapolated based on linear scaling

## Why the Comprehensive Benchmark Failed

The `research/comprehensive_analysis.py` script showed incorrect results (1.00× speedup) because:

1. **Modification functions didn't work:** The `modify_to_unbatched()` and `modify_to_batched()` functions performed string replacements, but the expected string patterns didn't match the actual file content (missing/different comments).

2. **Silent failure:** When the string replacement failed, the script continued without error, so both "non-batched" and "batched" benchmarks actually ran with the same code (whatever was currently in the file).

3. **Misleading results:** Both runs showed ~17ms because both were running the batched version.

## Files Modified

- **`src/_filter.py:115-118`** - Changed from loop-based to batched `tree_at` (committed)

## Verification

Direct benchmark script (`test_direct_benchmark.py`) with manual code switching confirms:
- ✅ Non-batched: 130ms
- ✅ Batched: 17ms
- ✅ Speedup: 7.6×

## Recommendation

**Keep batched implementation.** The optimization provides substantial performance improvement with no downsides:
- 7.6× faster TP initialization
- Lower variance (more predictable)
- Same functionality
- No added complexity
