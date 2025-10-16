# eqx.tree_at Overhead Analysis

## Summary

`eqx.tree_at` has significant overhead during tensor parallelism initialization due to repeated PyTree traversals. Each call must traverse the entire model PyTree to find and replace a single node.

## Measurements

### Direct Timing (BERT 4L-768H, 74 leaves)

**Per transformation:**
- Transform logic (column_parallel/row_parallel): **0.37ms** (6.8%)
- `eqx.tree_at` call: **5.04ms** (93.2%)
- **Total: 5.41ms**

**Total for 24 transformations:**
- Transform logic: 8.8ms
- `eqx.tree_at`: 121ms
- **Total: 129.8ms**

### Scaling Analysis

`eqx.tree_at` scales **perfectly linearly** with model size (R² = 0.9987):

| Config | Leaves | tree_at time | µs/leaf |
|--------|--------|--------------|---------|
| 2L-256H | 42 | 2.98ms | 70.90 |
| 4L-512H | 74 | 5.13ms | 69.30 |
| 4L-768H | 74 | 5.23ms | 70.65 |
| 8L-768H | 138 | 9.23ms | 66.90 |
| 12L-768H | 202 | 14.01ms | 69.34 |

**Linear fit:** `time = 0.0682 * leaves + 0.08`
- **~68µs per leaf** (consistent across all sizes)

### Projected Overhead for Large Models

For BERT-base (12L-768H) with standard TP plan (6 patterns × 12 layers = 72 transformations):
- 202 leaves × 68µs = **14ms per tree_at**
- 72 transformations × 14ms = **~1008ms (1 second)**

For larger models (e.g., GPT-3 175B with ~1000 leaves, ~500 TP transformations):
- 1000 leaves × 68µs = **68ms per tree_at**
- 500 transformations × 68ms = **34 seconds**

## Root Cause

From `src/_filter.py:116`:
```python
for path, sub_module in iter_module(module):
    path_str = _path_to_str(path)
    for pattern, transform in parallelism_plans.items():
        if fnmatch.fnmatchcase(path_str, pattern):
            replacement = transform(sub_module)
            # Creates a getter that traverses from root to target
            module = eqx.tree_at(getter, module, replacement)  # <-- BOTTLENECK
            break
```

Each `eqx.tree_at` call:
1. Traverses entire PyTree from root to find target node (~68µs per leaf)
2. Replaces the target node
3. Reconstructs all parent nodes in the path
4. Returns new PyTree

With N transformations on a model with L leaves:
- **Total traversals: N × L**
- **Time: N × L × 68µs**

## Why This Matters

### make_module_opt overhead (BERT-4L):
- Without TP: **833ms**
- With TP: **1094ms** (24 transformations)
- **Overhead: 261ms** (31.3% slower)
  - `apply_transforms`: 357ms (includes tree_at + JAX compilation)
  - Pure `tree_at`: 121ms (34% of TP overhead)

### Training initialization impact:
- One-time cost during model initialization
- Not a per-step overhead (unlike `eqx.combine` in training loop)
- But still painful for large models (30+ seconds for GPT-3 scale)

## Potential Optimizations

### 1. Batch replacements
Instead of N separate `tree_at` calls, collect all (path, replacement) pairs and apply in one pass:
```python
# Current: O(N * L)
for path, replacement in transformations:
    module = eqx.tree_at(getter(path), module, replacement)  # N traversals

# Optimized: O(L)
all_getters = [getter(path) for path, _ in transformations]
all_replacements = [repl for _, repl in transformations]
module = eqx.tree_at(all_getters, module, all_replacements)  # 1 traversal
```

**Expected speedup for BERT-4L:** 121ms → 5ms (24× faster)

### 2. Build TP-aware model from scratch
Instead of post-hoc transformation, construct model with TP sharding during `__init__`:
```python
# Instead of:
model = BertForMaskedLM(config, key=key)
model = apply_transforms(model, tp_plan)  # Expensive

# Do:
model = BertForMaskedLM(config, key=key, mesh=mesh, tp_axis="tp")  # Built-in TP
```

**Pros:**
- Zero `tree_at` overhead
- More explicit about sharding intent

**Cons:**
- Requires model code changes
- Less flexible (TP plan baked into model)

### 3. Sort and group transformations
The current implementation applies transformations in iteration order. We could:
- Group transformations by depth in the tree
- Apply deeper nodes first (reduces tree reconstruction)

**Expected speedup:** 10-20% (minor optimization)

### 4. Cache PyTree structure
If applying the same TP plan multiple times (e.g., multiple training runs):
- Cache the transformed model structure
- Reuse for subsequent initializations

**Expected speedup:** 100% for 2nd+ initialization

## Recommendation

**Priority 1: Batch replacements**
- Biggest impact (24× speedup)
- Relatively simple to implement
- Works with existing code structure
- No user-facing API changes

**Priority 2: Built-in TP (long-term)**
- Clean architecture
- Zero overhead
- Better for future models
- Requires model rewrites

**Priority 3: Not worth it**
- Sorting/grouping: Too complex for minor gains
- Caching: Only helps repeated runs

## Comparison with eqx.combine Overhead

From previous session:

**eqx.combine (training loop):**
- Per-step overhead: **40ms** (on every forward pass)
- Total impact: 40ms × num_steps (e.g., 40 seconds per 1000 steps)
- **Critical bottleneck** for training performance

**eqx.tree_at (initialization):**
- One-time overhead: **121ms** (BERT-4L) to **1000ms** (BERT-12L)
- Only during model setup
- **Minor annoyance** compared to combine

## Files Modified
- `research/profile_make_module_opt.py` - Full profiling benchmark
- `research/measure_tree_at_overhead.py` - Instrumented tree_at timing
- `research/test_tree_at_scaling.py` - Scaling analysis
