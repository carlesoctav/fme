# Comprehensive Performance Analysis: filter_jit Overhead in BERT Training

## Overview

This document consolidates all performance analyses of `eqx.filter_jit` overhead compared to manual partition/combine patterns across different configurations and optimizations.

## Summary Table

| Analysis | Configuration | Method | Step Time | Overhead | Speedup | Key Finding |
|----------|--------------|--------|-----------|----------|---------|-------------|
| **compare.md** (original) | No dropout + SDPA | filter_jit | 133.5ms | 47.2ms | - | Step 3: 156.63ms |
| | | manual_jit | 104.7ms | - | 1.27x | Step 3: 109.43ms |
| | | | | | | Preprocess: 12.5ms (26%), Postprocess: 34.3ms (74%) |
| **compare_dropout.md** | With dropout + eager | filter_jit | 159.7ms | 46.9ms | - | Step 3: 181.22ms |
| | | manual_jit | 130.1ms | - | 1.23x | Step 3: 134.32ms |
| | | | | | | Preprocess: 12.4ms (26%), Postprocess: 34.5ms (74%) |
| **make_train_step_comparison.md** | No dropout + SDPA | make_train_step | 134.5ms | 30.0ms | - | Uses Optimizer dataclass |
| | | manual_jit | 104.5ms | - | 1.29x | Improved from 47ms to 30ms overhead |
| | With dropout + eager | make_train_step | 160.6ms | 31.6ms | - | Optimizer reduces overhead |
| | | manual_jit | 129.0ms | - | 1.25x | Only 1.6ms difference vs no dropout |
| **fusion_analysis.md** | Dropout + eager | HLO analysis | - | - | - | Dropout: 200 ops fused into 1 kernel |
| | | | | | | | Attention mask: QK^T + mask + -inf in 1 kernel |
| | | | | | | | 72ms dropout overhead NOT from fusion issues |

## Key Findings

### 1. Raw filter_jit Overhead: ~47ms

**Components:**
- Preprocess (`hashable_partition`): 12.5ms (26%)
- Postprocess (`combine`): 34.3ms (74%)
- **Total**: 47ms per step

**What happens on every call:**
1. Before JIT: Partition args into `(dynamic_donate, dynamic_nodonate, static)`
2. JIT execution: Actual computation
3. After JIT: Reconstruct output pytree from `(dynamic_out, static_out)`

**Source files:**
- `compare.md` - Original analysis
- `research/bert-mlm/compare_jit_methods.py` - Benchmark code
- `trace_filter_jit/` - Profiler traces (steps 1-10)

### 2. make_train_step with Optimizer: ~30ms Overhead

**Improvement over raw filter_jit:**
- Reduces overhead from 47ms to 30ms (17ms improvement = 36% reduction)
- Uses `Optimizer` dataclass wrapper with static fields (`tx`, `wrt`)
- Static fields don't contribute to partition/combine overhead

**How it works:**
```python
@dataclass
class Optimizer(eqx.Module):
    opt_state: PyTree[Array]           # Dynamic
    wrt: PyTree[_AxisSpec] = eqx.field(static=True)  # Static
    tx: GradientTransformation = eqx.field(static=True)  # Static
    
    def __call__(self, grads, model):
        updates, opt_state = self.tx.update(grads, self.opt_state, ...)
        new_self = eqx.tree_at(lambda x: x.opt_state, self, opt_state)
        return new_model, new_self
```

**Key insight:**
- `eqx.tree_at` is pure Python pytree manipulation during tracing
- Compiled away - doesn't appear in HLO/traces
- XLA only sees: old opt_state arrays → new opt_state arrays
- Static fields appear as constants in compiled code

**Source files:**
- `research/bert-mlm-dropout/make_train_step_comparison.md`
- `src/_training.py:74-93` - Optimizer implementation

### 3. Overhead is Structure-Dependent, Not Computation-Dependent

**Evidence:**

| Configuration | Preprocess | Postprocess | Total Overhead | Step Difference |
|---------------|------------|-------------|----------------|-----------------|
| No dropout (SDPA) | 12.5ms | 34.3ms | 46.8ms | 47.2ms |
| With dropout (eager) | 12.4ms | 34.5ms | 46.9ms | 46.9ms |
| make_train_step (no dropout) | - | - | 30.0ms | 30.0ms |
| make_train_step (dropout) | - | - | 31.6ms | 31.6ms |

**Observations:**
- Raw filter_jit: Dropout adds only 0.1ms to overhead (47.2ms → 46.9ms)
- make_train_step: Dropout adds only 1.6ms to overhead (30.0ms → 31.6ms)
- Overhead depends on pytree structure (model + opt_state complexity)
- NOT affected by computation type (dropout, attention implementation)

**Implications:**
- Larger models (more parameters, deeper trees) = more overhead
- Faster models (SDPA) benefit more from optimization (higher relative speedup)
- The fixed overhead compounds over training runs

### 4. Manual Partition/Combine Eliminates All Overhead

**Pattern:**
```python
# Setup (ONCE at initialization)
params, static_parts = eqx.partition(model, eqx.is_array)

def loss_fn_params(params, batch, key):
    module_inst = eqx.combine(params, static_parts)  # Captured in closure
    return loss_function(module_inst, batch, key)

@jax.jit
def train_step(params, opt_state, batch, key):
    (loss, aux), grads = grad_fn(params, batch, key)
    updates, new_opt_state = optimizer.tx.update(grads, opt_state, params)
    new_params = eqx.apply_updates(params, updates)
    return new_params, new_opt_state, aux
```

**Why it works:**
- Partition happens ONCE at setup (not per-step)
- Static parts captured in closure (no serialization needed)
- `combine()` happens INSIDE JIT boundary (compiled away)
- JIT receives only pure arrays (no pytree overhead)

**Speedup:**
- No dropout: 1.27x-1.29x (depending on implementation)
- With dropout: 1.23x-1.25x (lower relative gain due to more computation)

**Time savings:**
- 30-47ms per step
- Over 10,000 steps: 300-470 seconds = 5-8 minutes saved

### 5. XLA Fusion is Working Correctly

**Dropout fusion** (fused_computation.58):
- ~200 operations fused into single kernel:
  - Random number generation (Philox/threefry RNG)
  - Uniform sampling
  - Mask generation (compare with threshold)
  - LayerNorm computation
  - Dropout application (scale + select)

**Attention mask fusion** (fused_computation.8):
- Single kernel contains:
  - QK^T computation (query @ key^T)
  - Mask broadcasting [B,T] → [B,H,T,T]
  - Mask application `where(mask, scores, -inf)`

**Attention dropout fusion** (fused_computation.12):
- Fuses attention weight dropout after softmax
- Scale by 1/keep_prob + apply mask in one kernel

**Conclusion:**
- The 72ms difference between no-dropout (109ms) and dropout (181ms) is NOT from fusion issues
- Overhead comes from:
  1. Actual computational cost of RNG and masking
  2. Eager attention being slower than SDPA (flash attention)

**Source file:**
- `research/fusion_analysis.md`

## Benchmark Configuration

All benchmarks use:
- Model: BERT (12 layers, 768 hidden, 12 heads, 110M params)
- Batch size: 64
- Sequence length: 512
- Steps: 100 (excluding compilation)
- Hardware: TPU v5e

## Compilation Time Analysis

| Configuration | filter_jit | manual_jit | Difference |
|---------------|-----------|-----------|------------|
| No dropout (SDPA) | 53-54s | 55-62s | ~+5s |
| With dropout (eager) | 205-213s | 227-228s | ~+20s |

**Observations:**
- Manual jit compiles slightly slower (needs to trace closure)
- Dropout + eager attention increases compilation 4x (complexity)
- Runtime overhead is NOT compilation overhead (happens every step)

## Evolution of Understanding

### Session 1: Original Discovery
- Identified 47ms overhead in filter_jit
- Broke down into preprocess (26%) and postprocess (74%)
- Showed manual partition/combine eliminates overhead
- Source: `compare.md`

### Session 2: Dropout Validation
- Confirmed overhead is structure-dependent (not computation-dependent)
- Dropout adds <1ms to overhead (46.9ms vs 47.2ms)
- Same 1.23x-1.27x speedup pattern holds
- Source: `research/bert-mlm-dropout/compare_dropout.md`

### Session 3: make_train_step Optimization
- Discovered Optimizer dataclass reduces overhead to 30ms
- Still 30ms slower than manual approach
- Provides cleaner API with acceptable performance trade-off
- Source: `research/bert-mlm-dropout/make_train_step_comparison.md`

### Session 4: XLA Fusion Analysis
- Verified XLA is fusing operations correctly
- Dropout overhead is real computation, not fusion failure
- Confirmed `eqx.tree_at` compiles away (doesn't appear in traces)
- Source: `research/fusion_analysis.md`

## Recommendations

### For Maximum Performance (Long Training Runs)
Use manual partition/combine pattern:
```python
params, static = eqx.partition(model, eqx.is_array)

@jax.jit
def train_step(params, opt_state, batch, key):
    model = eqx.combine(params, static)  # Inside JIT
    # ... training logic ...
```

**Benefits:**
- 30-47ms faster per step
- 1.23x-1.29x speedup overall
- Compounds over thousands of steps

**Use when:**
- Training runs with >1000 steps
- Performance is critical
- Hardware utilization matters (GPU/TPU cost)

### For Development and Prototyping
Use `make_train_step`:
```python
from src._training import make_train_step, Optimizer

optimizer = Optimizer.create(optax.adam(1e-4), model)
train_step = make_train_step(loss_fn)
model, optimizer, aux = train_step(model, optimizer, batch, key)
```

**Benefits:**
- Clean, simple API
- Only 30ms overhead (better than raw filter_jit)
- Acceptable for most use cases

**Use when:**
- Rapid iteration/debugging
- Acceptable 20-25% performance trade-off
- Code clarity > max performance

### For Inference or Short Runs
Use raw `eqx.filter_jit`:
```python
@eqx.filter_jit
def inference(model, batch):
    return model(batch)
```

**Benefits:**
- Simplest API
- Overhead matters less with fewer iterations
- Standard Equinox pattern

**Use when:**
- Single or few forward passes
- Inference workloads
- Prototyping

## Implementation Reference

### Available in Codebase

1. **Optimizer class**: `src/_training.py:74-93`
   - Wraps opt_state with static tx and wrt
   - Provides clean update interface
   - Used by make_train_step

2. **make_train_step**: `src/_training.py:145-220`
   - High-level training step factory
   - Uses filter_jit internally
   - ~30ms overhead

3. **make_train_step_with_partition**: `src/_training.py:335-430`
   - Manual partition/combine version
   - Zero overhead
   - More verbose API

### Benchmark Scripts

1. **Original comparison**: `research/bert-mlm/compare_jit_methods.py`
   - Raw filter_jit vs manual jit
   - No dropout, SDPA attention

2. **Dropout comparison**: `research/bert-mlm-dropout/compare_jit_methods.py`
   - Same comparison with dropout enabled
   - Eager attention

3. **make_train_step comparison**: 
   - `research/bert-mlm-dropout/compare_with_make_train_step.py` (dropout)
   - `research/bert-mlm-dropout/compare_with_make_train_step_no_dropout.py` (no dropout)

## Future Work

### Potential Optimizations

1. **Hybrid approach**: 
   - Use manual partition for training
   - Use filter_jit for eval/inference
   - Automatic selection based on mode

2. **Lazy partition caching**:
   - Cache partition results in filter_jit
   - Invalidate on structure change
   - Could reduce overhead to ~0ms

3. **Static argument promotion**:
   - Mark more args as static in JIT
   - Reduce pytree complexity
   - Trade-off: less flexible, more compilation

### Open Questions

1. **Does partition overhead scale linearly with model size?**
   - Test with smaller (BERT-tiny) and larger (BERT-large) models
   - Measure overhead vs parameter count

2. **Can we reduce Optimizer overhead further?**
   - Currently 30ms (down from 47ms)
   - Could static field optimization go further?

3. **What's the break-even point?**
   - At what training length does manual partition pay off?
   - Factor in implementation complexity

## Related Files

### Analysis Documents
- `compare.md` - Original filter_jit vs manual comparison
- `research/bert-mlm-dropout/compare_dropout.md` - Dropout validation
- `research/bert-mlm-dropout/make_train_step_comparison.md` - Optimizer analysis
- `research/fusion_analysis.md` - XLA fusion verification
- `SESSION_SUMMARY.md` - Latest session summary

### Benchmark Scripts
- `research/bert-mlm/compare_jit_methods.py`
- `research/bert-mlm-dropout/compare_jit_methods.py`
- `research/bert-mlm-dropout/compare_with_make_train_step.py`
- `research/bert-mlm-dropout/compare_with_make_train_step_no_dropout.py`

### Traces
- `trace_filter_jit/` - Original filter_jit traces
- `trace_manual_jit/` - Original manual_jit traces
- `research/bert-mlm-dropout/trace_filter_jit/` - Dropout filter_jit traces
- `research/bert-mlm-dropout/trace_manual_jit/` - Dropout manual_jit traces

### Source Code
- `src/_training.py` - Optimizer, make_train_step, make_train_step_with_partition
- `src/models/bert/` - BERT model implementation
- `src/data/masked_language_modeling.py` - MLM data pipeline
