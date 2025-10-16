# Flax BERT Benchmark Status

## Summary
Unable to run Flax benchmarks due to missing dependency.

## Issue
- `flaxformer` is not in project dependencies (`pyproject.toml`)
- The benchmark scripts require `flaxformer.architectures.bert.bert.BertEncoder` and `flaxformer.architectures.bert.heads.MLMHead`
- Without Flax/Flaxformer, we cannot compare partition/combine overhead between Equinox and Flax implementations

## Files Created
- `research/bert-mlm/bert-flax-benchmark-dropout.py` - Flax benchmark with dropout (0.1)
- `research/bert-mlm/bert-flax-benchmark-no-dropout.py` - Flax benchmark without dropout (0.0)

Both scripts are ready to run but require:
```toml
[dependencies]
flax = ">=0.8.0"
flaxformer = ">=0.8.0"  # Or appropriate version
```

## Why This Matters
The original investigation goal was to compare:
1. **Equinox `make_train_step`** (with `filter_jit` partition/combine overhead)
2. **Equinox manual partition/combine** (optimized)
3. **Flax implementation** (baseline comparison)

Without Flax benchmarks, we can only compare (1) vs (2), which we've already done successfully.

## Results So Far (Equinox only)

### With Dropout (0.1)
- `make_train_step`: 160.6ms/step (6.23 batches/sec)
- Manual partition: 129.0ms/step (7.75 batches/sec)
- **Overhead: 31.6ms** (1.25x speedup with manual approach)

### Without Dropout (0.0)
- `make_train_step`: 134.5ms/step (7.44 batches/sec)
- Manual partition: 104.5ms/step (9.57 batches/sec)
- **Overhead: 30.0ms** (1.29x speedup with manual approach)

## Recommendation
If Flax comparison is desired:
1. Add `flax` and `flaxformer` to `pyproject.toml`
2. Run `uv sync`
3. Execute the prepared benchmark scripts

Otherwise, the Equinox-only analysis is complete and comprehensive.
