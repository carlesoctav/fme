# JAX/XLA Fusion Analysis for BERT Training

## Summary

Analysis of XLA HLO dump from BERT forward pass with dropout (prob=0.1) and eager attention.

**Result: Both dropout and attention_mask operations ARE properly fused by XLA.**

## Dropout Fusion

XLA fuses the entire dropout pipeline into a single kernel (`fused_computation.58`):

1. **Random number generation** (Philox counter-based RNG)
   - ~100 ops for stateless RNG using threefry algorithm
   - All arithmetic for random state generation

2. **Random uniform sampling**
   - Bitcast convert to float
   - Subtract to get [0, 1) range

3. **Dropout mask generation**
   - Compare with threshold (0.9 for prob=0.1)
   - Creates boolean mask

4. **LayerNorm computation** (fused with dropout!)
   - Subtract mean: `(x - mean)`
   - Multiply by rsqrt(variance): `* rsqrt(var)`
   - Scale: `* gamma`
   - Shift: `+ beta`

5. **Dropout application**
   - Multiply by scale factor (1/0.9 = 1.111...)
   - Select based on mask

All of this happens in **one fused kernel** called `fused_computation.58`:
- Input: `f32[2,512,768]` (activations) + RNG state + layernorm params
- Output: `f32[2,512,768]` (dropout + layernorm applied)
- **~200 operations fused together**

The fusion is invoked as:
```
%multiply_select_fusion.1 = f32[2,512,768]{2,1,0:T(8,128)S(3)} fusion(
  %bitcast.149, %shift-right-logical_add_fusion, %fusion.195, %fusion.107, 
  %fusion.270, %copy-done.6, %copy-done.7, %add_rsqrt_fusion.3, 
  %reshape.219, %xor.405, %bitcast.32, %bitcast.30
), kind=kLoop, calls=%fused_computation.58
```

## Attention Mask Fusion

The attention mask application is fused with QK^T computation in `fused_computation.8.clone.clone`:

```hlo
%fused_computation.8.clone.clone (
  param_0.2262: bf16[2,512,12,64],  // query
  param_1.3382: bf16[2,512,12,64],  // key
  param_2.3659: pred[2,512]         // attention mask (bool)
) -> f32[2,512,12,512] {
  // Compute QK^T
  %convolution-base-dilated.7 = convolution(query, key), 
    metadata={op_name="jit(forward)/b t n h, b s n h -> b t n s/dot_general"}
  
  // Broadcast mask to full attention shape [B, H, T, T]
  %broadcast.17239 = broadcast(%param_2.3659), dimensions={0,2}
  
  // Broadcast -inf constant
  %constant.1827 = f32[] constant(-3.40282347e+38)
  %broadcast.17238 = broadcast(%constant.1827), dimensions={}
  
  // Apply mask: where(mask, scores, -inf)
  %select.142 = select(%broadcast.17239, %convolution-base-dilated.7, %broadcast.17238)
  ROOT %bitcast.105 = bitcast(%select.142)
}
```

This fuses:
1. **QK^T computation**: Query @ Key^T matrix multiplication
2. **Mask broadcasting**: Expand [B, T] mask to [B, H, T, T]
3. **Mask application**: `jnp.where(mask, scores, -inf)` (additive masking)

All 3 operations run in a single kernel.

## Attention Dropout Fusion

The attention dropout (applied to attention weights, not scores) is fused in `fused_computation.12`:

```hlo
%fused_computation.12 (
  param_0.58: f32[2,512,12,512],  // attention weights after softmax
  param_1.72: f32[2,12,512],       // dropout mask (broadcasted)
  param_2.37: f32[2,512,12]        // dropout scale (1/keep_prob)
) -> f32[2,512,12,512] {
  // Scale by 1/keep_prob
  %broadcast.2169 = broadcast(%param_2.37), dimensions={0,1,2}
  %divide.11 = divide(%param_0.58, %broadcast.2169)
  
  // Apply dropout mask
  %broadcast.2165 = broadcast(%param_1.72), dimensions={0,2,3}
  ROOT %multiply.103 = multiply(%divide.11, %broadcast.2165)
}
```

This is the dropout applied AFTER softmax on the attention weights, using inverted dropout (multiply by mask and scale by 1/keep_prob).

## Performance Impact

### Why This Matters

**Without fusion**, these would be separate kernel launches:
- Dropout: 5 kernels (RNG, uniform, compare, multiply, select)
- Attention mask: 3 kernels (scale, broadcast, multiply)
- **Total: 8 kernel launches per operation**

**With fusion**:
- Dropout: 1 kernel
- Attention mask: 1 kernel
- **Total: 2 kernel launches per operation**

### Kernel Launch Overhead

Each kernel launch has overhead:
- Host-device synchronization
- Kernel dispatch
- Memory bandwidth (reading/writing intermediate results)

Fusing eliminates:
- Intermediate memory traffic
- Multiple kernel launches
- Allows better register reuse

### Measured Impact

From our benchmarks:
- BERT with dropout + eager attention: **181ms/step** (manual jit)
- BERT without dropout + SDPA: **109ms/step** (manual jit)
- **Difference: 72ms** (dropout overhead)

This overhead is primarily from:
1. **Actual dropout computation** (RNG, mask generation, application)
2. **Eager attention** (SDPA can't be used with dropout in JAX)
3. NOT from lack of fusion (fusion is working!)

## Verification Method

```python
import os
os.environ['XLA_FLAGS'] = '--xla_dump_to=/tmp/xla_dump --xla_dump_hlo_as_text'

# Run model
output = jax.jit(model)(inputs)

# Check fusion
grep -A 200 "fused_computation" /tmp/xla_dump/*.after_optimizations.txt
```

Look for:
- `fused_computation.N` - HLO fusion kernel definitions
- Operations from different source files fused together
- Single `ROOT` operation outputting final result

## Conclusion

**XLA is doing its job correctly:**
- ✅ Dropout RNG, mask generation, and application are fused
- ✅ Dropout is fused with LayerNorm
- ✅ Attention mask scaling and application are fused

The 72ms overhead from dropout is **not** due to lack of fusion, but from:
1. The actual computational cost of RNG and masking
2. Eager attention being slower than SDPA (flash attention)
