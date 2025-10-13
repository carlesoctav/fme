import jax.numpy as jnp

# JAX implementation
def sliding_window_mask_overlay(window_size: int):
    def mask(b, h, q, kv):
        return jnp.where(q - kv >= 0, q - kv <= window_size, kv - q <= window_size)
    return mask

# Test with window_size=1, seq_len=5
window_size = 1
seq_len = 5

mask_fn = sliding_window_mask_overlay(window_size)
q_indices = jnp.arange(seq_len)[:, None]
kv_indices = jnp.arange(seq_len)[None, :]

# Create mask for batch=0, head=0
mask = mask_fn(0, 0, q_indices, kv_indices)

print("JAX sliding window mask (window_size=1):")
print(mask)
print()

# HF implementation (distance <= local_attention // 2)
# If local_attention=2, then local_attention//2=1
hf_mask = jnp.abs(q_indices - kv_indices) <= 1
print("HF-style mask (distance <= 1):")
print(hf_mask)
print()

print("Are they the same?", jnp.array_equal(mask, hf_mask))
