import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import torch

# Simple test: apply both RoPE methods and compare

# Setup
head_dim = 8  # Small for debugging
theta = 10000.0
position = 1

# Create Q vector
q_np = np.random.randn(head_dim).astype(np.float32)
q_torch = torch.from_numpy(q_np)
q_jax = jnp.array(q_np)

print(f"Q: {q_np[:4]}")

# PyTorch RoPE (rotate_half style)
# Create freqs - this should create head_dim/2 frequencies
dim = head_dim
inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
freqs = position * inv_freq
emb = torch.cat([freqs, freqs])  # Duplicate: [f0, f1, f2, f3, f0, f1, f2, f3]
cos_torch = emb.cos()
sin_torch = emb.sin()

print(f"\nPyTorch freqs: {freqs.numpy()}")
print(f"PyTorch emb (duplicated): {emb.numpy()}")
print(f"PyTorch cos: {cos_torch.numpy()}")
print(f"PyTorch sin: {sin_torch.numpy()}")

# rotate_half
q1 = q_torch[:head_dim // 2]
q2 = q_torch[head_dim // 2:]
q_rotated_torch = torch.cat([-q2, q1])

# Apply RoPE
q_rope_torch = q_torch * cos_torch + q_rotated_torch * sin_torch
print(f"\nPyTorch Q after RoPE: {q_rope_torch.numpy()}")

# JAX RoPE (complex number style)
# Create rtheta as complex numbers
dim = head_dim
inv_freq_jax = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
freqs_jax = position * inv_freq_jax
cos_jax = jnp.cos(freqs_jax)
sin_jax = jnp.sin(freqs_jax)
rtheta_jax = cos_jax + 1j * sin_jax  # Complex: [c0+is0, c1+is1, c2+is2, c3+is3]

print(f"\nJAX freqs: {freqs_jax}")
print(f"JAX rtheta (complex): {rtheta_jax}")
print(f"JAX cos: {cos_jax}")
print(f"JAX sin: {sin_jax}")

# Reshape Q into pairs and convert to complex
q_pairs = q_jax.reshape(head_dim // 2, 2)  # [(q0,q1), (q2,q3), (q4,q5), (q6,q7)]
q_complex = q_pairs[:, 0] + 1j * q_pairs[:, 1]  # [q0+iq1, q2+iq3, q4+iq5, q6+iq7]

print(f"\nJAX Q pairs: {q_pairs}")
print(f"JAX Q complex: {q_complex}")

# Apply RoPE
q_rotated_complex = q_complex * rtheta_jax
q_rotated_pairs = jnp.stack([jnp.real(q_rotated_complex), jnp.imag(q_rotated_complex)], axis=-1)
q_rope_jax = q_rotated_pairs.reshape(head_dim)

print(f"\nJAX Q rotated complex: {q_rotated_complex}")
print(f"JAX Q after RoPE: {q_rope_jax}")

# Compare
diff = np.abs(q_rope_torch.numpy() - q_rope_jax)
print(f"\n\nDifference: {diff}")
print(f"Max diff: {diff.max()}")
print(f"Are they equal? {np.allclose(q_rope_torch.numpy(), q_rope_jax, atol=1e-6)}")
