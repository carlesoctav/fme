import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

text1 = "hallo nama saya carles dan ini adalah text yang lebih panjang"
text2 = "hallo dunia"

# Simulate tokenizer output
# text1 is longer, text2 is shorter and gets padded
input_ids = np.array([
    [101, 25966, 10932, 24446, 22433, 8857, 13232, 12869, 3087, 8872, 102, 0, 0],  # text1 (real length: 11)
    [101, 25966, 18339, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # text2 (real length: 4, padded with 0s)
])

attention_mask = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],  # text1
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # text2
])

print("Input IDs shape:", input_ids.shape)
print("Attention mask:")
print(attention_mask)
print()

# JAX's current approach: simple range for all positions
batch_size, seq_len = input_ids.shape
position_ids_jax = jnp.broadcast_to(
    jnp.arange(seq_len, dtype=jnp.int32),
    (batch_size, seq_len)
)

print("JAX position IDs (current approach):")
print(position_ids_jax)
print()

# HuggingFace's approach: position IDs increment only for non-padded tokens
# For padded positions, position_ids should likely remain at the last valid position
# OR they could be 0, but that doesn't matter since attention mask zeros them out

# Let's check if position_ids should respect padding
# Looking at the attention mask, we can create position_ids that increment only for valid tokens

position_ids_hf = np.zeros_like(input_ids, dtype=np.int32)
for i in range(batch_size):
    pos = 0
    for j in range(seq_len):
        if attention_mask[i, j] == 1:
            position_ids_hf[i, j] = pos
            pos += 1
        else:
            # For padded positions, keep position as 0 or last valid position
            position_ids_hf[i, j] = 0  # or pos - 1

print("HuggingFace-style position IDs (respecting padding):")
print(position_ids_hf)
print()

print("Difference:")
print(position_ids_jax - position_ids_hf)
