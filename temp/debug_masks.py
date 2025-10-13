import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
from src._masking_utils import make_full_mask, slliding_window_full_mask

# Simple test case
B, T, H = 1, 5, 768
input_embeds = jnp.zeros((B, T, H))
attention_mask = jnp.ones((B, T), dtype=jnp.bool_)

# Test full mask
full_mask = make_full_mask(
    mask_impl="sdpa",
    input_embeds=input_embeds,
    attention_mask=attention_mask,
    segment_ids=None,
)
print("Full mask shape:", full_mask.shape)
print("Full mask (batch 0):")
print(full_mask[0])

# Test sliding window mask with window_size=1 (half of local_attention=2)
sliding_mask = slliding_window_full_mask(
    mask_impl="sdpa",
    input_embeds=input_embeds,
    window_size=1,
    attention_mask=attention_mask,
    segment_ids=None,
)
print("\nSliding window mask shape:", sliding_mask.shape)
print("Sliding window mask (batch 0, window_size=1):")
print(sliding_mask[0])
