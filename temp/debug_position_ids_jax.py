import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoTokenizer, ModernBertModel as TorchModernBertModel

from src.models.modernbert.modeling_modernbert import ModernBertModel
from tests.utils import set_attr, t2np

model_name = "answerdotai/ModernBERT-base"
text1 = "hallo nama saya carles dan ini adalah text yang lebih panjang"
text2 = "hallo dunia"

tokenizer = AutoTokenizer.from_pretrained(model_name)

encoded = tokenizer([text1, text2], padding=True, return_tensors="pt")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

print("Attention mask (PyTorch):")
print(attention_mask)
print()

# Create position IDs the HuggingFace way
position_ids_hf = attention_mask.long().cumsum(-1) - 1
position_ids_hf.masked_fill_(attention_mask == 0, 0)
print("HuggingFace position IDs:")
print(position_ids_hf)
print()

# Create position IDs the JAX way
attention_mask_jax = jnp.asarray(attention_mask.numpy(), dtype=jnp.bool_)
attention_mask_int = jnp.asarray(attention_mask_jax, dtype=jnp.int32)
position_ids_jax = jnp.cumsum(attention_mask_int, axis=-1) - 1
position_ids_jax = jnp.where(attention_mask_int == 0, 0, position_ids_jax)

print("JAX position IDs:")
print(position_ids_jax)
print()

print("Position IDs match?", np.allclose(position_ids_jax, position_ids_hf.numpy()))
