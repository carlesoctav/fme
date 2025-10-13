import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
from transformers import AutoTokenizer, ModernBertModel as TorchModernBertModel
import inspect

model_name = "answerdotai/ModernBERT-base"
th_model = TorchModernBertModel.from_pretrained(model_name)

# Get the RoPE class
rope = th_model.layers[0].attn.rotary_emb

print("RoPE object:", rope)
print("\nRoPE attributes:")
for attr in dir(rope):
    if not attr.startswith('_'):
        val = getattr(rope, attr, None)
        if not callable(val):
            print(f"  {attr}: {val}")

print("\n\nRoPE forward signature:")
print(inspect.signature(rope.forward))

print("\n\nRoPE forward source:")
try:
    print(inspect.getsource(rope.forward))
except:
    print("Could not get source")

# Check if attention_scaling is used
print("\n\nChecking attention_scaling usage...")
text1 = "hello world"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoded = tokenizer(text1, return_tensors="pt")

with torch.no_grad():
    # Get Q from first layer
    embeds = th_model.embeddings(encoded["input_ids"])
    qkv = th_model.layers[0].attn.Wqkv(embeds)
    B, T, _ = qkv.shape
    qkv = qkv.reshape(B, T, 3, th_model.config.num_attention_heads, -1)
    q = qkv[:, :, 0, :, :]  # (B, T, N, H)
    
    print(f"\nQ shape before RoPE: {q.shape}")
    
    # Apply RoPE
    position_ids = torch.arange(T).unsqueeze(0)
    q_rot = rope(q, position_ids)
    
    print(f"Q shape after RoPE: {q_rot.shape}")
    print(f"Scaling factor applied: check if output magnitude changed")
    print(f"Q before RoPE norm: {q.norm()}")
    print(f"Q after RoPE norm: {q_rot.norm()}")
