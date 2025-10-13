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

def copy_modernbert_weights(jx_model: ModernBertModel, th_model: TorchModernBertModel):
    jx_model = set_attr(
        jx_model,
        "embeddings.tok_embeddings.weight",
        t2np(th_model.embeddings.tok_embeddings.weight),
    )
    jx_model = set_attr(
        jx_model,
        "embeddings.norm.weight",
        t2np(th_model.embeddings.norm.weight),
    )

    for i, th_layer in enumerate(th_model.layers):
        if hasattr(th_layer, "attn_norm") and hasattr(th_layer.attn_norm, "weight"):
            jx_model = set_attr(
                jx_model,
                f"encoder.layers.{i}.attn_norm.weight",
                t2np(th_layer.attn_norm.weight),
            )

        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.attention.Wqkv.weight",
            t2np(th_layer.attn.Wqkv.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.attention.Wo.weight",
            t2np(th_layer.attn.Wo.weight),
        )

        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.mlp_norm.weight",
            t2np(th_layer.mlp_norm.weight),
        )

        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.mlp.Wi.weight",
            t2np(th_layer.mlp.Wi.weight),
        )
        jx_model = set_attr(
            jx_model,
            f"encoder.layers.{i}.mlp.Wo.weight",
            t2np(th_layer.mlp.Wo.weight),
        )

    jx_model = set_attr(
        jx_model,
        "final_norm.weight",
        t2np(th_model.final_norm.weight),
    )

    return jx_model

model_name = "answerdotai/ModernBERT-base"
text1 = "hello"

tokenizer = AutoTokenizer.from_pretrained(model_name)
th_model = TorchModernBertModel.from_pretrained(model_name)
th_model.eval()

encoded = tokenizer([text1], return_tensors="pt")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

print("Input IDs:", input_ids)
print("Seq len:", input_ids.shape[1])

# Setup JAX model
key = jax.random.key(42)
jx_model = ModernBertModel(th_model.config, key=key)
jx_model = copy_modernbert_weights(jx_model, th_model)
jx_model = eqx.nn.inference_mode(jx_model)

input_ids_jax = jnp.asarray(input_ids.numpy())

# Get embeddings
with torch.no_grad():
    th_embeds = th_model.embeddings(input_ids)
    
jx_embeds = jx_model.embeddings(input_ids_jax)

print("\n=== LAYER 0 ===")
print(f"Layer 0 is: {'GLOBAL' if jx_model.encoder.layers[0].use_global else 'LOCAL'}")

# PyTorch Layer 0
with torch.no_grad():
    # Layer 0 has Identity norm
    th_normed = th_embeds  # No normalization
    
    # QKV projection
    th_qkv = th_model.layers[0].attn.Wqkv(th_normed)
    print(f"\nPyTorch QKV: shape={th_qkv.shape}, mean={th_qkv.mean():.6f}")
    print(f"PyTorch QKV[0, 0, :5]: {th_qkv[0, 0, :5].numpy()}")
    
    # Reshape to separate Q, K, V
    B, T, _ = th_qkv.shape
    th_qkv_reshaped = th_qkv.reshape(B, T, 3, 12, 64)  # (B, T, 3, num_heads, head_dim)
    th_q, th_k, th_v = th_qkv_reshaped[:, :, 0], th_qkv_reshaped[:, :, 1], th_qkv_reshaped[:, :, 2]
    print(f"\nPyTorch Q before RoPE: shape={th_q.shape}, mean={th_q.mean():.6f}")
    print(f"PyTorch Q[0, 0, 0, :5]: {th_q[0, 0, 0, :5].numpy()}")
    
# JAX Layer 0
jx_normed = jx_embeds  # Layer 0 has Identity norm

jx_qkv = jx_model.encoder.layers[0].attention.Wqkv(jx_normed)
print(f"\nJAX QKV: shape={jx_qkv.shape}, mean={jx_qkv.mean():.6f}")
print(f"JAX QKV[0, 0, :5]: {jx_qkv[0, 0, :5]}")

qkv_diff = np.abs(jx_qkv - th_qkv.numpy())
print(f"\nQKV diff: max={qkv_diff.max():.10f}")

# Reshape JAX QKV
B, T, _ = jx_qkv.shape
jx_qkv_reshaped = jx_qkv.reshape(B, T, 3, 12, 64)
jx_q_pre, jx_k_pre, jx_v = jnp.split(jx_qkv_reshaped, 3, axis=2)
jx_q_pre = jnp.squeeze(jx_q_pre, axis=2)
jx_k_pre = jnp.squeeze(jx_k_pre, axis=2)
jx_v = jnp.squeeze(jx_v, axis=2)

print(f"\nJAX Q before RoPE: shape={jx_q_pre.shape}, mean={jx_q_pre.mean():.6f}")
print(f"JAX Q[0, 0, 0, :5]: {jx_q_pre[0, 0, 0, :5]}")

q_diff = np.abs(jx_q_pre - th_q.numpy())
print(f"\nQ (before RoPE) diff: max={q_diff.max():.10f}")

# Now apply RoPE and compare
print("\n=== RoPE Application ===")

# PyTorch RoPE
with torch.no_grad():
    # Create position_ids
    position_ids = torch.arange(T).unsqueeze(0)  # (1, T)
    
    # Get cos, sin from rotary_emb - it takes qkv and position_ids
    cos, sin = th_model.layers[0].attn.rotary_emb(th_qkv_reshaped, position_ids=position_ids)
    print(f"PyTorch cos: shape={cos.shape}, mean={cos.mean():.6f}")
    print(f"PyTorch sin: shape={sin.shape}, mean={sin.mean():.6f}")
    print(f"PyTorch cos[0, 0, :5]: {cos[0, 0, :5].numpy()}")
    print(f"PyTorch sin[0, 0, :5]: {sin[0, 0, :5].numpy()}")
    
    # Need to transpose for apply_rotary_pos_emb
    # qkv_reshaped: (B, T, 3, num_heads, head_dim) -> (B, num_heads, T, head_dim)
    th_q_for_rope = th_q.transpose(1, 2)  # (B, num_heads, T, head_dim)
    th_k_for_rope = th_k.transpose(1, 2)
    
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
    th_q_roped, th_k_roped = apply_rotary_pos_emb(th_q_for_rope, th_k_for_rope, cos, sin)
    
    # Transpose back: (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim)
    th_q_roped = th_q_roped.transpose(1, 2)
    th_k_roped = th_k_roped.transpose(1, 2)
    
    print(f"\nPyTorch Q after RoPE: shape={th_q_roped.shape}, mean={th_q_roped.mean():.6f}")
    print(f"PyTorch Q[0, 0, 0, :5]: {th_q_roped[0, 0, 0, :5].numpy()}")
    print(f"PyTorch K after RoPE: shape={th_k_roped.shape}, mean={th_k_roped.mean():.6f}")

# JAX RoPE
jx_rope = jx_model.encoder.layers[0].attention.rotary_emb

# JAX RoPE takes the actual hidden states (Q or K)
# Create position_ids for JAX (same as PyTorch)
position_ids_jax = jnp.arange(T).reshape(1, T)  # (1, T)

# Apply RoPE to Q and K
jx_q_roped = jx_rope(jx_q_pre, position_ids=position_ids_jax)
jx_k_roped = jx_rope(jx_k_pre, position_ids=position_ids_jax)

print(f"\nJAX Q after RoPE: shape={jx_q_roped.shape}, mean={jx_q_roped.mean():.6f}")
print(f"JAX Q[0, 0, 0, :5]: {jx_q_roped[0, 0, 0, :5]}")
print(f"JAX K after RoPE: shape={jx_k_roped.shape}, mean={jx_k_roped.mean():.6f}")

q_roped_diff = np.abs(jx_q_roped - th_q_roped.numpy())
k_roped_diff = np.abs(jx_k_roped - th_k_roped.numpy())
print(f"\nQ (after RoPE) diff: max={q_roped_diff.max():.10f}, mean={q_roped_diff.mean():.10f}")
print(f"K (after RoPE) diff: max={k_roped_diff.max():.10f}, mean={k_roped_diff.mean():.10f}")

# Check if the differences are in specific positions
print(f"\nQ diff > 0.01: {np.sum(q_roped_diff > 0.01)} values")
print(f"Q diff > 0.1: {np.sum(q_roped_diff > 0.1)} values")
print(f"Q diff > 1.0: {np.sum(q_roped_diff > 1.0)} values")

# Debug the RoPE formulas
print("\n=== Debugging RoPE Formulas ===")
print(f"PyTorch Q before RoPE (unchanged): {th_q[0, 0, 0, :5].numpy()}")
print(f"PyTorch Q after RoPE (same!): {th_q_roped[0, 0, 0, :5].numpy()}")

# Check why PyTorch Q didn't change
print(f"\ncos[0, 0, :10]: {cos[0, 0, :10].numpy()}")
print(f"sin[0, 0, :10]: {sin[0, 0, :10].numpy()}")

# Manual computation for position 0
with torch.no_grad():
    from transformers.models.modernbert.modeling_modernbert import rotate_half
    q_test = th_q_for_rope[0, 0, 0, :]  # First head, first position
    cos_test = cos[0, 0, :]  # First position
    sin_test = sin[0, 0, :]  # First position
    
    rotated = rotate_half(q_test)
    print(f"\nq_test[:10]: {q_test[:10].numpy()}")
    print(f"rotate_half(q_test)[:10]: {rotated[:10].numpy()}")
    
    result = (q_test * cos_test) + (rotated * sin_test)
    print(f"Manual result[:10]: {result[:10].numpy()}")
    print(f"PyTorch result[:10]: {th_q_roped[0, 0, 0, :10].numpy()}")

# Check JAX approach
print(f"\n\nJAX approach (complex numbers):")
print(f"JAX Q before RoPE: {jx_q_pre[0, 0, 0, :5]}")
print(f"JAX Q after RoPE: {jx_q_roped[0, 0, 0, :5]}")

# Check JAX rtheta
jx_rtheta = jx_rope.rtheta  # (max_seq, halfdim)
print(f"\nJAX rtheta shape: {jx_rtheta.shape}")
print(f"JAX rtheta[0, :5] (position 0): {jx_rtheta[0, :5]}")
print(f"JAX rtheta[0] is complex: {jnp.iscomplexobj(jx_rtheta)}")

# Extract cos/sin from complex rtheta
cos_jax = jnp.real(jx_rtheta[0])
sin_jax = jnp.imag(jx_rtheta[0])
print(f"\ncos_jax[:10]: {cos_jax[:10]}")
print(f"sin_jax[:10]: {sin_jax[:10]}")

# Check position 1 (should have non-zero sin)
print(f"\n\n=== Position 1 (should have rotation) ===")
print(f"PyTorch cos[0, 1, :10]: {cos[0, 1, :10].numpy()}")
print(f"PyTorch sin[0, 1, :10]: {sin[0, 1, :10].numpy()}")

print(f"\nPyTorch Q[0, 1, 0, :5] before RoPE: {th_q[0, 1, 0, :5].numpy()}")
print(f"PyTorch Q[0, 1, 0, :5] after RoPE: {th_q_roped[0, 1, 0, :5].numpy()}")

print(f"\nJAX Q[0, 1, 0, :5] before RoPE: {jx_q_pre[0, 1, 0, :5]}")
print(f"JAX Q[0, 1, 0, :5] after RoPE: {jx_q_roped[0, 1, 0, :5]}")

cos_jax_1 = jnp.real(jx_rtheta[1])
sin_jax_1 = jnp.imag(jx_rtheta[1])
print(f"\nJAX cos[1, :10]: {cos_jax_1[:10]}")
print(f"JAX sin[1, :10]: {sin_jax_1[:10]}")

# Compare the cos/sin values
print(f"\n\nCompare cos/sin at position 1:")
print(f"PyTorch - JAX cos diff: max={np.abs(cos[0, 1].numpy() - cos_jax_1).max():.10f}")
print(f"PyTorch - JAX sin diff: max={np.abs(sin[0, 1].numpy() - sin_jax_1).max():.10f}")
