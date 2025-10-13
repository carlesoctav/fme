import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    ModernBertModel as TorchModernBertModel,
)

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
text1 = "hallo nama saya carles dan ini adalah text yang lebih panjang"
text2 = "hallo dunia"

print(f"Loading tokenizer and model from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
th_model = TorchModernBertModel.from_pretrained(model_name)
th_model.eval()

print(f"PyTorch model config._attn_implementation: {th_model.config._attn_implementation}")

encoded = tokenizer(
    [text1, text2],
    padding=True,
    return_tensors="pt",
)
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

print(f"Input shape: {input_ids.shape}")
print(f"Attention mask shape: {attention_mask.shape}")

with torch.no_grad():
    th_out = th_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state

print(f"PyTorch output shape: {th_out.shape}")
print(f"PyTorch output stats: mean={th_out.mean():.6f}, std={th_out.std():.6f}, min={th_out.min():.6f}, max={th_out.max():.6f}")

key = jax.random.key(42)
print(f"\nCreating JAX model with config._attn_implementation: {th_model.config._attn_implementation}")
jx_model = ModernBertModel(th_model.config, key=key)
jx_model = copy_modernbert_weights(jx_model, th_model)
jx_model = eqx.nn.inference_mode(jx_model)

jx_out = jx_model(
    jnp.asarray(input_ids.numpy()),
    attention_mask=jnp.asarray(attention_mask.numpy(), dtype=np.bool_),
    key=key,
)

print(f"\nJAX output shape: {jx_out.shape}")
print(f"JAX output stats: mean={jx_out.mean():.6f}, std={jx_out.std():.6f}, min={jx_out.min():.6f}, max={jx_out.max():.6f}")

diff = np.abs(jx_out - th_out.numpy())
print(f"\nAbsolute difference: mean={diff.mean():.6f}, max={diff.max():.6f}")

mismatch = diff > 5e-2
mismatch_pct = mismatch.sum() / mismatch.size * 100
print(f"Mismatch percentage (>5e-2): {mismatch_pct:.2f}%")

print("\nAttempting assertion...")
try:
    np.testing.assert_allclose(jx_out, th_out.numpy(), atol=5e-2, rtol=5e-2)
    print("✅ Test passed!")
except AssertionError as e:
    print(f"❌ Test failed: {e}")
