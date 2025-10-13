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
text1 = "hallo nama saya carles dan ini adalah text yang lebih panjang"
text2 = "hallo dunia"

tokenizer = AutoTokenizer.from_pretrained(model_name)
th_model = TorchModernBertModel.from_pretrained(model_name)
th_model.eval()

encoded = tokenizer([text1, text2], padding=True, return_tensors="pt")
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
attention_mask_jax = jnp.asarray(attention_mask.numpy(), dtype=np.bool)

# Get embeddings
with torch.no_grad():
    th_embeds = th_model.embeddings(input_ids)
    
jx_embeds = jx_model.embeddings(input_ids_jax)

print("\n=== EMBEDDINGS ===")
embed_diff = np.abs(jx_embeds - th_embeds.numpy())
print(f"Embeddings diff: max={embed_diff.max():.10f}")

# Run full model
print("\n=== FULL MODEL ===")
with torch.no_grad():
    th_out = th_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).last_hidden_state
    
    print(f"PyTorch output: shape={th_out.shape}, mean={th_out.mean():.6f}, std={th_out.std():.6f}")
    print(f"PyTorch output[0, 0, :5]: {th_out[0, 0, :5].numpy()}")

jx_out = jx_model(
    input_ids_jax,
    attention_mask=attention_mask_jax,
    key=key,
)

print(f"\nJAX output: shape={jx_out.shape}, mean={jx_out.mean():.6f}, std={jx_out.std():.6f}")
print(f"JAX output[0, 0, :5]: {jx_out[0, 0, :5]}")

out_diff = np.abs(jx_out - th_out.numpy())
print(f"\nOutput diff: max={out_diff.max():.10f}, mean={out_diff.mean():.10f}")
print(f"Matches? {np.allclose(jx_out, th_out.numpy(), atol=5e-2)}")
