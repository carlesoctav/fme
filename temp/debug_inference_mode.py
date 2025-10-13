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
text1 = "hello world"

tokenizer = AutoTokenizer.from_pretrained(model_name)
th_model = TorchModernBertModel.from_pretrained(model_name)
th_model.eval()

encoded = tokenizer([text1], return_tensors="pt")
input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

print("Input IDs:", input_ids)
print("Attention mask:", attention_mask)

# PyTorch forward
with torch.no_grad():
    th_out = th_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    print(f"\nPyTorch output: mean={th_out.mean():.6f}, std={th_out.std():.6f}")
    print(f"PyTorch [0, 0, :5]: {th_out[0, 0, :5].numpy()}")

# JAX forward WITHOUT inference_mode
key = jax.random.key(42)
jx_model = ModernBertModel(th_model.config, key=key)
jx_model = copy_modernbert_weights(jx_model, th_model)

input_ids_jax = jnp.asarray(input_ids.numpy())
attention_mask_jax = jnp.asarray(attention_mask.numpy(), dtype=np.bool_)

jx_out = jx_model(input_ids_jax, attention_mask=attention_mask_jax, key=None)  # key=None for no dropout
print(f"\nJAX output (key=None): mean={jx_out.mean():.6f}, std={jx_out.std():.6f}")
print(f"JAX [0, 0, :5]: {jx_out[0, 0, :5]}")

diff = np.abs(jx_out - th_out.numpy())
print(f"\nDiff (key=None): max={diff.max():.6f}, mean={diff.mean():.6f}")

# JAX forward WITH inference_mode
jx_model_inf = eqx.nn.inference_mode(jx_model)
jx_out_inf = jx_model_inf(input_ids_jax, attention_mask=attention_mask_jax, key=key)
print(f"\nJAX output (inference_mode): mean={jx_out_inf.mean():.6f}, std={jx_out_inf.std():.6f}")
print(f"JAX [0, 0, :5]: {jx_out_inf[0, 0, :5]}")

diff_inf = np.abs(jx_out_inf - th_out.numpy())
print(f"\nDiff (inference_mode): max={diff_inf.max():.6f}, mean={diff_inf.mean():.6f}")
