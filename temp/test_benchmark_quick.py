"""Quick test that the benchmark script setup works."""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from transformers import BertConfig
from jax.sharding import Mesh

from src.training_utils import Optimizer
from src.distributed.params import unbox_params
from src.models.bert import BertForMaskedLM
from src.modeling_utils import Rngs

SEED = 42


def main():
    key = jr.PRNGKey(SEED)

    devices = jax.devices()
    mesh = Mesh(devices, ("dp",))

    config = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        _attn_implementation="eager",
    )

    grad_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=5e-5, weight_decay=0.01),
    )

    print("Creating model and optimizer...")
    with mesh:
        model = BertForMaskedLM(config, rngs=Rngs(params=key, dropout=key))
        model = unbox_params(model)
        optimizer = Optimizer(model, grad_tx, wrt=eqx.is_array)

    print("✓ Model and optimizer created successfully")

    # Create fake batch and gradients
    batch_size = 4
    seq_len = 512

    fake_input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    fake_position_ids = jnp.broadcast_to(
        jnp.arange(seq_len)[None, :], (batch_size, seq_len)
    )
    fake_attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    fake_token_type_ids = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    print("Testing forward pass...")
    key, dropout_key = jr.split(key)
    logits = model(
        input_ids=fake_input_ids,
        position_ids=fake_position_ids,
        token_type_ids=fake_token_type_ids,
        attention_mask=fake_attention_mask,
        rngs=Rngs(dropout=dropout_key),
    )
    print(f"✓ Forward pass works, logits shape: {logits.shape}")

    # Test gradient computation
    print("Testing gradient computation...")

    def loss_fn(model):
        logits = model(
            input_ids=fake_input_ids,
            position_ids=fake_position_ids,
            token_type_ids=fake_token_type_ids,
            attention_mask=fake_attention_mask,
            rngs=Rngs(dropout=dropout_key),
        )
        return jnp.mean(logits**2)

    grad_fn = eqx.filter_value_and_grad(loss_fn)
    loss, grads = grad_fn(model)
    print(f"✓ Gradient computation works, loss: {loss:.4f}")

    # Test optimizer update
    print("Testing optimizer update...")
    new_model, new_optimizer = optimizer(grads, model)
    print("✓ Optimizer update works!")

    return 0


if __name__ == "__main__":
    exit(main())
