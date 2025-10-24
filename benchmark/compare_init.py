"""
Compare initialization between Equinox and Flax.
This script traces and analyzes the differences.
"""

import time
import jax
import jax.numpy as jnp
from transformers.models.bert.configuration_bert import BertConfig
from flaxformer.architectures.bert.bert import BertEncoder
from src.models.bert import BertModel


def create_config():
    """Small config for quick testing."""
    return BertConfig(
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        vocab_size=30522,
        type_vocab_size=2,
        _attn_implementation="sdpa",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def test_equinox():
    print("=" * 60)
    print("Testing Equinox BertModel")
    print("=" * 60)

    config = create_config()
    key = jax.random.key(10)

    @jax.jit
    def init_eqx():
        return BertModel(config, key=key)

    # Warmup
    print("Warming up...")
    _ = init_eqx()
    jax.block_until_ready(_)

    # Time it
    print("Timing...")
    start = time.monotonic()
    module = init_eqx()
    jax.block_until_ready(jax.tree.leaves(module))
    diff = time.monotonic() - start

    print(f"Equinox init time: {diff:.4f}s")

    # Get compilation info
    lowered = init_eqx.lower()
    print(f"\nCompiler info:")
    print(f"  Output type: {type(module)}")

    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(module) if hasattr(x, "size"))
    print(f"  Parameters: {param_count:,}")

    # Save HLO
    with open("/tmp/eqx_hlo.txt", "w") as f:
        f.write(lowered.as_text())
    print(f"  HLO saved to /tmp/eqx_hlo.txt")

    return module, diff


def test_flax():
    print("\n" + "=" * 60)
    print("Testing Flaxformer BertEncoder")
    print("=" * 60)

    key = jax.random.key(10)

    encoder = BertEncoder(
        vocab_size=30522,
        hidden_size=768,
        intermediate_dim=3072,
        max_length=512,
        num_segments=2,
        num_hidden_layers=2,
        num_attention_heads=12,
        dropout_rate=0.0,
    )

    dummy_input = {
        "token_ids": jnp.zeros((1, 512), dtype=jnp.int32),
        "position_ids": jnp.arange(512)[None, :],
        "segment_ids": jnp.zeros((1, 512), dtype=jnp.int32),
        "input_mask": jnp.ones((1, 512), dtype=jnp.int32),
    }

    @jax.jit
    def init_flax():
        return encoder.init(key, **dummy_input, enable_dropout=False)

    # Warmup
    print("Warming up...")
    _ = init_flax()
    jax.block_until_ready(_)

    # Time it
    print("Timing...")
    start = time.monotonic()
    params = init_flax()
    jax.block_until_ready(params)
    diff = time.monotonic() - start

    print(f"Flax init time: {diff:.4f}s")

    # Get compilation info
    lowered = init_flax.lower()
    print(f"\nCompiler info:")
    print(f"  Output type: {type(params)}")

    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Parameters: {param_count:,}")

    # Save HLO
    with open("/tmp/flax_hlo.txt", "w") as f:
        f.write(lowered.as_text())
    print(f"  HLO saved to /tmp/flax_hlo.txt")

    return params, diff


def main():
    eqx_module, eqx_time = test_equinox()
    flax_params, flax_time = test_flax()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Equinox: {eqx_time:.4f}s")
    print(f"Flax:    {flax_time:.4f}s")
    print(f"Ratio:   {eqx_time / flax_time:.2f}x")
    print(f"\nEquinox is {eqx_time / flax_time:.2f}x slower than Flax")

    print("\nAnalyze the HLO files to understand the difference:")
    print("  /tmp/eqx_hlo.txt")
    print("  /tmp/flax_hlo.txt")


if __name__ == "__main__":
    main()
