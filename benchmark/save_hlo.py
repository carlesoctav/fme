"""
Save HLO for large models to compare.
"""

import jax
import jax.numpy as jnp
from transformers.models.bert.configuration_bert import BertConfig
from flaxformer.architectures.bert.bert import BertEncoder
from src.models.bert import BertModel


def create_large_config():
    return BertConfig(
        hidden_size=2560,
        num_hidden_layers=32,
        num_attention_heads=20,
        intermediate_size=10240,
        max_position_embeddings=512,
        vocab_size=30522,
        type_vocab_size=2,
        _attn_implementation="sdpa",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )


def save_equinox_hlo():
    print("Generating Equinox HLO...")
    config = create_large_config()
    key = jax.random.key(10)

    @jax.jit
    def init_eqx():
        return BertModel(config, key=key)

    lowered = init_eqx.lower()

    with open("/tmp/eqx_large_hlo.txt", "w") as f:
        f.write(lowered.as_text())

    print(f"  Saved to /tmp/eqx_large_hlo.txt")

    # Get stats
    hlo_text = lowered.as_text()
    print(f"  HLO size: {len(hlo_text):,} chars")
    print(f"  HLO lines: {hlo_text.count(chr(10)):,}")


def save_flax_hlo():
    print("\nGenerating Flax HLO...")
    key = jax.random.key(10)

    encoder = BertEncoder(
        vocab_size=30522,
        hidden_size=2560,
        intermediate_dim=10240,
        max_length=512,
        num_segments=2,
        num_hidden_layers=32,
        num_attention_heads=20,
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

    lowered = init_flax.lower()

    with open("/tmp/flax_large_hlo.txt", "w") as f:
        f.write(lowered.as_text())

    print(f"  Saved to /tmp/flax_large_hlo.txt")

    # Get stats
    hlo_text = lowered.as_text()
    print(f"  HLO size: {len(hlo_text):,} chars")
    print(f"  HLO lines: {hlo_text.count(chr(10)):,}")


def main():
    save_equinox_hlo()
    save_flax_hlo()

    print("\nComparison:")
    print("  Use 'wc -l /tmp/*_large_hlo.txt' to compare line counts")
    print("  Use 'head -n 200 /tmp/eqx_large_hlo.txt' to inspect Equinox HLO")
    print("  Use 'head -n 200 /tmp/flax_large_hlo.txt' to inspect Flax HLO")


if __name__ == "__main__":
    main()
