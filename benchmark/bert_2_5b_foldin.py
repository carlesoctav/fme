"""
Benchmark BERT 2.5B with fold_in instead of split for RNG keys.

This tests whether replacing jax.random.split() with jax.random.fold_in()
reduces HLO complexity and improves compilation time.

Expected improvement: 606 slice/reshape ops → ~100-150 ops
Target compilation time: ~40-50s (closer to Flax's 33s)
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from transformers.models.bert.configuration_bert import BertConfig

from src.models.bert import BertModel


def create_bert_large_config():
    """Create a ~2.5B parameter BERT config."""
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


def main():
    print("=" * 80)
    print("BERT 2.5B Initialization Benchmark: fold_in vs split")
    print("=" * 80)

    config = create_bert_large_config()
    key = jax.random.key(10)

    mesh = jax.make_mesh((2, 2), ("fsdp", "tp"))

    # First, benchmark the ORIGINAL version (with split)
    print("\n[1/2] Benchmarking ORIGINAL (using jax.random.split)...")

    @jax.jit
    def init_original():
        return BertModel(config, key=key)

    start = time.monotonic()
    with mesh:
        model_orig = init_original()
    jax.block_until_ready(jax.tree.leaves(model_orig))
    time_orig = time.monotonic() - start

    print(f"✓ Original init time: {time_orig:.2f}s")

    # TODO: Next, we would benchmark the fold_in version after implementing it
    print("\n[2/2] fold_in version: NOT YET IMPLEMENTED")
    print("      Need to modify BertEncoder, BertLayer, etc. to use fold_in")

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"Original (split):  {time_orig:.2f}s")
    print(f"fold_in version:   TBD")
    print(f"Target (Flax):     47s")
    print("\nNext steps:")
    print("1. Create modified BertEncoder that uses fold_in instead of split")
    print("2. Modify BertLayer, BertAttention to propagate fold_in style")
    print("3. Update Linear, LayerNorm, etc. to use fold_in with static keys")
    print("4. Re-run benchmark and compare HLO complexity")


if __name__ == "__main__":
    main()
