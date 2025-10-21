"""
Benchmark Flax/Flaxformer BERT 2.5B model initialization.

Config:
- Model: BERT with 2560 hidden, 32 layers, 20 heads, 10240 FFN (~ 2.5B params)
- Mesh: 2D (tp=2, fsdp=2) on 4 TPU devices
- Using Flaxformer BertEncoder

Timing results (single run):
- TBD
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from flaxformer.architectures.bert.bert import BertEncoder
from flaxformer.architectures.bert.heads import MLMHead
from jax.experimental import pjit


def create_bert_large_config():
    """Create a ~2.5B parameter BERT config for Flaxformer."""
    return {
        "vocab_size": 30522,
        "hidden_size": 2560,
        "intermediate_dim": 10240,
        "max_length": 512,
        "num_segments": 2,
        "num_hidden_layers": 32,
        "num_attention_heads": 20,
        "dropout_rate": 0.1,
    }


def main():
    config = create_bert_large_config()
    key = jax.random.key(10)

    devices = jax.devices()
    mesh = jax.make_mesh(
        (2, 1, 1, 2),
        ("data", "x", "y", "model"),
    )

    encoder = BertEncoder(
        vocab_size=config["vocab_size"],
        hidden_size=config["hidden_size"],
        intermediate_dim=config["intermediate_dim"],
        max_length=config["max_length"],
        num_segments=config["num_segments"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        dropout_rate=config["dropout_rate"],
    )

    dummy_input = {
        "token_ids": jnp.zeros((1, config["max_length"]), dtype=jnp.int32),
        "position_ids": jnp.arange(config["max_length"])[None, :],
        "segment_ids": jnp.zeros((1, config["max_length"]), dtype=jnp.int32),
        "input_mask": jnp.ones((1, config["max_length"]), dtype=jnp.int32),
    }

    print("Initializing Flaxformer BERT 2.5B model...")
    start = time.monotonic()

    def init_module():
        encoder_params = encoder.init(
            key,
            **dummy_input,
            enable_dropout=False,
        )

        return encoder_params

    params = init_module()

    jax.block_until_ready(params)
    diff = time.monotonic() - start
    print(f"Flax init time: {diff:.2f}s")


if __name__ == "__main__":
    main()
