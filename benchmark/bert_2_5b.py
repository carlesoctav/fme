"""
Benchmark apply_transforms (TP + FSDP) on 2.5B BERT model.

Config:
- Model: BERT with 2560 hidden, 32 layers, 20 heads, 10240 FFN (~ 2.5B params)
- Mesh: 2D (tp=2, fsdp=2) on 4 TPU devices
- TP: Q/K/V column parallel, attention output row parallel, FFN w1 column, w2 row
- FSDP: Each BertLayer sharded on fsdp axis

Timing results (single run):
- Transform: TBD
- Unbox: TBD
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding, SingleDeviceSharding
from transformers.models.bert.configuration_bert import BertConfig

from src.distributed.params import fully_shard, unbox_params
from src.filter import apply_transforms
from src.distributed.tp import column_parallel, row_parallel
from src.models.bert import BertModel
from src.nn import Linear


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
    )


def create_transform_dict(mesh, num_layers):
    """Create pattern-to-transform dictionary for apply_transforms."""

    # TP transforms only (will be applied first)
    tp_transforms = {}

    # TP transforms for attention Q/K/V (column parallel)
    for proj in ["query", "key", "value"]:
        tp_transforms[f"encoder.layer.*.attention.self.{proj}"] = partial(
            column_parallel,
            axis_name="tp",
            mesh=mesh,
            outputs_layout=P(None, "tp"),
        )

    # TP transform for attention output (row parallel)
    tp_transforms["encoder.layer.*.attention.output.dense"] = partial(
        row_parallel,
        axis_name="tp",
        mesh=mesh,
        outputs_layout=P(),
    )

    # TP transform for FFN intermediate (column parallel)
    tp_transforms["encoder.layer.*.intermediate.dense"] = partial(
        column_parallel,
        axis_name="tp",
        mesh=mesh,
        outputs_layout=P(None, "tp"),
    )

    # TP transform for FFN output (row parallel)
    tp_transforms["encoder.layer.*.output.dense"] = partial(
        row_parallel,
        axis_name="tp",
        mesh=mesh,
        outputs_layout=P(),
    )

    # FSDP transforms (will be applied second)
    # Use exact layer indices to avoid matching nested children
    fsdp_transforms = {}
    for i in range(num_layers):
        fsdp_transforms[f"encoder.layer.{i}"] = partial(
            fully_shard,
            mesh=mesh,
            axis_name="fsdp",
        )

    return tp_transforms, fsdp_transforms


def create_sharded_array(shape, mesh, pspec, key=None):
    """Create an array directly sharded across devices."""
    sharding = NamedSharding(mesh, pspec)
    if key is None:
        # Use zeros for initialization
        return jax.device_put(jnp.zeros(shape), sharding)
    else:
        # Use random initialization
        return jax.device_put(jax.random.normal(key, shape), sharding)


def create_model_sharded(config, key, mesh):
    """Create model - for large models this creates on default device first."""
    print("  Creating model...")
    model = BertModel(config, key=key)
    return model


def main():
    devices = jax.devices()
    print(f"Number of devices: {len(devices)}")
    print(f"Device type: {devices[0].platform}")

    # Create 2D mesh: tp=2, fsdp=2 (for 4 TPU devices)
    mesh_shape = (2, 2)
    mesh = Mesh(np.array(devices[:4]).reshape(mesh_shape), axis_names=("tp", "fsdp"))
    print(f"Mesh shape: {mesh_shape}, axes: tp={mesh_shape[0]}, fsdp={mesh_shape[1]}")

    # Create model
    config = create_bert_large_config()
    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")

    # Estimate parameter count
    # Embedding: vocab_size * hidden_size + max_pos * hidden_size + type_vocab * hidden_size
    embed_params = (
        config.vocab_size + config.max_position_embeddings + config.type_vocab_size
    ) * config.hidden_size
    # Per layer: 4 * hidden^2 (Q,K,V,O) + 2 * hidden * intermediate (FFN) + layer norms
    layer_params = (
        4 * config.hidden_size**2 + 2 * config.hidden_size * config.intermediate_size
    ) * config.num_hidden_layers
    total_params = embed_params + layer_params
    print(f"  Estimated params: {total_params / 1e9:.2f}B")

    key = jax.random.PRNGKey(42)

    # Create model with arrays sharded across all devices from the start
    print("\nCreating model with sharded arrays...")
    model = create_model_sharded(config, key, mesh)

    # Create transform dictionaries
    print("\nCreating transform dictionaries...")
    tp_transforms, fsdp_transforms = create_transform_dict(
        mesh, config.num_hidden_layers
    )
    print(f"  TP patterns: {len(tp_transforms)}")
    print(f"  FSDP patterns: {len(fsdp_transforms)}")

    # Benchmark apply_transforms
    print("\n" + "=" * 60)
    print("Benchmarking apply_transforms with TP + FSDP")
    print("=" * 60)

    # Single timed run
    print("\nApplying transforms...")
    start = time.perf_counter()
    model = apply_transforms(model, tp_transforms)
    transformed_model = apply_transforms(model, fsdp_transforms)
    jax.block_until_ready(jax.tree.leaves(transformed_model))
    end = time.perf_counter()

    transform_time = end - start
    print(f"Transform time: {transform_time:.4f}s")

    # Verify sharding BEFORE unboxing
    print(
        "\nVerifying sharding (before unbox) on encoder.layer.0.attention.self.query:"
    )
    sample_weight_before = transformed_model.encoder.layer[
        0
    ].attention.self.query.weight.value
    print(f"  Shape: {sample_weight_before.shape}")
    print(f"  Sharding: {sample_weight_before.sharding}")

    # Unbox and verify
    print("\nUnboxing params...")
    start = time.perf_counter()
    unboxed = unbox_params(transformed_model, mesh)
    jax.block_until_ready(jax.tree.leaves(unboxed))
    end = time.perf_counter()

    unbox_time = end - start
    print(f"Unbox time: {unbox_time:.4f}s")

    # Verify sharding AFTER unboxing (should be None)
    print("\nVerifying after unbox:")
    sample_weight_after = unboxed.encoder.layer[0].attention.self.query.weight
    print(f"  Shape: {sample_weight_after.shape}")
    print(f"  Sharding: {sample_weight_after.sharding}")

    print("\nDone!")


if __name__ == "__main__":
    import equinox as eqx

    main()
