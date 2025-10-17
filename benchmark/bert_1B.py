"""
Benchmark apply_transforms (TP + FSDP) on 1B BERT model.

Config:
- Model: BERT with 2048 hidden, 24 layers, 32 heads, 8192 FFN (~ 1.27B params)
- Mesh: 2D (tp=2, fsdp=2) on 4 TPU devices
- TP: Q/K/V column parallel, attention output row parallel, FFN w1 column, w2 row
- FSDP: Each BertLayer sharded on fsdp axis

- transfomrs: 0.2860s (first timed run, includes compilation)
- Unbox time: 0.8182s

"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P
from transformers.models.bert.configuration_bert import BertConfig

from src.distributed.params import fully_shard, unbox_params
from src.filter import apply_transforms
from src.distributed.tp import column_parallel, row_parallel
from src.models.bert import BertModel
from src.nn import Linear


def create_bert_1b_config():
    """Create a ~1B parameter BERT config."""
    return BertConfig(
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        intermediate_size=8192,
        max_position_embeddings=512,
        vocab_size=30522,
        type_vocab_size=2,
        _attn_implementation="sdpa",
    )


def create_transform_dict(mesh):
    """Create pattern-to-transform dictionary for apply_transforms."""
    from functools import partial

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
    # Use ? to match single digit (0-9) or ?? for two digits
    # This matches encoder.layer.0, encoder.layer.1, ..., encoder.layer.9
    # For more layers, use encoder.layer.[0-9] or encoder.layer.[0-9][0-9]
    fsdp_transforms = {}
    for i in range(24):  # 24 layers in our 1B config
        fsdp_transforms[f"encoder.layer.{i}"] = partial(
            fully_shard,
            mesh=mesh,
            axis_name="fsdp",
        )

    return tp_transforms, fsdp_transforms


def main():
    devices = jax.devices()
    print(f"Number of devices: {len(devices)}")
    print(f"Device type: {devices[0].platform}")

    # Create 2D mesh: tp=2, fsdp=2 (for 4 TPU devices)
    mesh_shape = (2, 2)
    mesh = Mesh(np.array(devices[:4]).reshape(mesh_shape), axis_names=("tp", "fsdp"))
    print(f"Mesh shape: {mesh_shape}, axes: tp={mesh_shape[0]}, fsdp={mesh_shape[1]}")

    # Create model
    config = create_bert_1b_config()
    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")

    key = jax.random.PRNGKey(42)
    print("\nCreating model...")
    model = BertModel(config, key=key)

    # Create transform dictionaries
    print("\nCreating transform dictionaries...")
    tp_transforms, fsdp_transforms = create_transform_dict(mesh)
    print(f"  TP patterns: {len(tp_transforms)}")
    print(f"  FSDP patterns: {len(fsdp_transforms)}")

    # Benchmark apply_transforms
    print("\n" + "=" * 60)
    print("Benchmarking apply_transforms with TP + FSDP")
    print("=" * 60)

    # Warmup
    print("\nWarmup run...")
    temp = apply_transforms(model, tp_transforms)
    _ = apply_transforms(temp, fsdp_transforms)

    # Actual timing
    print("\nTiming apply_transforms...")
    num_runs = 5
    times = []

    for i in range(num_runs):
        # Recreate model each time to get fresh state
        model = BertModel(config, key=key)

        start = time.perf_counter()
        # First apply TP to individual layers
        model = apply_transforms(model, tp_transforms)
        # Then apply FSDP to whole layers
        transformed_model = apply_transforms(model, fsdp_transforms)
        # Block until complete
        jax.block_until_ready(jax.tree.leaves(transformed_model))
        end = time.perf_counter()

        elapsed = end - start
        times.append(elapsed)
        print(f"  Run {i + 1}/{num_runs}: {elapsed:.4f}s")

    print(f"\nResults:")
    print(f"  Mean: {sum(times) / len(times):.4f}s")
    print(f"  Min: {min(times):.4f}s")
    print(f"  Max: {max(times):.4f}s")
    print(f"  Std: {jnp.std(jnp.array(times)):.4f}s")

    # Unbox and verify
    print("\nUnboxing parameters...")
    start = time.perf_counter()
    final_model = unbox_params(transformed_model, mesh)
    jax.block_until_ready(jax.tree.leaves(final_model))
    end = time.perf_counter()
    print(f"  Unbox time: {end - start:.4f}s")

    # Verify some shardings
    print("\nVerifying shardings...")
    first_layer = final_model.encoder.layer[0]
    print(
        f"  Q weight sharding: {first_layer.attention.self.query.weight.sharding.spec}"
    )
    print(
        f"  output weight sharding: {first_layer.attention.output.dense.weight.sharding.spec}"
    )
    print(f"  FFN w1 sharding: {first_layer.intermediate.dense.weight.sharding.spec}")
    print(f"  FFN w2 sharding: {first_layer.output.dense.weight.sharding.spec}")

    print("\nDone!")


if __name__ == "__main__":
    import equinox as eqx

    main()
