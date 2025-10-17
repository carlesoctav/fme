"""
Benchmark apply_transforms (TP + FSDP) on 10B BERT model.

Strategy: Initialize on CPU, transfer to TPU with sharding, then apply transforms.

Config:
- Model: BERT with ~10B params (4096 hidden, 48 layers)
- Mesh: 2D (tp=2, fsdp=2) on 4 TPU devices
- TP: Q/K/V column parallel, attention output row parallel, FFN w1 column, w2 row
- FSDP: Each BertLayer sharded on fsdp axis

Note: This requires sufficient system RAM (~40GB) for CPU initialization.
"""

import time
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from transformers.models.bert.configuration_bert import BertConfig

from src.distributed.params import fully_shard, unbox_params
from src.filter import apply_transforms
from src.distributed.tp import column_parallel, row_parallel
from src.models.bert import BertModel


def create_bert_10b_config():
    """Create a ~10B parameter BERT config."""
    return BertConfig(
        hidden_size=4096,
        num_hidden_layers=48,
        num_attention_heads=32,
        intermediate_size=16384,
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
    fsdp_transforms = {}
    for i in range(num_layers):
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

    # Create model config
    config = create_bert_10b_config()
    print(f"\nModel config:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num heads: {config.num_attention_heads}")
    print(f"  Intermediate size: {config.intermediate_size}")

    # Estimate parameter count
    embed_params = (
        config.vocab_size + config.max_position_embeddings + config.type_vocab_size
    ) * config.hidden_size
    layer_params = (
        4 * config.hidden_size**2 + 2 * config.hidden_size * config.intermediate_size
    ) * config.num_hidden_layers
    total_params = embed_params + layer_params
    print(f"  Estimated params: {total_params / 1e9:.2f}B")
    print(f"  Estimated size (fp32): {total_params * 4 / 1e9:.2f}GB")

    key = jax.random.PRNGKey(42)

    # Step 1: Initialize on CPU
    print("\n[1/4] Initializing model on CPU...")
    print("  (This requires ~40GB RAM and may take a while)")
    cpu_device = jax.devices("cpu")[0]

    start = time.perf_counter()
    with jax.default_device(cpu_device):
        model = BertModel(config, key=key)
        # Force materialization on CPU
        jax.block_until_ready(jax.tree.leaves(model))
    end = time.perf_counter()

    cpu_init_time = end - start
    print(f"  CPU init time: {cpu_init_time:.2f}s")

    # Step 2: Transfer to TPU with 2D sharding to maximize distribution
    print("\n[2/4] Transferring to TPU with 2D sharding...")
    print("  (Sharding all weights on both tp and fsdp axes to fit in memory)")

    def transfer_to_tpu(x):
        if not eqx.is_array(x):
            return x

        # Shard all 2D arrays on both axes to maximize distribution
        if len(x.shape) == 2:
            sharding = NamedSharding(mesh, P("tp", "fsdp"))
        elif len(x.shape) == 1:
            # 1D arrays on fsdp axis
            sharding = NamedSharding(mesh, P("fsdp"))
        else:
            # Other arrays replicated
            sharding = NamedSharding(mesh, P())

        return jax.device_put(x, sharding)

    start = time.perf_counter()
    model = jax.tree.map(transfer_to_tpu, model)
    jax.block_until_ready(jax.tree.leaves(model))
    end = time.perf_counter()

    transfer_time = end - start
    print(f"  Transfer time: {transfer_time:.2f}s")

    # Step 3: Apply transforms
    print("\n[3/4] Applying TP + FSDP transforms...")
    tp_transforms, fsdp_transforms = create_transform_dict(
        mesh, config.num_hidden_layers
    )
    print(f"  TP patterns: {len(tp_transforms)}")
    print(f"  FSDP patterns: {len(fsdp_transforms)}")

    start = time.perf_counter()
    model = apply_transforms(model, tp_transforms)
    transformed_model = apply_transforms(model, fsdp_transforms)
    jax.block_until_ready(jax.tree.leaves(transformed_model))
    end = time.perf_counter()

    transform_time = end - start
    print(f"  Transform time: {transform_time:.4f}s")

    # Step 4: Verify and unbox
    print("\n[4/4] Verifying sharding and unboxing...")
    sample_weight = transformed_model.encoder.layer[0].attention.self.query.weight
    if hasattr(sample_weight, "value"):
        print(f"  Before unbox - sharding: {sample_weight.value.sharding}")
    else:
        print(f"  Before unbox - sharding: {sample_weight.sharding}")

    start = time.perf_counter()
    unboxed = unbox_params(transformed_model, mesh)
    jax.block_until_ready(jax.tree.leaves(unboxed))
    end = time.perf_counter()

    unbox_time = end - start
    print(f"  Unbox time: {unbox_time:.4f}s")

    sample_after = unboxed.encoder.layer[0].attention.self.query.weight
    print(f"  After unbox - sharding: {sample_after.sharding}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"CPU init:     {cpu_init_time:.2f}s")
    print(f"Transfer:     {transfer_time:.2f}s")
    print(f"Transforms:   {transform_time:.4f}s")
    print(f"Unbox:        {unbox_time:.4f}s")
    print(
        f"Total:        {cpu_init_time + transfer_time + transform_time + unbox_time:.2f}s"
    )
    print("\nDone!")


if __name__ == "__main__":
    main()
