from functools import partial
import os

# Force CPU to avoid TPU/GPU probing noise in local runs
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from jax.sharding import Mesh

from dataclasses import dataclass

from src._filter import apply_transforms
from src.models.bert.modeling_bert import BertModel
from src.distributed import (
    tensor_parallel,
    get_partition_spec,
    as_column_parallel,
    as_row_parallel,
)


def _contains_axis(pspec_entry, axis: str) -> bool:
    if pspec_entry is None:
        return False
    if isinstance(pspec_entry, str):
        return pspec_entry == axis
    if isinstance(pspec_entry, tuple):
        return axis in pspec_entry
    return False


def pspec_has_axis(pspec, axis: str) -> bool:
    if pspec is None:
        return False
    if isinstance(pspec, tuple):
        return any(_contains_axis(p, axis) for p in pspec)
    if isinstance(pspec, str):
        return pspec == axis
    return False


@dataclass
class SmallConfig:
    vocab_size: int = 30522
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 8
    intermediate_size: int = 256
    max_position_embeddings: int = 96
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0


def main():
    cfg = SmallConfig()

    key = jax.random.key(0)
    model = BertModel(cfg, key=key)

    # Single-device mesh for annotation-only validation
    devices = np.array([jax.devices()[0]])
    mesh = Mesh(devices, ("tp",))

    # Tensor-parallel sharding plan (Megatron-style common practice):
    # - Embedding: shard vocab (dim 0)
    # - Attention Q, K, V: column-parallel (dim 0)
    # - Attention output dense: row-parallel (dim 1)
    # - MLP first dense: column-parallel (dim 0)
    # - MLP second dense: row-parallel (dim 1)
    col_tp = partial(
        tensor_parallel,
        mesh=mesh,
        axis_name="tp",
        dim_to_sharded=0,
        min_weight_size=2**10,
    )
    row_tp = partial(
        tensor_parallel,
        mesh=mesh,
        axis_name="tp",
        dim_to_sharded=1,
        min_weight_size=2**10,
    )

    # First pass: annotate weights with TP sharding
    col_tp = partial(
        tensor_parallel,
        mesh=mesh,
        axis_name="tp",
        dim_to_sharded=0,
        min_weight_size=2**10,
    )
    row_tp = partial(
        tensor_parallel,
        mesh=mesh,
        axis_name="tp",
        dim_to_sharded=1,
        min_weight_size=2**10,
    )

    sharded = apply_transforms(
        model,
        {
            # Embedding
            "embeddings.word_embeddings": col_tp,
            # Self-attention projections
            "encoder.layer.*.attention.self.query": col_tp,
            "encoder.layer.*.attention.self.key": col_tp,
            "encoder.layer.*.attention.self.value": col_tp,
            # Self-attention output projection
            "encoder.layer.*.attention.output.dense": row_tp,
            # MLP
            "encoder.layer.*.intermediate.dense": col_tp,
            "encoder.layer.*.output.dense": row_tp,
        },
    )

    # Second pass: add activation layout constraints by dynamically subclassing
    # the selected modules; preserves original types while overriding __call__
    # to apply with_sharding_constraint.
    sharded = apply_transforms(
        sharded,
        {
            # Column-parallel: inputs replicated, outputs sharded on last dim
            "encoder.layer.*.attention.self.query": lambda m: as_column_parallel(m, "tp"),
            "encoder.layer.*.attention.self.key": lambda m: as_column_parallel(m, "tp"),
            "encoder.layer.*.attention.self.value": lambda m: as_column_parallel(m, "tp"),
            "encoder.layer.*.intermediate.dense": lambda m: as_column_parallel(m, "tp"),

            # Row-parallel: inputs sharded on last dim; outputs sharded to
            # encourage reduce-scatter across the boundary.
            "encoder.layer.*.attention.output.dense": lambda m: as_row_parallel(m, "tp", out_sharded=True),
            "encoder.layer.*.output.dense": lambda m: as_row_parallel(m, "tp", out_sharded=True),
        },
    )

    part_spec = get_partition_spec(sharded)
    print(f"DEBUG TP part-spec: {part_spec}")

    # Validate a few canonical parameters got the 'tp' axis
    we = sharded.embeddings.word_embeddings.weight
    assert pspec_has_axis(we.pspec, "tp"), "Expected word_embeddings.weight to be sharded on tp"

    q_w = sharded.encoder.layer[0].attention.self.query.weight
    assert pspec_has_axis(q_w.pspec, "tp"), "Expected attention.query.weight to be sharded on tp (column)"

    attn_out_w = sharded.encoder.layer[0].attention.output.dense.weight
    assert pspec_has_axis(
        attn_out_w.pspec, "tp"
    ), "Expected attention.output.dense.weight to be sharded on tp (row)"

    inter_w = sharded.encoder.layer[0].intermediate.dense.weight
    assert pspec_has_axis(
        inter_w.pspec, "tp"
    ), "Expected intermediate.dense.weight to be sharded on tp (column)"

    out_w = sharded.encoder.layer[0].output.dense.weight
    assert pspec_has_axis(out_w.pspec, "tp"), "Expected output.dense.weight to be sharded on tp (row)"

    # LayerNorm params should remain unsharded due to small size threshold
    ln_w = sharded.embeddings.LayerNorm.weight
    if ln_w is not None:
        assert not pspec_has_axis(
            ln_w.pspec, "tp"
        ), "LayerNorm.weight should remain unsharded in TP"

    print("TP sharding + IO constraints (subclass) applied and validated on BertModel.")


if __name__ == "__main__":
    main()
