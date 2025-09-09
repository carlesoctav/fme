from functools import partial
import os

# Force CPU to avoid TPU/GPU probing noise in local runs
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np

from jax.sharding import Mesh

from src._filter import apply_transforms
from src.models.bert.modeling_bert import BertModel
from dataclasses import dataclass
from src.distributed import fully_shard, simulate_CPU_devices, get_partition_spec


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

    devices = np.array([jax.devices()[0]])
    mesh = Mesh(devices, ("fsdp",))

    fsdp_tx = partial(fully_shard, mesh=mesh, axis_name="fsdp", min_weight_size=2**10)
    sharded = apply_transforms(model, {"*": fsdp_tx})
    part_spec = get_partition_spec(sharded)
    print(f"DEBUGPRINT[239]: test_fsdp.py:63: part_spec={part_spec}")

    we = sharded.embeddings.word_embeddings.weight
    assert pspec_has_axis(we.pspec, "fsdp"), "Expected word_embeddings.weight to be sharded on fsdp"

    ln_bias = sharded.embeddings.LayerNorm.bias
    if ln_bias is not None:
        assert not pspec_has_axis(
            ln_bias.pspec, "fsdp"
        ), "LayerNorm.bias should remain unsharded due to min_weight_size"

    q_w = sharded.encoder.layer[0].attention.self.query.weight
    assert pspec_has_axis(q_w.pspec, "fsdp"), "Expected attention.query.weight to be sharded on fsdp"

    q_b = sharded.encoder.layer[0].attention.self.query.bias
    if q_b is not None:
        assert not pspec_has_axis(
            q_b.pspec, "fsdp"
        ), "attention.query.bias should remain unsharded due to min_weight_size"

    print("FSDP sharding transform applied and validated on BertModel.")


if __name__ == "__main__":
    main()
