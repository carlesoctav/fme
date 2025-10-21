"""
Benchmark apply_transforms (TP + FSDP) on 2.5B BERT model.

Config:
- Model: BERT with 2560 hidden, 32 layers, 20 heads, 10240 FFN (~ 2.5B params)
- Mesh: 2D (tp=2, fsdp=2) on 4 TPU devices
- TP: Q/K/V column parallel, attention output row parallel, FFN w1 column, w2 row
- FSDP: Each BertLayer sharded on fsdp axis

Timing results (single run):
- 89s
- 78 s without transformation (single device array)
- 87s so almost free apply_transforms?
- 80s without unbox_params (just transforms)
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding, SingleDeviceSharding
from transformers.models.bert.configuration_bert import BertConfig
import optax

from src.distributed.params import fully_shard, unbox_params
from src.filter import apply_transforms
from src.training_utils import make_module_opts
from src.distributed.tp import column_parallel, row_parallel
from src.models.bert import BertModel
from src.nn import Linear
from src.modeling_utils import Rngs


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
    config = create_bert_large_config()
    key = jax.random.key(10)
    mesh = jax.make_mesh(
        (
            2,
            2,
        ),
        ("fsdp", "tp"),
    )
    tp, fsdp = create_transform_dict(mesh, config.num_hidden_layers)
    grad_tx = optax.sgd(1e-3)

    @jax.jit
    def init_model():
        module = BertModel(config, rngs = Rngs(params = jax.random.key(10)))
        module = unbox_params(module)
        return module


    start = time.monotonic()
    with mesh:
        module = init_model()

    jax.block_until_ready(jax.tree.leaves(module))
    diff = time.monotonic() - start
    print(f"JIT init time: {diff:.2f}s")

    print("\nVerifying shardings...")
    first_layer = module.encoder.layer[0]
    print(
        f"  Q weight sharding: {first_layer.attention.self.query.weight.sharding.spec}"
    )
    print(
        f"  output weight sharding: {first_layer.attention.output.dense.weight.sharding.spec}"
    )
    print(f"  FFN w1 sharding: {first_layer.intermediate.dense.weight.sharding.spec}")
    print(f"  FFN w2 sharding: {first_layer.output.dense.weight.sharding.spec}")


if __name__ == "__main__":
    import equinox as eqx

    main()
