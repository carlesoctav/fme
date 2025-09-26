import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import jax
import equinox as eqx
import optax
from jax import P
from jax.sharding import Mesh

from transformers.models.bert.configuration_bert import BertConfig

from src.models.bert.modeling_bert import BertModel
from src._training import make_module_opts
from src.distributed import (
    column_parallel,
    row_parallel,
    fully_shard,
    simulate_CPU_devices,
)

# Ensure we simulate an 8-CPU environment for sharding/mesh
simulate_CPU_devices(8)


def _make_mesh(*axes: str) -> Mesh:
    # Build a mesh that works regardless of device count:
    # put all devices on the first axis, and 1 on the rest.
    devs = np.array(jax.devices())
    n = devs.size if devs.size > 0 else 1
    if not axes:
        axes = ("tp",)
    shape = (n,) + (1,) * (len(axes) - 1)
    arr = devs.reshape(shape)
    return Mesh(arr, axes)


def _bert_tp_plan(mesh: Mesh, axis_name: str):
    """
    Tensor-parallel plan using PartitionSpec-based IO constraints.
    Mirrors a torchtitan-style plan with col/row parallel and explicit P specs.
    """
    return {
        # Embedding: row-parallel over vocab; replicate inputs, shard outputs on last dim
        "embeddings.word_embeddings": lambda m: row_parallel(
            m,
            axis_name=axis_name,
            mesh=mesh,
            inputs_layout=P(),
            outputs_layout=P(None, axis_name),
        ),

        # Self-attention projections: column-parallel; replicate inputs, shard outputs on last dim
        "encoder.layer.*.self.query": lambda m: column_parallel(
            m,
            axis_name=axis_name,
            mesh=mesh,
            inputs_layout=P(),
            outputs_layout=P(None, axis_name),
        ),
        "encoder.layer.*.self.key": lambda m: column_parallel(
            m,
            axis_name=axis_name,
            mesh=mesh,
            inputs_layout=P(),
            outputs_layout=P(None, axis_name),
        ),
        "encoder.layer.*.self.value": lambda m: column_parallel(
            m,
            axis_name=axis_name,
            mesh=mesh,
            inputs_layout=P(),
            outputs_layout=P(None, axis_name),
        ),

        # Output projection: row-parallel; replicate inputs, shard outputs on last dim
        "encoder.layer.*.output.dense": lambda m: row_parallel(
            m,
            axis_name=axis_name,
            mesh=mesh,
            inputs_layout=P(),
            outputs_layout=P(None, axis_name),
        ),
    }


def _bert_fsdp_plan(mesh: Mesh, axis_name: str):
    # Apply fully sharded parameter policy as a broad fallback
    return {
        "*": lambda m: fully_shard(m, mesh=mesh, axis_name=axis_name)
    }


def test_single_module_tp_then_fsdp():
    cfg = BertConfig(
        vocab_size = 30522,
        hidden_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 8,
        intermediate_size = 256,
        max_position_embeddings = 96,
        type_vocab_size = 2,
        layer_norm_eps = 1e-12,
        hidden_dropout_prob = 0.0,
        attention_probs_dropout_prob = 0.0,
    )  # default config
    key = jax.random.PRNGKey(0)
    abs_model = eqx.filter_eval_shape(BertModel, cfg, key=key)
    print(f"DEBUGPRINT[294]: test_training.py:108: abs_model={abs_model}")

    # Mesh with both tp and fsdp axes (both size 1 so it works anywhere)
    # Prefer a 2x4 mesh over 8 host CPUs if available; fallback to 1x1
    devs = np.array(jax.devices())
    try:
        mesh = jax.make_mesh((2, 4), ("tp", "fsdp"), devices=devs)
    except Exception:
        mesh = _make_mesh("tp", "fsdp")

    tp_plan = _bert_tp_plan(mesh, axis_name="tp")
    fsdp_plan = _bert_fsdp_plan(mesh, axis_name="fsdp")

    grad_tx = optax.adam(1e-3)
    new_model, opt = make_module_opts(
        abs_model,
        grad_tx,
        mesh,
        wrt=eqx.is_inexact_array,
        parallelism_plans=[tp_plan, fsdp_plan],
        key=key,
    )

    print(f"DEBUGPRINT[299]: test_training.py:123: opt={opt}")
    print(f"DEBUGPRINT[298]: test_training.py:123: setup_module_opts={make_module_opts}")

    print(f"DEBUGPRINT[293]: test_training.py:111: new_model={new_model}")
    # Basic structural checks on annotated pspecs
    q_w = new_model.encoder.layer[0].attention.self.query.weight
    o_w = new_model.encoder.layer[0].output.dense.weight


    print(f"DEBUGPRINT[296]: test_training.py:139: q_w.sharding.pspec={q_w.value.sharding.spec}")

    # Expect TP axis annotated on query weight (dim 0 for Linear)
    assert isinstance(q_w.pspec, tuple)
    assert any(
        (e == "tp") or (isinstance(e, tuple) and "tp" in e)
        for e in q_w.pspec
    )

    # Expect TP axis annotated on output dense (row-parallel -> dim 1)
    assert isinstance(o_w.pspec, tuple)
    assert any(
        (e == "tp") or (isinstance(e, tuple) and "tp" in e)
        for e in o_w.pspec
    )


def test_two_modules_with_distinct_tp_fsdp_sequences():
    cfg = BertConfig(
        vocab_size = 30522,
        hidden_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 8,
        intermediate_size = 256,
        max_position_embeddings = 96,
        type_vocab_size = 2,
        layer_norm_eps = 1e-12,
        hidden_dropout_prob = 0.0,
        attention_probs_dropout_prob = 0.0,
    )  # default config
    key = jax.random.PRNGKey(0)
    mkey1, mkey2, tkey = jax.random.split(key, 3)
    # Treat these as Generator and Discriminator for the scenario
    generator = BertModel(cfg, key=mkey1)
    discriminator = BertModel(cfg, key=mkey2)

    # Mesh with four axes so each module can use its own axis names
    devs = np.array(jax.devices())
    try:
        mesh = jax.make_mesh((2, 2, 2, 2), ("tp", "fsdp", "tp2", "fsdp2"), devices=devs)
    except Exception:
        mesh = _make_mesh("tp", "fsdp", "tp2", "fsdp2")

    tp_a = _bert_tp_plan(mesh, axis_name="tp")
    fsdp_a = _bert_fsdp_plan(mesh, axis_name="fsdp")
    tp_b = _bert_tp_plan(mesh, axis_name="tp2")
    fsdp_b = _bert_fsdp_plan(mesh, axis_name="fsdp2")

    grad_tx = optax.adam(1e-3)

    g_key, d_key = jax.random.split(tkey)
    new_a, opt_a = make_module_opts(
        generator,
        grad_tx,
        mesh,
        wrt=eqx.is_inexact_array,
        parallelism_plans=[tp_a, fsdp_a],
        key=g_key,
    )
    new_b, opt_b = make_module_opts(
        discriminator,
        grad_tx,
        mesh,
        wrt=eqx.is_inexact_array,
        parallelism_plans=[tp_b, fsdp_b],
        key=d_key,
    )

    qa = new_a.encoder.layer[0].attention.self.query.weight
    qb = new_b.encoder.layer[0].attention.self.query.weight

    # Module A should be annotated with 'tp' but not 'tp2'
    assert any((e == "tp") or (isinstance(e, tuple) and "tp" in e) for e in qa.pspec)
    assert not any((e == "tp2") or (isinstance(e, tuple) and "tp2" in e) for e in qa.pspec)

    # Module B should be annotated with 'tp2' but not 'tp'
    assert any((e == "tp2") or (isinstance(e, tuple) and "tp2" in e) for e in qb.pspec)
    assert not any((e == "tp") or (isinstance(e, tuple) and "tp" in e) for e in qb.pspec)


if __name__ == "__main__":
    test_single_module_tp_then_fsdp()
