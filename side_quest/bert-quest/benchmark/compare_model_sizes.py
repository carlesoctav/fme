#!/usr/bin/env python3
"""Compare make_module_opt first run time across model sizes."""

import time
import jax
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import optax
from jax.sharding import Mesh
from transformers import BertConfig

from src.models.bert import BertForMaskedLM
from src._training import make_module_opt
from src.distributed import column_parallel, row_parallel


def simple_tp_plan(mesh: Mesh, axis_name: str = "tp"):
    plan = {
        "*.intermediate.dense": lambda m: column_parallel(m, axis_name, mesh),
        "*.output.dense": lambda m: row_parallel(m, axis_name, mesh),
        "*.attention.self.query": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.self.key": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.self.value": lambda m: column_parallel(m, axis_name, mesh),
        "*.attention.output.dense": lambda m: row_parallel(m, axis_name, mesh),
    }
    return plan


def benchmark_config(config_name, config):
    """Benchmark a single config."""
    key = jr.PRNGKey(42)
    devices = jax.devices()
    mesh = Mesh(devices, ("tp",))
    
    # Calculate size
    params_per_layer = (
        4 * config.hidden_size * config.hidden_size +
        config.hidden_size * config.intermediate_size +
        config.intermediate_size * config.hidden_size +
        2 * 2 * config.hidden_size
    )
    embedding_params = (
        config.vocab_size * config.hidden_size +
        config.max_position_embeddings * config.hidden_size +
        config.type_vocab_size * config.hidden_size
    )
    total_params = params_per_layer * config.num_hidden_layers + embedding_params
    size_gb = total_params * 4 / (1024**3)
    
    # Create model on CPU
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        model = BertForMaskedLM(config, key=key)
    
    # Count transformations
    from src._filter import iter_module, _path_to_str
    import fnmatch
    tp_plan = simple_tp_plan(mesh, axis_name="tp")
    
    matches = 0
    for path, _ in iter_module(model):
        path_str = _path_to_str(path)
        for pattern in tp_plan.keys():
            if fnmatch.fnmatchcase(path_str, pattern):
                matches += 1
                break
    
    grad_tx = optax.adamw(learning_rate=1e-4)
    
    # FIRST RUN
    with mesh:
        start = time.monotonic()
        sharded_model, opt = make_module_opt(
            model,
            grad_tx,
            mesh=mesh,
            parallelism_plans=tp_plan,
            key=key,
        )
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            sharded_model,
        )
        jtu.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
            opt,
        )
        first_run_time = time.monotonic() - start
    
    return {
        "name": config_name,
        "params_b": total_params / 1e9,
        "size_gb": size_gb,
        "transformations": matches,
        "time_s": first_run_time,
    }


def main():
    configs = [
        ("BERT-4L-768H", BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=4,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )),
        ("BERT-12L-1024H", BertConfig(
            vocab_size=30522,
            hidden_size=1024,
            num_hidden_layers=12,
            num_attention_heads=16,
            intermediate_size=4096,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )),
        ("BERT-24L-2048H", BertConfig(
            vocab_size=30522,
            hidden_size=2048,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=8192,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )),
        ("BERT-36L-2048H", BertConfig(
            vocab_size=30522,
            hidden_size=2048,
            num_hidden_layers=36,
            num_attention_heads=16,
            intermediate_size=8192,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )),
    ]
    
    print("=" * 80)
    print("make_module_opt FIRST RUN COMPARISON")
    print("=" * 80)
    
    results = []
    for name, config in configs:
        print(f"\nBenchmarking {name}...")
        result = benchmark_config(name, config)
        results.append(result)
        print(f"  First run: {result['time_s']:.4f}s")
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'Params':>10} {'Size':>10} {'Transforms':>12} {'First Run':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} {r['params_b']:>9.1f}B {r['size_gb']:>9.1f}GB {r['transformations']:>12} {r['time_s']:>11.4f}s")


if __name__ == "__main__":
    main()
