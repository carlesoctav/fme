#!/usr/bin/env python3
"""
Find the largest model we can benchmark by trying progressively larger configs.
"""

import time
import jax
import jax.random as jr
import jax.tree_util as jtu
import optax
from jax.sharding import Mesh
from transformers import BertConfig

from src.models.bert import BertForMaskedLM
from src._training import make_module_opt
from src.distributed import column_parallel, row_parallel


def tp_plan(mesh: Mesh):
    """TP plan for sharding."""
    plan = {
        "*.intermediate.dense": lambda m: column_parallel(m, "tp", mesh),
        "*.output.dense": lambda m: row_parallel(m, "tp", mesh),
        "*.attention.self.query": lambda m: column_parallel(m, "tp", mesh),
        "*.attention.self.key": lambda m: column_parallel(m, "tp", mesh),
        "*.attention.self.value": lambda m: column_parallel(m, "tp", mesh),
        "*.attention.output.dense": lambda m: row_parallel(m, "tp", mesh),
    }
    return plan


def try_model_size(config_name, config, mesh):
    """Try to benchmark a specific model config."""
    print(f"\n{'=' * 80}")
    print(f"Trying: {config_name}")
    print(f"{'=' * 80}")
    
    key = jr.PRNGKey(42)
    
    # Calculate model size
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
    
    print(f"Config: BERT-{config.num_hidden_layers}L-{config.hidden_size}H")
    print(f"Params: {total_params / 1e9:.1f}B")
    print(f"Size: {size_gb:.1f}GB (fp32)")
    print(f"Per-device (TP4): ~{size_gb / 4:.1f}GB")
    
    # Create model on CPU
    cpu_device = jax.devices("cpu")[0]
    
    try:
        print("\nCreating model on CPU...")
        with jax.default_device(cpu_device):
            start_create = time.monotonic()
            model = BertForMaskedLM(config, key=key)
            create_time = time.monotonic() - start_create
        print(f"Model creation: {create_time:.2f}s")
        
    except Exception as e:
        print(f"FAILED at model creation: {e}")
        return None
    
    # Count transformations
    from src._filter import iter_module, _path_to_str
    import fnmatch
    
    plan = tp_plan(mesh)
    matches = 0
    for path, _ in iter_module(model):
        path_str = _path_to_str(path)
        for pattern in plan.keys():
            if fnmatch.fnmatchcase(path_str, pattern):
                matches += 1
                break
    
    print(f"Transformations: {matches}")
    
    grad_tx = optax.adamw(learning_rate=1e-4)
    
    # Try make_module_opt
    try:
        print("\nRunning make_module_opt...")
        with mesh:
            start = time.monotonic()
            sharded_model, opt = make_module_opt(
                model,
                grad_tx,
                mesh=mesh,
                parallelism_plans=plan,
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
        
        print(f"✓ SUCCESS: {first_run_time:.4f}s")
        
        return {
            "name": config_name,
            "params_b": total_params / 1e9,
            "size_gb": size_gb,
            "transforms": matches,
            "time_s": first_run_time,
        }
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return None


def main():
    devices = jax.devices()
    mesh = Mesh(devices[:4], ("tp",))
    
    print("=" * 80)
    print("FINDING MAXIMUM MODEL SIZE")
    print("=" * 80)
    print(f"Mesh: {mesh}")
    
    configs = [
        # Start from known working size
        ("BERT-36L-2048H", BertConfig(
            vocab_size=30522, hidden_size=2048, num_hidden_layers=36,
            num_attention_heads=16, intermediate_size=8192,
            max_position_embeddings=512, type_vocab_size=2,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )),
        
        # Try 48 layers (2.5B params, 9.4GB)
        ("BERT-48L-2048H", BertConfig(
            vocab_size=30522, hidden_size=2048, num_hidden_layers=48,
            num_attention_heads=16, intermediate_size=8192,
            max_position_embeddings=512, type_vocab_size=2,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )),
        
        # Try 60 layers (3.1B params, 11.7GB)
        ("BERT-60L-2048H", BertConfig(
            vocab_size=30522, hidden_size=2048, num_hidden_layers=60,
            num_attention_heads=16, intermediate_size=8192,
            max_position_embeddings=512, type_vocab_size=2,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )),
        
        # Try larger hidden size (3.9B params, 14.5GB)
        ("BERT-48L-2560H", BertConfig(
            vocab_size=30522, hidden_size=2560, num_hidden_layers=48,
            num_attention_heads=20, intermediate_size=10240,
            max_position_embeddings=512, type_vocab_size=2,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )),
        
        # Try 72 layers with 2048 hidden (3.8B params, 14.0GB)
        ("BERT-72L-2048H", BertConfig(
            vocab_size=30522, hidden_size=2048, num_hidden_layers=72,
            num_attention_heads=16, intermediate_size=8192,
            max_position_embeddings=512, type_vocab_size=2,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            _attn_implementation="sdpa",
        )),
    ]
    
    results = []
    for name, config in configs:
        result = try_model_size(name, config, mesh)
        if result:
            results.append(result)
        else:
            print(f"\n⚠ Stopping at {name} - too large")
            break
    
    if results:
        print("\n" + "=" * 80)
        print("SUCCESSFUL BENCHMARKS")
        print("=" * 80)
        print(f"{'Model':<20} {'Params':>10} {'Size':>10} {'Transforms':>12} {'First Run':>12}")
        print("-" * 80)
        for r in results:
            print(f"{r['name']:<20} {r['params_b']:>9.1f}B {r['size_gb']:>9.1f}GB {r['transforms']:>12} {r['time_s']:>11.4f}s")


if __name__ == "__main__":
    main()
