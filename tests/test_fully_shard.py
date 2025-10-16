import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from src.distributed.array import ArrayWithSharding
from src.distributed.params import fully_shard, unbox_params
from src.distributed.utils import simulate_CPU_devices
from src.models.bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig


simulate_CPU_devices(device_count=8)


def test_fully_shard_bert():
    config = BertConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=512,
    )

    key = jax.random.PRNGKey(42)
    model = BertModel(config=config, key=key)

    devices = jax.devices()[:8]
    mesh = Mesh(devices, axis_names=("fsdp",))

    sharded_model = fully_shard(model, mesh, axis_name="fsdp", min_weight_size=1024)

    unboxed_model = unbox_params(sharded_model, mesh)

    def check_sharding(path, leaf):
        if isinstance(leaf, jax.Array):
            sharding = leaf.sharding
            spec = sharding.spec if hasattr(sharding, "spec") else None

            if leaf.size >= 1024:
                shape = leaf.shape
                largest_dim_idx = jnp.argmax(jnp.array(shape))
                largest_dim = shape[largest_dim_idx]

                if largest_dim % 8 == 0:
                    assert spec is not None, (
                        f"Expected sharding for {path} with shape {shape}"
                    )
                    assert len(spec) == len(shape), (
                        f"Spec dimension mismatch for {path}"
                    )

                    has_fsdp_sharding = any(
                        s is not None
                        and ("fsdp" in s if isinstance(s, tuple) else s == "fsdp")
                        for s in spec
                    )
                    assert has_fsdp_sharding, (
                        f"Expected 'fsdp' sharding for {path} with shape {shape}, got {spec}"
                    )

    jax.tree_util.tree_map_with_path(check_sharding, unboxed_model)
    print("✓ All large parameters correctly sharded on 'fsdp' axis")


if __name__ == "__main__":
    test_fully_shard_bert()
