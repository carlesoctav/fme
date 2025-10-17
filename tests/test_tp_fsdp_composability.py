import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from src.nn import Linear
from src.distributed.tp import column_parallel, row_parallel
from src.distributed.params import fully_shard, unbox_params


def test_tp_then_fsdp():
    """Test applying TP first, then FSDP on a 2D mesh."""
    devices = jax.devices()

    if len(devices) < 2:
        print("⚠ Skipping test: need at least 2 devices")
        return

    # Create 2D mesh: (fsdp=2, tp=2) if we have 4+ devices, else (fsdp=1, tp=remaining)
    import numpy as np

    if len(devices) >= 4:
        devices = np.array(devices[:4]).reshape(2, 2)
        mesh = Mesh(devices, axis_names=("fsdp", "tp"))
    else:
        devices = np.array(devices).reshape(1, len(devices))
        mesh = Mesh(devices, axis_names=("fsdp", "tp"))

    key = jax.random.PRNGKey(42)
    linear = Linear(in_features=256, out_features=512, key=key)

    # Apply TP first (column parallel on tp axis)
    linear = column_parallel(linear, axis_name="tp", mesh=mesh)

    # Then apply FSDP (shard on fsdp axis)
    linear = fully_shard(linear, mesh=mesh, axis_name="fsdp")

    # Unbox to get actual sharded arrays
    linear = unbox_params(linear, mesh)

    x = jax.random.normal(jax.random.PRNGKey(0), (8, 256))

    with mesh:
        output = linear(x)

    print(f"Mesh shape: {mesh.shape}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight sharding: {linear.weight.sharding}")
    print(f"Output sharding: {output.sharding}")

    # Weight should be sharded on both tp (dim 0) and fsdp (dim 1)
    weight_spec = linear.weight.sharding.spec
    print(f"Weight spec: {weight_spec}")

    # Check that weight is sharded on tp axis (dim 0) and fsdp axis (dim 1)
    assert "tp" in str(weight_spec[0]), f"Expected 'tp' in dim 0, got {weight_spec}"
    assert "fsdp" in str(weight_spec[1]), f"Expected 'fsdp' in dim 1, got {weight_spec}"

    print("✓ TP + FSDP composability test passed")


def test_fsdp_then_tp():
    """Test applying FSDP first, then TP on a 2D mesh."""
    devices = jax.devices()

    if len(devices) < 2:
        print("⚠ Skipping test: need at least 2 devices")
        return

    # Create 2D mesh
    import numpy as np

    if len(devices) >= 4:
        devices = np.array(devices[:4]).reshape(2, 2)
        mesh = Mesh(devices, axis_names=("fsdp", "tp"))
    else:
        devices = np.array(devices).reshape(1, len(devices))
        mesh = Mesh(devices, axis_names=("fsdp", "tp"))

    key = jax.random.PRNGKey(42)
    linear = Linear(in_features=256, out_features=512, key=key)

    # Apply FSDP first
    linear = fully_shard(linear, mesh=mesh, axis_name="fsdp")

    # Then apply TP (column parallel on tp axis)
    linear = column_parallel(linear, axis_name="tp", mesh=mesh)

    # Unbox to get actual sharded arrays
    linear = unbox_params(linear, mesh)

    x = jax.random.normal(jax.random.PRNGKey(0), (8, 256))

    with mesh:
        output = linear(x)

    print(f"\nMesh shape: {mesh.shape}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight sharding: {linear.weight.sharding}")
    print(f"Output sharding: {output.sharding}")

    # Weight should be sharded on both fsdp and tp axes
    weight_spec = linear.weight.sharding.spec
    print(f"Weight spec: {weight_spec}")

    # When applying FSDP then TP, both get combined on dim 0: ('fsdp', 'tp')
    # This is because FSDP shards the largest dimension (dim 0 with size 512)
    # Then TP also wants to shard dim 0, so they get combined
    assert "tp" in str(weight_spec[0]), f"Expected 'tp' in dim 0, got {weight_spec}"
    assert "fsdp" in str(weight_spec[0]), f"Expected 'fsdp' in dim 0, got {weight_spec}"

    print("✓ FSDP + TP composability test passed")


if __name__ == "__main__":
    test_tp_then_fsdp()
    test_fsdp_then_tp()
