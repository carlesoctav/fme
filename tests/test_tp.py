import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from src.nn import Linear
from src.distributed.tp import column_parallel, row_parallel
from src.distributed.params import unbox_params


def test_column_parallel():
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("tp",))

    key = jax.random.PRNGKey(42)
    linear = Linear(in_features=256, out_features=512, key=key)

    linear_col = column_parallel(
        linear, axis_name="tp", mesh=mesh, outputs_layout=P(None, "tp")
    )
    linear_col = unbox_params(linear_col, mesh)

    x = jax.random.normal(jax.random.PRNGKey(0), (8, 256))

    with mesh:
        output = linear_col(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sharding: {output.sharding}")
    print(f"Weight sharding: {linear_col.weight.sharding}")

    expected_weight_spec = P("tp", None)
    assert linear_col.weight.sharding.spec == expected_weight_spec, (
        f"Expected weight spec {expected_weight_spec}, got {linear_col.weight.sharding.spec}"
    )

    expected_output_spec = P(None, "tp")
    assert output.sharding.spec == expected_output_spec, (
        f"Expected output spec {expected_output_spec}, got {output.sharding.spec}"
    )

    print("✓ Column parallel test passed")


def test_row_parallel():
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("tp",))

    key = jax.random.PRNGKey(42)
    linear = Linear(in_features=256, out_features=512, key=key)

    linear_row = row_parallel(linear, axis_name="tp", mesh=mesh, outputs_layout=P())
    linear_row = unbox_params(linear_row, mesh)

    x = jax.random.normal(jax.random.PRNGKey(0), (8, 256))

    with mesh:
        output = linear_row(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sharding: {output.sharding}")
    print(f"Weight sharding: {linear_row.weight.sharding}")

    expected_weight_spec = P(None, "tp")
    assert linear_row.weight.sharding.spec == expected_weight_spec, (
        f"Expected weight spec {expected_weight_spec}, got {linear_row.weight.sharding.spec}"
    )

    expected_output_spec = P()
    assert output.sharding.spec == expected_output_spec, (
        f"Expected output spec {expected_output_spec}, got {output.sharding.spec}"
    )

    print("✓ Row parallel test passed")


if __name__ == "__main__":
    test_column_parallel()
    test_row_parallel()
