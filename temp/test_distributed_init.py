"""Test distributed initialization pattern."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial

devices = jax.devices()
mesh = Mesh(np.array(devices[:2]), axis_names=("tp",))

print("Test: Using jax.jit with out_shardings and static shape")
print(f"Devices: {len(devices)}, mesh: {mesh}")

# Define initialization inside jit with out_shardings and static shape
@partial(jax.jit, out_shardings=NamedSharding(mesh, P("tp", None)), static_argnums=(1,))
def init_sharded_weight(key, shape):
    """Initialize weight directly with sharding."""
    return jax.random.normal(key, shape)

key = jax.random.PRNGKey(42)
weight = init_sharded_weight(key, (1024, 2048))
print(f"Weight shape: {weight.shape}")
print(f"Weight sharding: {weight.sharding}")
print(f"Weight is concrete: {not isinstance(weight, jax.core.Tracer)}")

# Now test with a module
print("\nTest with equinox module:")

class SimpleModel(eqx.Module):
    weight: jax.Array
    
    def __init__(self, shape, *, key):
        # Don't initialize here, will do it in sharded_init
        self.weight = jax.ShapeDtypeStruct(shape, jnp.float32)

@partial(jax.jit, out_shardings=eqx.filter_spec(lambda path, x: NamedSharding(mesh, P("tp", None)) if eqx.is_array(x) else None, eqx.filter_eval_shape(SimpleModel, (1024, 2048), key=key)))
def sharded_init(key, shape):
    """Create model with sharded weights."""
    return SimpleModel.__init__.__wrapped__(SimpleModel.__new__(SimpleModel), shape, key=key)

print("This approach is getting complex... let me try a simpler pattern")
