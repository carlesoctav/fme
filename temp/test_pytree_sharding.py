"""Test PyTree sharded initialization."""

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from functools import partial

devices = jax.devices()
mesh = Mesh(np.array(devices[:2]), axis_names=("tp",))

print("Test: PyTree sharded initialization")
print(f"Devices: {len(devices)}")

class SimpleModel(eqx.Module):
    w1: jax.Array
    w2: jax.Array
    bias: jax.Array
    
    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.w1 = jax.random.normal(k1, (1024, 2048))
        self.w2 = jax.random.normal(k2, (2048, 512))
        self.bias = jax.random.normal(k3, (512,))

# Create abstract model to get structure
key = jax.random.PRNGKey(42)
abstract_model = eqx.filter_eval_shape(SimpleModel, key=key)
print(f"Abstract model: {jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else x, abstract_model)}")

# Create out_shardings matching the structure
# w1 and w2 sharded on first axis, bias replicated
def get_sharding(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        if len(x.shape) == 2:
            return NamedSharding(mesh, P("tp", None))
        elif len(x.shape) == 1:
            return NamedSharding(mesh, P())
    return None

out_shardings = jax.tree.map(get_sharding, abstract_model)
print(f"Out shardings: {out_shardings}")

# Now wrap init in jit with out_shardings
@partial(jax.jit, out_shardings=out_shardings)
def sharded_init(key):
    return SimpleModel(key)

print("\nInitializing with sharding...")
model = sharded_init(key)

print(f"w1 shape: {model.w1.shape}, sharding: {model.w1.sharding}")
print(f"w2 shape: {model.w2.shape}, sharding: {model.w2.sharding}")
print(f"bias shape: {model.bias.shape}, sharding: {model.bias.sharding}")
