import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

devices = jax.devices()
print(f"DEBUGPRINT[28]: sharding_contraint.py:6: devices={devices}")
mesh = Mesh(devices, axis_names=('tp',))

x = jnp.ones((8, 512))

print('Without mesh context:')
print(f'x.sharding before: {x.sharding}')
print('x sharding visualization:')
jax.debug.visualize_array_sharding(x)

y = jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(None, 'tp')))
print(f'\ny.sharding after constraint (no jit): {y.sharding}')
print('y sharding visualization:')
jax.debug.visualize_array_sharding(y)
