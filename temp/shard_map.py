import equinox as eqx
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import jax
import jax.numpy as jnp
from collections.abc import Callable
from typing import Any
from src.distributed._utils import simulate_CPU_devices

simulate_CPU_devices()

class A(eqx.Module):
    random_str: str

    def __call__(self, x):
        if self.random_str == "a":
            return x + 1
        return x + 2

def f(a, x):
    return a(x)

a = A("a")

mesh = jax.make_mesh((8,), ("batch",))
spec = P("batch")
sharding = jax.NamedSharding(mesh, spec)
new_a, new_x = eqx.filter_shard((a, jnp.ones(8)), sharding)

class filter_shard_map(eqx.Module):
    f: Callable
    mesh: Any
    in_specs: Any
    out_specs: Any
    check_rep: bool

    def __call__(self, *args):
        arr, static = eqx.partition(args, eqx.is_array)

        def _f(_args):
            a = eqx.combine(_args, static)
            return self.f(*a)
        
        return shard_map(_f, self.mesh, in_specs=self.in_specs, out_specs=self.out_specs, check_rep=self.check_rep)(arr)

# this errors
sharded_fun = shard_map(f, mesh, in_specs=spec, out_specs=spec, check_rep=False)
# this works
# sharded_fun = eqx.filter_jit(filter_shard_map(f, mesh, in_specs=spec, out_specs=spec, check_rep=False))
_ = sharded_fun(a, new_x).block_until_ready()
