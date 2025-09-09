from functools import partial

import equinox as eqx
import jax
from jax import P, shard_map

from src import apply_transforms, nn
from src.distributed import column_parallel, row_parallel, simulate_CPU_devices, get_partition_spec, prepare_input, prepare_output


class SuperLinear(eqx.Module):
    linear1: nn.Linear
    linear2: nn.Linear

    def __init__(self, flow: list[int], *, key): 
        key1, key2 = jax.random.split(key, 2)
        self.linear1 = nn.Linear(flow[0], flow[1], key=key1)
        self.linear2 = nn.Linear(flow[1], flow[2], key=key2)

    def __call__(self, x):
        return self.linear2(self.linear1(x))


simulate_CPU_devices()

abstract_super_linear = SuperLinear([8, 16, 32], key = jax.random.key(1))

devices = jax.devices()
print(f"DEBUGPRINT[244]: test_my_great_tp_plan.py:28: devices={devices}")
mesh = jax.make_mesh((2,4), ("data", "tp" ), devices = devices)
print(f"DEBUGPRINT[243]: test_my_great_tp_plan.py:28: mesh={mesh}")

plan = {
    "linear1": partial(
        prepare_input,
        inputs_layout = (P("data","tp")),
    ),
    "linear2": partial(
        prepare_output,
        outputs_layout = (P(None, "data"), )
    )
}

print(f"DEBUGPRINT[250]: test_my_great_tp_plan.py:50: abstract_super_linear={abstract_super_linear}")
abstract_super_linear = apply_transforms(abstract_super_linear, plan)
pspec = get_partition_spec(abstract_super_linear)
print(f"DEBUGPRINT[249]: test_my_great_tp_plan.py:52: pspec={pspec}")

@jax.jit
def init():
    new_abstract_module = jax.lax.with_sharding_constraint(abstract_super_linear, pspec)
    return new_abstract_module


with mesh:
    module = init()

x = jax.random.normal(jax.random.key(123), (12, 8 ))

jax.debug.visualize_array_sharding(module.linear1.weight.value)
jax.debug.visualize_array_sharding(module.linear2.weight.value)


def f(model, x):
    results = model(x)
    return results


with mesh:
    res = eqx.filter_jit(f)(module ,x)


jax.debug.visualize_array_sharding(res)


