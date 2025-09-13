from functools import partial

import equinox as eqx
import jax
from jax import P, shard_map

from src import apply_transforms, nn
import dataclasses as dc
import jax.tree_util as jtu
from src.distributed import column_parallel, row_parallel, simulate_CPU_devices, get_partition_spec, prepare_input, prepare_output, fully_shard


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

abstract_super_linear = eqx.filter_eval_shape(SuperLinear, [8, 16, 32], key = jax.random.key(1))

devices = jax.devices()
mesh = jax.make_mesh((2,4), ("data", "tp" ), devices = devices)

plan = {
    "linear1": partial(
        column_parallel,
        axis_name = "tp",
        mesh = mesh,
        inputs_layout = P("data","tp"),
        outputs_layout = P("data", "tp"),
    ),
    "linear2": partial(
        prepare_output,
        outputs_layout = (P(None, "data"), )
    )
}


fsdp_plan = {
    "*": partial(fully_shard, mesh = mesh, axis_name = "data", min_weight_size = 0)
}


abstract_super_linear = apply_transforms(abstract_super_linear, plan)
pspec_abstract = get_partition_spec(abstract_super_linear)

# Helper to apply PartitionSpec leaves onto Darray.value leaves

@jax.jit
def init():
    new_module = SuperLinear([8, 16, 32], key = jax.random.key(10))
    new_module = apply_transforms(new_module, plan)
    new_module = fully_shard(new_module, mesh = mesh, axis_name = "data", min_weight_size = 0)
    pspec_inst = get_partition_spec(new_module)
    print(f"DEBUGPRINT[260]: tp_with_fsdp.py:63: pspec_inst={pspec_inst}")
    # new_module = apply_transforms(new_module, fsdp_plan)
    new_module = eqx.filter_shard(new_module, pspec_inst)

    return new_module

# i can do other stuff, like new_abstract_module -> apply_transforms -> partition_spec -> real_module, with_appl


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
