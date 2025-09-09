import jax
from flax import nnx
from jax import tree_util as jtu
from jax.sharding import AxisType 

from src.distributed._utils import simulate_CPU_devices


simulate_CPU_devices()


class Module(nnx.Module):
    def __init__(
        self, 
        in_feat,
        out_feat,
        *, 
        rngs = None
    ):
        self.linear1 = nnx.Linear(in_feat, out_feat, rngs = rngs)
        self.linear2 = nnx.Linear(in_feat, out_feat, rngs = rngs)


def dp_partition_spec(tree):
    def _f(leaf):
        return jax.P() 

    return jtu.tree_map(_f, tree, is_leaf = lambda x: isinstance(x, nnx.Variable))

def make_module(module, *module_args, mesh, **module_kwargs):
    print(f"DEBUGPRINT[161]: test_nnx.py:27: module_kwargs={module_kwargs}")
    abstract_module = nnx.eval_shape(lambda: module(*module_args, **module_kwargs, rngs = nnx.Rngs(10)))
    print(f"DEBUGPRINT[158]: test_nnx.py:28: abstract_module={abstract_module}")
    partition_spec = dp_partition_spec(nnx.state(abstract_module))
    print(f"DEBUGPRINT[159]: test_nnx.py:30: partition_spec={partition_spec}")

    @nnx.jit
    @nnx.shard_map(out_specs=(nnx.StateSharding(partition_spec)), in_specs =(), mesh = mesh)
    def partition():
        real_module = module(*module_args, **module_kwargs, rngs = nnx.Rngs(10))
        with mesh:
            jax.lax.with_sharding_constraint(nnx.state(real_module), partition_spec)
        return real_module 

    module: nnx.Module = partition()
    def printing(leaf: nnx.Variable):
        jax.debug.visualize_array_sharding(leaf.value)

    jtu.tree_map(printing, module, is_leaf = lambda x: isinstance(x, nnx.Variable))

    print(f"DEBUGPRINT[167]: test_nnx.py:41: module={module}")



def main():
    mesh = jax.make_mesh((8,), ("data",), devices = jax.devices())
    make_module(Module, 10, 10, mesh = mesh)


if __name__ == "__main__":
    main()
