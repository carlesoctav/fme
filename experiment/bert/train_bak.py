from typing import Any, Callable
import equinox as eqx
import jax
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from transformers.models.bert.configuration_bert import BertConfig
from src.models.bert.modeling_bert import BertModel
from optax import adam
from jax import tree_util as jtu
from src.distributed._utils import  simulate_CPU_devices
eqx.filter_value_and_grad()

simulate_CPU_devices()

# Predicates
is_array_runtime = eqx.is_array
is_array_spec = lambda x: isinstance(x, (jax.Array, jax.ShapeDtypeStruct))

def loss_fn(module, *args):
    pass


def annotate_params(tree):
    def _f(leaf):
        return P() if is_array_spec(leaf) else None
    return jtu.tree_map(_f, tree)


def get_partition_spec(tree):
    pass

class filter_shard_map(eqx.Module):
    f: Callable
    mesh: Any
    in_specs: Any
    out_specs: Any
    check_rep: bool

    def __call__(self, *args):
        # Split inputs into arrays vs static (supports eqx.Module and arbitrary pytrees)
        arr_args, static_args = eqx.partition(args, is_array_runtime)

        # Compute abstract output to determine array-output structure and static output.
        abstract_out = eqx.filter_eval_shape(self.f, *eqx.combine(arr_args, static_args))
        out_arr_template, out_static = eqx.partition(abstract_out, is_array_spec)

        print(f"DEBUGPRINT[178]: train.py:44: abstract_out={abstract_out}")
        print(f"DEBUGPRINT[179]: train.py:45: out_arr_template={out_arr_template}")
        print(f"DEBUGPRINT[180]: train.py:45: out_static={out_static}")
        # Define arrays-only function for shard_map.
        # Accept a variable number of array args to handle the no-arg case too.
        def arrays_only_fn(*arr_only_args):
            full_args = eqx.combine(arr_only_args, static_args)
            out_full = self.f(*full_args)
            out_arr, _ = eqx.partition(out_full, is_array_runtime)
            return out_arr

        # Align out_specs with the arrays-only output structure.
        # If self.out_specs was specified for the full output (e.g., Module),
        # extract only the specs corresponding to array leaves.
        try:
            flat_specs, _ = jtu.tree_flatten(self.out_specs)
            print(f"DEBUGPRINT[181]: train.py:62: flat_specs={flat_specs}")
            flat_out, _ = jtu.tree_flatten(abstract_out)
            print(f"DEBUGPRINT[182]: train.py:64: flat_out={flat_out}")
            is_arr_mask = [is_array_spec(x) for x in flat_out]
            print(f"DEBUGPRINT[183]: train.py:66: is_arr_mask={is_arr_mask}")
            arr_specs = [s for s, m in zip(flat_specs, is_arr_mask) if m]
            print(f"DEBUGPRINT[184]: train.py:68: arr_specs={arr_specs}")
            _, arr_treedef = jtu.tree_flatten(out_arr_template)
            out_specs_aligned = jtu.tree_unflatten(arr_treedef, arr_specs)
        except Exception:
            # Assume already arrays-only specs
            out_specs_aligned = self.out_specs

        # If out_specs root is a callable dataclass (e.g., eqx.Module),
        # wrap it in a thunk so jax doesn't treat it as the out_specs function.
        out_specs_param = (lambda: out_specs_aligned)
        print(f"DEBUGPRINT[185]: train.py:78: out_specs_param={out_specs_param}")

        mapped = shard_map(
            arrays_only_fn,
            mesh=self.mesh,
            in_specs=self.in_specs,
            out_specs=out_specs_param,
            check_rep=self.check_rep,
        )(*arr_args)

        # Reattach static parts (e.g., to return eqx.Module, not just arrays)
        return eqx.combine(mapped, out_static)

def make_module(module, *module_args, mesh, mode="dp", debug=False, **module_kwargs) -> eqx.Module:
    if mode != "dp":
        raise ValueError("not supported")

    # Get abstract module to build matching out_specs
    abstract_module = eqx.filter_eval_shape(module, *module_args, **module_kwargs)
    pspec = annotate_params(abstract_module)

    # Define zero-arg initializer to run under shard_map
    def _init_params():
        return module(*module_args, **module_kwargs)

    init_fn = filter_shard_map(
        f=_init_params,
        in_specs=(),  # no mapped inputs
        out_specs=pspec,  # PartitionSpec for arrays inside the returned Module
        mesh=mesh,
        check_rep=False,
    )

    _module = eqx.filter_jit(init_fn)()
    def printing(leaf):
        if isinstance(leaf, jax.Array):
            jax.debug.visualize_array_sharding(leaf)

    jtu.tree_map(printing, _module)
    return _module

def main():
    config = BertConfig(num_hidden_layers=1)
    devices = jax.devices()
    mesh = jax.make_mesh((8,), ("data",), devices=jax.devices())
    print(f"DEBUGPRINT[147]: train.py:41: mesh={mesh}")
    module = make_module(BertModel, config, key=jax.random.key(1), mesh = mesh)
    print(f"DEBUGPRINT[146]: train.py:41: module={module}")
    optimizer = adam(learning_rate = 1e-5,)

if __name__ == "__main__":
    main()
