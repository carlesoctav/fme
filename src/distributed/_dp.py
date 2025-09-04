import equinox as eqx
import jax.tree_util as jtu
from equinox import is_array
from jax import P


def get_dp_partition_spec(tree):
    def _f(x):
        if is_array(x):
            return P()

    return jtu.tree_map(_f, eqx.filter(tree, is_array))
