import dataclasses as dc
import logging
import typing as tp

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
from jax import P, lax
from jax.sharding import Mesh
from jaxtyping import Array

from src.utils import is_in_jit
from src.distributed.array import ArrayWithSharding


def infer_value_sharding(x):
    if isinstance(x, ArrayWithSharding):
        return x.value, x.sharding
    elif isinstance(x, Array):
        return x, (None,) * x.ndim

    return None, None


def is_darray(x: tp.Any) -> bool:
    return isinstance(x, ArrayWithSharding)


def fully_shard(
    module: eqx.Module,
    mesh: Mesh,
    axis_name: str = "fsdp",
    *,
    min_weight_size: int = 2**10,
    strategy: tp.Literal["unsharded", "greatest_size"] = "greatest_size",
):
    if axis_name not in mesh.shape:
        raise ValueError(
            f"axis {axis_name} is not in mesh; mesh contains {tuple(mesh.shape.keys())}"
        )

    axis_size = mesh.shape[axis_name]

    def _effective_div(ps):
        if ps is None:
            return 1
        if isinstance(ps, str):
            if ps not in mesh.shape:
                raise ValueError(
                    f"axis {ps} is not in mesh; mesh contains {tuple(mesh.shape.keys())}"
                )
            return mesh.shape[ps]
        if isinstance(ps, tuple):
            div = 1
            for x in ps:
                if x not in mesh.shape:
                    raise ValueError(
                        f"axis {x} is not in mesh; mesh contains {tuple(mesh.shape.keys())}"
                    )
                div *= mesh.shape[x]
            return div
        return 1

    def _append_axis(ps, name: str):
        if ps is None:
            return name
        if isinstance(ps, str):
            return (ps, name)
        if isinstance(ps, tuple):
            return ps + (name,)
        return name

    def _annotate_pspec(path, leaf: tp.Any):
        if not isinstance(leaf, ArrayWithSharding | Array):
            return leaf

        value, sharding = infer_value_sharding(leaf)
        if value is None:
            return leaf

        if len(sharding) != value.ndim:
            raise ValueError(
                "attribute names on Darray should have the same dimension with the attribute value"
            )

        if any(
            (p == axis_name) or (isinstance(p, tuple) and axis_name in p)
            for p in sharding
        ):
            logging.warning(
                f"Parameter {value.shape} with names {jax.tree_util.keystr(path)} already sharded on axis {axis_name}. the partition spec is {sharding}."
            )
            return ArrayWithSharding(value=value, sharding=sharding)

        if value.size <= min_weight_size:
            logging.info(
                f"Parameter {value.shape} with names {sharding} too small to shard, size {value.size} < {min_weight_size}."
            )
            return ArrayWithSharding(value=value, sharding=sharding)

        shape = value.shape
        divs = tuple(_effective_div(p) for p in sharding)
        eff_shape = tuple(int(s // d) for s, d in zip(shape, divs))

        if strategy == "unsharded":
            for i, s in enumerate(eff_shape):
                if s % axis_size == 0:
                    new_i_pspec = _append_axis(sharding[i], axis_name)
                    new_pspec = sharding[:i] + (new_i_pspec,) + sharding[i + 1 :]
                    return ArrayWithSharding(value=value, sharding=new_pspec)
            logging.warning(
                f"Could not shard {value.shape} with names {sharding} on axis {axis_name}, no suitable axis found"
            )
            return ArrayWithSharding(value=value, sharding=sharding)

        elif strategy == "greatest_size":
            idx = np.argsort(eff_shape)[::-1]
            for i in idx:
                if eff_shape[i] % axis_size == 0:
                    new_i_pspec = _append_axis(sharding[i], axis_name)
                    new_pspec = sharding[:i] + (new_i_pspec,) + sharding[i + 1 :]
                    return ArrayWithSharding(value=value, sharding=new_pspec)
            logging.warning(
                f"Could not shard {value.shape} with names {sharding} on axis {axis_name}, no suitable axis found"
            )
            return ArrayWithSharding(value=value, sharding=sharding)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return jtu.tree_map_with_path(_annotate_pspec, module, is_leaf=is_darray)


def tensor_parallel(
    module: eqx.Module,
    mesh: Mesh,
    axis_name: str,
    tensor_dim_to_sharded: int = -1,
    *,
    min_weight_size: int = 0,
    skip_on_dim_mismatch: bool = True,
):
    if axis_name not in mesh.shape:
        raise ValueError(
            f"axis {axis_name} is not in mesh; mesh contains {tuple(mesh.shape.keys())}"
        )

    axis_size = mesh.shape[axis_name]

    def _effective_div(ps):
        if ps is None:
            return 1
        if isinstance(ps, str):
            if ps not in mesh.shape:
                raise ValueError(
                    f"axis {ps} is not in mesh; mesh contains {tuple(mesh.shape.keys())}"
                )
            return mesh.shape[ps]
        if isinstance(ps, tuple):
            div = 1
            for x in ps:
                if x not in mesh.shape:
                    raise ValueError(
                        f"axis {x} is not in mesh; mesh contains {tuple(mesh.shape.keys())}"
                    )
                div *= mesh.shape[x]
            return div
        return 1

    def _append_axis(ps, name: str):
        if ps is None:
            return name
        if isinstance(ps, str):
            return (ps, name)
        if isinstance(ps, tuple):
            return ps + (name,)
        return name

    def _annotate_pspec(path, leaf: tp.Any):
        if not isinstance(leaf, ArrayWithSharding | Array):
            return leaf

        value, sharding = infer_value_sharding(leaf)
        if value is None:
            return leaf

        if len(sharding) != value.ndim:
            raise ValueError(
                "attribute names on Darray should have the same dimension with the attribute value"
            )

        if any(
            (p == axis_name) or (isinstance(p, tuple) and axis_name in p)
            for p in sharding
        ):
            logging.warning(
                f"Parameter {value.shape} with names {jax.tree_util.keystr(path)} already sharded on axis {axis_name}. the partition spec is {sharding}."
            )
            return ArrayWithSharding(value=value, sharding=sharding)

        ndim = value.ndim
        dim = (
            tensor_dim_to_sharded
            if tensor_dim_to_sharded >= 0
            else ndim + tensor_dim_to_sharded
        )
        if dim < 0 or dim >= ndim:
            if skip_on_dim_mismatch:
                logging.info(
                    f"Skip sharding: dim {tensor_dim_to_sharded} out of range for shape {value.shape}"
                )
                return ArrayWithSharding(value=value, sharding=sharding)
            else:
                raise ValueError(
                    f"dim_to_sharded {tensor_dim_to_sharded} out of range for value.ndim={ndim}"
                )

        shape = value.shape
        divs = tuple(_effective_div(p) for p in sharding)
        eff_shape = tuple(int(s // d) for s, d in zip(shape, divs))

        if value.size < min_weight_size:
            logging.info(
                f"Skip sharding: small array {value.shape}, size {value.size} < {min_weight_size}"
            )
            return ArrayWithSharding(value=value, sharding=sharding)

        if eff_shape[dim] % axis_size == 0:
            new_i_pspec = _append_axis(sharding[dim], axis_name)
            new_pspec = sharding[:dim] + (new_i_pspec,) + sharding[dim + 1 :]
            return ArrayWithSharding(value=value, sharding=new_pspec)
        else:
            logging.warning(
                f"Could not shard {value.shape} with names {jtu.keystr(path)} on axis {axis_name} for dim {dim}; "
                f"effective size {eff_shape[dim]} not divisible by {axis_size}"
            )
            return ArrayWithSharding(value=value, sharding=sharding)

    return jtu.tree_map_with_path(_annotate_pspec, module, is_leaf=is_darray)



def unbox_get_partition_spec(tree):
    def maybe_has_shape(leaf): # already include array
        if hasattr(leaf, "shape"):
            return P()

    def f(leaf):
        if isinstance(leaf, ArrayWithSharding):
            return P(*leaf.sharding) if isinstance(leaf.sharding, tuple) else P(leaf.sharding)
        return maybe_has_shape(leaf)
        
    return jtu.tree_map(f, tree, is_leaf = is_darray)


def unbox_params(module: eqx.Module) -> eqx.Module:
    def is_leaf(x):
        return isinstance(x, ArrayWithSharding) or isinstance(x, jax.Array)

    def _unbox(leaf):
        if not isinstance(leaf, ArrayWithSharding):
            if isinstance(leaf, jax.Array):
                pspec = P()
                return lax.with_sharding_constraint(leaf, pspec)
            return leaf

        value = leaf.value
        if value is None:
            return None

        if leaf.sharding is None:
            pspec = P()
        else:
            pspec = (
                P(*leaf.sharding)
                if isinstance(leaf.sharding, tuple)
                else P(leaf.sharding)
            )

            return lax.with_sharding_constraint(value, pspec)


    return jtu.tree_map(_unbox, module, is_leaf=is_leaf)
