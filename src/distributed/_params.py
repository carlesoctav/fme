import dataclasses as dc
import logging
import typing as tp

import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
from jax import P
from jax.sharding import Mesh

from src import DArray


def is_darray(x: tp.Any) -> bool:
    return isinstance(x, DArray)


def get_partition_spec(module: eqx.Module):
    def _maybe_replicate(x):
        if hasattr(x, "shape"):
            return P()
        else:
            return None

    def _f(leaf):
        if isinstance(leaf, DArray):
            if leaf.pspec is not None:
                return dc.replace(leaf, value = P(*leaf.pspec)) 
            else:
                return dc.replace(leaf, value = _maybe_replicate(leaf.value))
        else:
            return _maybe_replicate(leaf) 

    return jtu.tree_map(_f, module, is_leaf = is_darray)


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
        if not isinstance(leaf, DArray):
            return leaf
        value, pspec = leaf.value, leaf.pspec

        if value is None:
            return leaf

        if pspec is None:
            pspec = (None,) * value.ndim

        if len(pspec) != value.ndim:
            raise ValueError(
                "attribute names on Darray should have the same dimension with the attribute value"
            )

        if any(
            (p == axis_name)
            or (isinstance(p, tuple) and axis_name in p)
            for p in pspec
        ):
            logging.warning(
                f"Parameter {value.shape} with names {jax.tree_util.keystr(path)} already sharded on axis {axis_name}. the partition spec is {pspec}."
            )
            return DArray(value=value, pspec=pspec)

        if value.size <= min_weight_size:
            logging.info(
                f"Parameter {value.shape} with names {pspec} too small to shard, size {value.size} < {min_weight_size}."
            )
            return DArray(value=value, pspec=pspec)

        shape = value.shape
        divs = tuple(_effective_div(p) for p in pspec)
        eff_shape = tuple(int(s // d) for s, d in zip(shape, divs))

        if strategy == "unsharded":
            for i, s in enumerate(eff_shape):
                if s % axis_size == 0:
                    new_i_pspec = _append_axis(pspec[i], axis_name)
                    new_pspec = pspec[:i] + (new_i_pspec,) + pspec[i + 1 :]
                    return DArray(value=value, pspec=new_pspec)
            logging.warning(
                f"Could not shard {value.shape} with names {pspec} on axis {axis_name}, no suitable axis found"
            )
            return DArray(value=value, pspec=pspec)

        elif strategy == "greatest_size":
            idx = np.argsort(eff_shape)[::-1]
            for i in idx:
                if eff_shape[i] % axis_size == 0:
                    new_i_pspec = _append_axis(pspec[i], axis_name)
                    new_pspec = pspec[:i] + (new_i_pspec,) + pspec[i + 1 :]
                    return DArray(value=value, pspec=new_pspec)
            logging.warning(
                f"Could not shard {value.shape} with names {pspec} on axis {axis_name}, no suitable axis found"
            )
            return DArray(value=value, pspec=pspec)

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
    """
    Shard parameters along a specific tensor dimension using the given mesh axis.

    This is similar to `fully_shard` but gives explicit control on which
    dimension to shard, enabling typical tensor-parallel patterns like
    column-parallel and row-parallel layers.

    - Validates that `axis_name` exists in `mesh`.
    - Respects existing sharding on each leaf by computing the effective shape
      after prior shardings.
    - If the effective size of `dim_to_sharded` is divisible by `mesh[axis_name]`,
      append `axis_name` to that pspec entry; otherwise leave unchanged and warn.
    """
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
        if not isinstance(leaf, DArray):
            return leaf

        value, pspec = leaf.value, leaf.pspec
        if value is None:
            return leaf

        if pspec is None:
            pspec = (None,) * value.ndim

        if len(pspec) != value.ndim:
            raise ValueError(
                "attribute names on Darray should have the same dimension with the attribute value"
            )

        if any(
            (p == axis_name)
            or (isinstance(p, tuple) and axis_name in p)
            for p in pspec
        ):
            logging.warning(
                f"Parameter {value.shape} with names {jax.tree_util.keystr(path)} already sharded on axis {axis_name}. the partition spec is {pspec}."
            )
            return DArray(value=value, pspec=pspec)

        ndim = value.ndim
        dim = tensor_dim_to_sharded if tensor_dim_to_sharded >= 0 else ndim + tensor_dim_to_sharded
        if dim < 0 or dim >= ndim:
            if skip_on_dim_mismatch:
                logging.info(
                    f"Skip sharding: dim {tensor_dim_to_sharded} out of range for shape {value.shape}"
                )
                return DArray(value=value, pspec=pspec)
            else:
                raise ValueError(
                    f"dim_to_sharded {tensor_dim_to_sharded} out of range for value.ndim={ndim}"
                )

        shape = value.shape
        divs = tuple(_effective_div(p) for p in pspec)
        eff_shape = tuple(int(s // d) for s, d in zip(shape, divs))

        if value.size < min_weight_size:
            logging.info(
                f"Skip sharding: small array {value.shape}, size {value.size} < {min_weight_size}"
            )
            return DArray(value=value, pspec=pspec)

        if eff_shape[dim] % axis_size == 0:
            new_i_pspec = _append_axis(pspec[dim], axis_name)
            new_pspec = pspec[:dim] + (new_i_pspec,) + pspec[dim + 1 :]
            return DArray(value=value, pspec=new_pspec)
        else:
            logging.warning(
                f"Could not shard {value.shape} with names {jtu.keystr(path)} on axis {axis_name} for dim {dim}; "
                f"effective size {eff_shape[dim]} not divisible by {axis_size}"
            )
            return DArray(value=value, pspec=pspec)

    return jtu.tree_map_with_path(_annotate_pspec, module, is_leaf=is_darray)


def shard_params(
    module: eqx.Module,
    mesh: Mesh,
    *,
    dim_to_axes: dict[int, tuple[str, ...]] | dict[int, list[str]],
    min_weight_size: int = 0,
) -> eqx.Module:
    """
    General parameter sharder that can attach multiple mesh axes to the same
    tensor dimension. Ensures composability by validating that the effective size
    of each dim is divisible by the product of mesh sizes for all axes being added.

    Example: to shard dim 0 over axes (tp, pp) and dim 1 over axis (sp):
        shard_params(m, mesh, dim_to_axes={0: ("tp", "pp"), 1: ("sp",)})
    """

    def _effective_div_for_entry(entry):
        if entry is None:
            return 1
        if isinstance(entry, str):
            if entry not in mesh.shape:
                raise ValueError(
                    f"axis {entry} is not in mesh; mesh contains {tuple(mesh.shape.keys())}"
                )
            return mesh.shape[entry]
        if isinstance(entry, tuple):
            d = 1
            for x in entry:
                if x not in mesh.shape:
                    raise ValueError(
                        f"axis {x} is not in mesh; mesh contains {tuple(mesh.shape.keys())}"
                    )
                d *= mesh.shape[x]
            return d
        return 1

    def _append_axes(entry, axes: tuple[str, ...]):
        if entry is None:
            if len(axes) == 0:
                return None
            return axes[0] if len(axes) == 1 else tuple(axes)
        if isinstance(entry, str):
            present = (entry,)
        else:
            present = tuple(entry)
        new = tuple(a for a in axes if a not in present)
        if len(new) == 0:
            return entry
        merged = present + new
        return merged[0] if len(merged) == 1 else merged

    def _annotate_pspec(leaf: tp.Any):
        if not isinstance(leaf, DArray):
            return leaf
        value, pspec = leaf.value, leaf.pspec
        if value is None:
            return leaf
        if pspec is None:
            pspec = (None,) * value.ndim
        if len(pspec) != value.ndim:
            raise ValueError(
                "attribute names on Darray should have the same dimension with the attribute value"
            )

        divs = tuple(_effective_div_for_entry(p) for p in pspec)
        eff_shape = tuple(int(s // d) for s, d in zip(value.shape, divs))

        new_pspec = list(pspec)
        for dim, axes in dim_to_axes.items():
            ndim = value.ndim
            d = dim if dim >= 0 else ndim + dim
            if d < 0 or d >= ndim:
                continue
            axes_tuple = tuple(axes)
            prod = 1
            for a in axes_tuple:
                if a not in mesh.shape:
                    raise ValueError(
                        f"axis {a} is not in mesh; mesh contains {tuple(mesh.shape.keys())}"
                    )
                prod *= mesh.shape[a]
            if value.size < min_weight_size:
                continue
            if eff_shape[d] % prod != 0:
                raise ValueError(
                    f"Could not shard {value.shape} with names {jtu.keystr(path)} on axes {axes_tuple} for dim {d}; "
                    f"effective size {eff_shape[d]} not divisible by {prod}"
                )
            new_pspec[d] = _append_axes(new_pspec[d], axes_tuple)

        return DArray(value=value, pspec=tuple(new_pspec))

    return jtu.tree_map(_annotate_pspec, module, is_leaf=is_darray)
