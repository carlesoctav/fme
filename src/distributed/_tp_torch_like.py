from typing import Callable, Sequence
import dataclasses as dc
import types
import functools as ft

import equinox as eqx
import jax
from jax import P
from jax.sharding import PartitionSpec, Mesh
from jaxtyping import Array


# Simple placement schema similar to torch parallel plans.
@dc.dataclass(frozen=True)
class Placement:
    mesh_axis_name: str


@dc.dataclass(frozen=True)
class Shard(Placement):
    dim: int  # tensor dimension to shard for this mesh axis


@dc.dataclass(frozen=True)
class Replicate(Placement):
    pass


class ModuleWithShardingConstraint:
    pass


def _append_axis_entry(entry, axis_name: str):
    if entry is None:
        return axis_name
    if isinstance(entry, str):
        if entry == axis_name:
            return entry
        return (entry, axis_name)
    if isinstance(entry, tuple):
        if axis_name in entry:
            return entry
        return entry + (axis_name,)
    return axis_name


def _remove_axis_entry(entry, axis_name: str):
    if entry is None:
        return None
    if isinstance(entry, str):
        return None if entry == axis_name else entry
    if isinstance(entry, tuple):
        newt = tuple(x for x in entry if x != axis_name)
        return None if len(newt) == 0 else newt
    return entry


def infer_pspec_from_placement(arr: Array, placements: Sequence[Placement]) -> PartitionSpec:
    """
    Build a PartitionSpec for `arr` from a list of mesh-dimension placements.

    - Each `Shard(dim, axis_name)` appends `axis_name` to entry `dim`.
    - Each `Replicate(axis_name)` removes `axis_name` from all entries, i.e. replicate across that mesh axis.
    - If `arr` already has sharding info, we start from an all-None baseline and only use `placements`.
    """
    # Start from current sharding if available; otherwise all-None.
    current_spec = getattr(getattr(arr, "sharding", None), "spec", None)
    entries = None
    if current_spec is not None:
        # Try common PartitionSpec internals
        entries = getattr(current_spec, "partitions", None)
        if entries is None:
            try:
                entries = tuple(current_spec)
            except Exception:
                entries = None
    if entries is None:
        entries = (None,) * arr.ndim

    spec_list: list[object] = list(entries)
    # Apply placements in order, allowing multiple mesh axes to shard the same tensor dim.
    for placement in placements:
        if isinstance(placement, Shard):
            dim = placement.dim if placement.dim >= 0 else arr.ndim + placement.dim
            if dim < 0 or dim >= arr.ndim:
                raise ValueError(f"Shard dim {placement.dim} out of range for ndim={arr.ndim}")
            spec_list[dim] = _append_axis_entry(spec_list[dim], placement.mesh_axis_name)
        elif isinstance(placement, Replicate):
            # Ensure that axis_name is not used in any dim, which encodes replication along that axis.
            spec_list = [_remove_axis_entry(e, placement.mesh_axis_name) for e in spec_list]
        else:
            raise TypeError(f"Unknown placement: {placement}")
    return PartitionSpec(*spec_list)


def make_module_with_sharding_constraint(
    module: eqx.Module,
    params_partition_fn: Callable[[eqx.Module], eqx.Module] | None,
    prepare_input_fn: Callable[[eqx.Module, tuple, dict], tuple] | None,
    prepare_output_fn: Callable[[eqx.Module, object], object] | None,
    name: str,
    methods: tuple[str, ...] = ("__call__",),
) -> eqx.Module:
    """
    Create a lightweight subclass of `module` that applies input/output sharding constraints
    around selected methods. Optionally apply a parameter-partitioning transform beforehand.
    """
    base_cls = module.__class__
    subclass = (base_cls, ModuleWithShardingConstraint)

    def _wrap_method(method_name: str, orig_fn):
        @ft.wraps(orig_fn)
        def method_with_prepare(self, *args, **kwargs):
            a, kw = args, kwargs
            if prepare_input_fn is not None:
                a = prepare_input_fn(self, a, kw)
            out = orig_fn(self, *a, **kw)
            if prepare_output_fn is not None:
                out = prepare_output_fn(self, out)
            return out

        return method_with_prepare

    dct: dict[str, object] = {}
    for m in methods:
        if not hasattr(module, m):
            raise ValueError(f"module {module} doesn't have method {m}")
        orig_method = getattr(base_cls, m, None)
        if orig_method is None:
            # Fallback to bound attribute and extract function
            bound = getattr(module, m)
            if isinstance(bound, types.MethodType):
                orig_method = bound.__func__
            elif isinstance(bound, types.FunctionType):
                orig_method = bound
            else:
                raise TypeError("method must be function or method")
        dct[m] = _wrap_method(m, orig_method)

    new_cls = type(f"{name}{base_cls.__name__}", subclass, dct)
    new_module = object.__new__(new_cls)
    # Copy dataclass fields
    for f in dc.fields(base_cls):
        try:
            setattr(new_module, f.name, getattr(module, f.name))
        except Exception:
            pass

    # Optionally partition params by returning a transformed module
    if params_partition_fn is not None:
        new_module = params_partition_fn(new_module)

    return new_module


def _pspec_last_dim(axis_name: str, arr: Array):
    if not isinstance(arr, jax.Array) or arr.ndim == 0:
        return None
    return P(*( (None,) * (arr.ndim - 1) + (axis_name,) ))


def _pspec_last_dim_replicated(arr: Array):
    if not isinstance(arr, jax.Array) or arr.ndim == 0:
        return None
    return P(*( (None,) * arr.ndim ))


def row_parallel(
    module: eqx.Module,
    axis_name: str,
    mesh: Mesh,
    *,
    shard_input_last_dim: bool = True,
    replicate_output: bool = True,
) -> eqx.Module:
    """
    Row-parallel transform:
    - Shard parameters along the input dimension (dim=1 for typical Linear [out,in]).
    - Constrain inputs to shard on the last dim; outputs replicated by default.
    """

    # lazy import to avoid circulars
    from ._spmd import tensor_parallel

    def _params_fn(m: eqx.Module) -> eqx.Module:
        return tensor_parallel(m, mesh=mesh, axis_name=axis_name, dim_to_sharded=1)

    def _prep_in(self, args: tuple, kwargs: dict) -> tuple:
        if len(args) == 0:
            return args
        x0 = args[0]
        if isinstance(x0, jax.Array) and shard_input_last_dim:
            ps = _pspec_last_dim(axis_name, x0)
            if ps is not None:
                x0 = jax.lax.with_sharding_constraint(x0, ps)
        return (x0,) + tuple(args[1:])

    def _prep_out(self, out):
        if isinstance(out, jax.Array):
            ps = _pspec_last_dim_replicated(out) if replicate_output else _pspec_last_dim(axis_name, out)
            if ps is not None:
                out = jax.lax.with_sharding_constraint(out, ps)
        return out

    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=_params_fn,
        prepare_input_fn=_prep_in,
        prepare_output_fn=_prep_out,
        name="RowParallel",
    )


def column_parallel(
    module: eqx.Module,
    axis_name: str,
    mesh: Mesh,
    *,
    input_layouts: tuple[Sequence[Placement] | None, ...] | None = None,
    output_layouts: tuple[Sequence[Placement] | None, ...] | Sequence[Placement] | None = None,
) -> eqx.Module:
    """
    Column-parallel transform:
    - Shard parameters along the output dimension (dim=0 for typical Linear [out,in]).
    - Use explicit `input_layout`/`output_layout` (PartitionSpec) to constrain I/O.

    If `input_layout`/`output_layout` is None, no constraint is applied on that side.
    This allows composing with other mesh axes: provide a PartitionSpec that includes
    all desired axes (e.g., last dim sharded by ("dp","tp")).
    """
    from ._spmd import tensor_parallel

    def _params_fn(m: eqx.Module) -> eqx.Module:
        return tensor_parallel(m, mesh=mesh, axis_name=axis_name, dim_to_sharded=0)

    def _prep_in(self, args: tuple, kwargs: dict) -> tuple:
        if not input_layouts:
            return args
        if len(args) != len(input_layouts):
            raise ValueError("input_layouts must match number of positional args")
        out = list(args)
        for i, (x, pls) in enumerate(zip(args, input_layouts)):
            if pls is None or not isinstance(x, jax.Array):
                continue
            ps = infer_pspec_from_placement(x, pls)
            out[i] = jax.lax.with_sharding_constraint(x, ps)
        return tuple(out)

    def _prep_out(self, out):
        if isinstance(out, tuple):
            if not isinstance(output_layouts, tuple):
                raise ValueError("Multi-output requires a tuple of per-output layouts")
            if len(output_layouts) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            outs = list(out)
            for i, (o, pls) in enumerate(zip(out, output_layouts)):
                if pls is None or not isinstance(o, jax.Array):
                    continue
                ps = infer_pspec_from_placement(o, pls)
                outs[i] = jax.lax.with_sharding_constraint(o, ps)
            return tuple(outs)
        else:
            if not isinstance(out, jax.Array) or output_layouts is None:
                return out
            if isinstance(output_layouts, tuple):
                raise ValueError("Single output expects Sequence[Placement], not tuple of outputs")
            ps = infer_pspec_from_placement(out, output_layouts)
            return jax.lax.with_sharding_constraint(out, ps)

    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=_params_fn,
        prepare_input_fn=_prep_in,
        prepare_output_fn=_prep_out,
        name="ColParallel",
    )


def prepare_input(
    module: eqx.Module,
    input_layouts: tuple[Sequence[Placement] | None, ...] | None,
) -> eqx.Module:
    """
    Only annotate input constraints. `input_layouts` must match len(args) at call time; each
    entry is a sequence of Placement for the corresponding arg (or None to skip).
    """

    def _prep_in(self, args: tuple, kwargs: dict) -> tuple:
        if not input_layouts:
            return args
        if len(args) != len(input_layouts):
            raise ValueError("input_layouts must match number of positional args")
        out = list(args)
        for i, (x, pls) in enumerate(zip(args, input_layouts)):
            if pls is None or not isinstance(x, jax.Array):
                continue
            ps = infer_pspec_from_placement(x, pls)
            out[i] = jax.lax.with_sharding_constraint(x, ps)
        return tuple(out)

    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=None,
        prepare_input_fn=_prep_in,
        prepare_output_fn=None,
        name="PrepareInput",
    )


def prepare_output(
    module: eqx.Module,
    output_layouts: tuple[Sequence[Placement] | None, ...] | Sequence[Placement] | None,
) -> eqx.Module:
    """
    Only annotate output constraints.
    - If the module returns a single array, pass `output_layouts` as a Sequence[Placement].
    - If the module returns a tuple of outputs, pass a tuple with per-output Sequence[Placement] or None.
    """

    def _prep_out(self, out):
        if isinstance(out, tuple):
            if not isinstance(output_layouts, tuple):
                raise ValueError("Multi-output requires a tuple of per-output layouts")
            if len(output_layouts) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            outs = list(out)
            for i, (o, pls) in enumerate(zip(out, output_layouts)):
                if pls is None or not isinstance(o, jax.Array):
                    continue
                ps = infer_pspec_from_placement(o, pls)
                outs[i] = jax.lax.with_sharding_constraint(o, ps)
            return tuple(outs)
        else:
            if not isinstance(out, jax.Array) or output_layouts is None:
                return out
            if isinstance(output_layouts, tuple):
                raise ValueError("Single output expects Sequence[Placement], not tuple of outputs")
            ps = infer_pspec_from_placement(out, output_layouts)
            return jax.lax.with_sharding_constraint(out, ps)

    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=None,
        prepare_input_fn=None,
        prepare_output_fn=_prep_out,
        name="PrepareOutput",
    )


def prepare_input_output(
    module: eqx.Module,
    *,
    input_layouts: tuple[Sequence[Placement] | None, ...] | None,
    output_layouts: tuple[Sequence[Placement] | None, ...] | Sequence[Placement] | None,
) -> eqx.Module:
    """Annotate both input and output constraints without sharding parameters."""

    def _prep_in(self, args: tuple, kwargs: dict) -> tuple:
        if not input_layouts:
            return args
        if len(args) != len(input_layouts):
            raise ValueError("input_layouts must match number of positional args")
        out = list(args)
        for i, (x, pls) in enumerate(zip(args, input_layouts)):
            if pls is None or not isinstance(x, jax.Array):
                continue
            ps = infer_pspec_from_placement(x, pls)
            out[i] = jax.lax.with_sharding_constraint(x, ps)
        return tuple(out)

    def _prep_out(self, out):
        if isinstance(out, tuple):
            if not isinstance(output_layouts, tuple):
                raise ValueError("Multi-output requires a tuple of per-output layouts")
            if len(output_layouts) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            outs = list(out)
            for i, (o, pls) in enumerate(zip(out, output_layouts)):
                if pls is None or not isinstance(o, jax.Array):
                    continue
                ps = infer_pspec_from_placement(o, pls)
                outs[i] = jax.lax.with_sharding_constraint(o, ps)
            return tuple(outs)
        else:
            if not isinstance(out, jax.Array) or output_layouts is None:
                return out
            if isinstance(output_layouts, tuple):
                raise ValueError("Single output expects Sequence[Placement], not tuple of outputs")
            ps = infer_pspec_from_placement(out, output_layouts)
            return jax.lax.with_sharding_constraint(out, ps)

    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=None,
        prepare_input_fn=_prep_in,
        prepare_output_fn=_prep_out,
        name="PrepareInputOutput",
    )


def _resolve_path(obj, path: str):
    parts = path.split(".") if path else []
    parent = None
    key = None
    cur = obj
    for i, p in enumerate(parts):
        parent = cur
        key = p
        # numeric index for sequences
        if isinstance(parent, (list, tuple)) and p.isdigit():
            idx = int(p)
            cur = parent[idx]
        else:
            cur = getattr(parent, p)
    return parent, key, cur


def _assign_path(parent, key: str, value):
    if isinstance(parent, list) and key.isdigit():
        parent[int(key)] = value
    else:
        setattr(parent, key, value)


def apply_tp_plan(
    module: eqx.Module,
    mesh: Mesh,
    plan: dict[str, Callable[[eqx.Module], eqx.Module]],
) -> eqx.Module:
    """
    Apply a tensor-parallel plan dictionary to selected submodules.

    Example:
        apply_tp_plan(model, mesh, {
            "w1": lambda m: column_parallel(m, axis_name="tp", mesh=mesh),
            "w2": lambda m: row_parallel(m, axis_name="tp", mesh=mesh),
            "attn": lambda m: prepare_input_output(m, in_axis_name="tp", out_axis_name="tp"),
        })
    """
    new_root = module
    for path, fn in plan.items():
        parent, key, sub = _resolve_path(new_root, path)
        if parent is None or key is None:
            raise ValueError(f"Invalid path '{path}' in tp_plan")
        new_sub = fn(sub)
        _assign_path(parent, key, new_sub)
    return new_root
