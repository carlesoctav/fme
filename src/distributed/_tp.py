import dataclasses as dc
import functools as ft
import types
from collections.abc import Callable

import equinox as eqx
from jax import P
from jax.sharding import Mesh

from src import nn
from ._params import tensor_parallel

class ModuleWithShardingConstraint(eqx.Module):
    _inputs_layout: P | tuple[P, ...] | None = eqx.field(static=True, default=None)
    _outputs_layout: P | tuple[P, ...] | None = eqx.field(static=True, default=None)

def _normalize_layout(layout, n_args):
    if layout is None:
        return None
    if isinstance(layout, tuple):
        return layout
    return (layout,) * n_args

def make_module_with_sharding_constraint(
    module: eqx.Module,
    params_partition_fn: Callable[[eqx.Module], eqx.Module] | None,
    prepare_input_fn: Callable[[eqx.Module, tuple, dict], tuple] | None,
    prepare_output_fn: Callable[[eqx.Module, object], object] | None,
    name: str,
    methods: tuple[str, ...] = ("__call__",),
) -> eqx.Module:
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

    for f in dc.fields(base_cls):
        try:
            object.__setattr__(new_module, f.name, getattr(module, f.name))
        except Exception:
            pass

    if params_partition_fn is not None:
        new_module = params_partition_fn(new_module)

    return new_module

def column_parallel(
    module: eqx.Module,
    axis_name: str,
    mesh: Mesh,
    *,
    inputs_layout: P | tuple[P, ...] | None = None,
    outputs_layout: P | tuple[P, ...] | None = None,
) -> eqx.Module:
    def _params_fn(m: eqx.Module) -> eqx.Module:
        if isinstance(m, nn.Linear):
            return tensor_parallel(m, mesh=mesh, axis_name=axis_name, dim_to_sharded=0)
        elif isinstance(m, nn.Embedding):
            return tensor_parallel(m, mesh=mesh, axis_name=axis_name, dim_to_sharded=1)
        else:
            raise ValueError(f"column_parallel only supports nn.Linear and nn.Embedding, got {m}")

    def _prep_in(self, args: tuple, kwargs: dict) -> tuple:
        norm_layout = _normalize_layout(inputs_layout, len(args))
        if norm_layout is None:
            return args
        if len(args) != len(norm_layout):
            raise ValueError("input_layouts must match number of positional args")
        return eqx.filter_shard(args, norm_layout)

    def _prep_out(self, out):
        if isinstance(out, tuple):
            norm_layout = _normalize_layout(outputs_layout, len(out))
            if norm_layout is None:
                return out
            if len(norm_layout) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            return eqx.filter_shard(out, norm_layout)
        else:
            if isinstance(outputs_layout, tuple):
                if len(outputs_layout) != 1:
                    raise ValueError("Single output expects P or tuple of length 1")
                norm_layout = outputs_layout[0]
            else:
                norm_layout = outputs_layout
            return eqx.filter_shard(out, norm_layout)

    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=_params_fn,
        prepare_input_fn=_prep_in,
        prepare_output_fn=_prep_out,
        name="ColParallel",
    )

def row_parallel(
    module: eqx.Module,
    axis_name: str,
    mesh: Mesh,
    *,
    inputs_layout: P | tuple[P, ...] | None = None,
    outputs_layout: P | tuple[P, ...] | None = None,
) -> eqx.Module:
    def _params_fn(m: eqx.Module) -> eqx.Module:
        if isinstance(m, nn.Linear):
            return tensor_parallel(m, mesh=mesh, axis_name=axis_name, dim_to_sharded=1)
        elif isinstance(m, nn.Embedding):
            return tensor_parallel(m, mesh=mesh, axis_name=axis_name, dim_to_sharded=0)
        else:
            raise ValueError(f"Row-parallel only supports Linear/Embedding, got {m}")
    def _prep_in(self, args: tuple, kwargs: dict) -> tuple:
        norm_layout = _normalize_layout(inputs_layout, len(args))
        if norm_layout is None:
            return args
        if len(args) != len(norm_layout):
            raise ValueError("input_layouts must match number of positional args")
        return eqx.filter_shard(args, norm_layout)
    def _prep_out(self, out):
        if isinstance(out, tuple):
            norm_layout = _normalize_layout(outputs_layout, len(out))
            if norm_layout is None:
                return out
            if len(norm_layout) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            return eqx.filter_shard(out, norm_layout)
        else:
            if isinstance(outputs_layout, tuple):
                if len(outputs_layout) != 1:
                    raise ValueError("Single output expects P or tuple of length 1")
                norm_layout = outputs_layout[0]
            else:
                norm_layout = outputs_layout
            return eqx.filter_shard(out, norm_layout)

    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=_params_fn,
        prepare_input_fn=_prep_in,
        prepare_output_fn=_prep_out,
        name="RowParallel",
    )

def prepare_input(
    module: eqx.Module,
    inputs_layout: P | tuple[P, ...] | None,
) -> eqx.Module:
    def _prep_in(self, args: tuple, kwargs: dict) -> tuple:
        norm_layout = _normalize_layout(inputs_layout, len(args))
        if norm_layout is None:
            return args
        if len(args) != len(norm_layout):
            raise ValueError("input_layouts must match number of positional args")
        return eqx.filter_shard(args, norm_layout)
    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=None,
        prepare_input_fn=_prep_in,
        prepare_output_fn=None,
        name="WithPrepareInput",
    )

def prepare_output(
    module: eqx.Module,
    outputs_layout: P | tuple[P, ...] | None,
) -> eqx.Module:
    def _prep_out(self, out):
        if isinstance(out, tuple):
            norm_layout = _normalize_layout(outputs_layout, len(out))
            if norm_layout is None:
                return out
            if len(norm_layout) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            return eqx.filter_shard(out, norm_layout)
        else:
            if isinstance(outputs_layout, tuple):
                if len(outputs_layout) != 1:
                    raise ValueError("Single output expects P or tuple of length 1")
                norm_layout = outputs_layout[0]
            else:
                norm_layout = outputs_layout
            return eqx.filter_shard(out, norm_layout)
    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=None,
        prepare_input_fn=None,
        prepare_output_fn=_prep_out,
        name="WithPrepareOutput",
    )

def prepare_input_output(
    module: eqx.Module,
    *,
    inputs_layout: P | tuple[P, ...] | None,
    outputs_layout: P | tuple[P, ...] | None,
) -> eqx.Module:
    def _prep_in(self, args: tuple, kwargs: dict) -> tuple:
        norm_layout = _normalize_layout(inputs_layout, len(args))
        if norm_layout is None:
            return args
        if len(args) != len(norm_layout):
            raise ValueError("input_layouts must match number of positional args")
        return eqx.filter_shard(args, norm_layout)
    def _prep_out(self, out):
        if isinstance(out, tuple):
            norm_layout = _normalize_layout(outputs_layout, len(out))
            if norm_layout is None:
                return out
            if len(norm_layout) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            return eqx.filter_shard(out, norm_layout)
        else:
            if isinstance(outputs_layout, tuple):
                if len(outputs_layout) != 1:
                    raise ValueError("Single output expects P or tuple of length 1")
                norm_layout = outputs_layout[0]
            else:
                norm_layout = outputs_layout
            return eqx.filter_shard(out, norm_layout)
    return make_module_with_sharding_constraint(
        module,
        params_partition_fn=None,
        prepare_input_fn=_prep_in,
        prepare_output_fn=_prep_out,
        name="PrepareInputOutput",
    )
