from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
from jax import P
from jax.sharding import Mesh

from src import nn
from ..module_utils import PrepareableModule, replace_prepare_hooks
from .params import tensor_parallel


PrepareFn = Callable[[PrepareableModule, tuple], tuple]
OutputFn = Callable[[PrepareableModule, object], object]


def _normalize_layout(layout, n_args):
    if layout is None:
        return None
    if isinstance(layout, tuple):
        return layout
    return (layout,) * n_args


def make_module_prepare_hooks(
    module: PrepareableModule,
    params_partition_fn: Callable[[PrepareableModule], PrepareableModule] | None,
    prepare_input_fn: PrepareFn | None,
    prepare_output_fn: OutputFn | None,
) -> PrepareableModule:
    if params_partition_fn is not None:
        module = params_partition_fn(module)

    if prepare_input_fn is None and prepare_output_fn is None:
        return module

    return replace_prepare_hooks(
        module,
        prepare_input=prepare_input_fn,
        prepare_output=prepare_output_fn,
    )


def column_parallel(
    module: PrepareableModule,
    axis_name: str,
    mesh: Mesh,
    *,
    inputs_layout: P | tuple[P, ...] | None = None,
    outputs_layout: P | tuple[P, ...] | None = None,
) -> PrepareableModule:
    def _params_fn(m: PrepareableModule) -> PrepareableModule:
        if isinstance(m, nn.Linear):
            return tensor_parallel(
                m, mesh=mesh, axis_name=axis_name, tensor_dim_to_sharded=0
            )
        if isinstance(m, nn.Embedding):
            return tensor_parallel(
                m, mesh=mesh, axis_name=axis_name, tensor_dim_to_sharded=1
            )
        raise ValueError(
            f"column_parallel only supports nn.Linear and nn.Embedding, got {m}"
        )

    def _prep_in(_module: PrepareableModule, args: tuple) -> tuple:
        norm_layout = _normalize_layout(inputs_layout, len(args))
        if norm_layout is None:
            return args
        if len(args) != len(norm_layout):
            raise ValueError("input_layouts must match number of positional args")
        return tuple(eqx.filter_shard(args, norm_layout))

    def _prep_out(_module: PrepareableModule, out):
        if isinstance(out, tuple):
            norm_layout = _normalize_layout(outputs_layout, len(out))
            if norm_layout is None:
                return out
            if len(norm_layout) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            return tuple(eqx.filter_shard(out, norm_layout))

        if isinstance(outputs_layout, tuple):
            if len(outputs_layout) != 1:
                raise ValueError("Single output expects P or tuple of length 1")
            norm_layout = outputs_layout[0]
        else:
            norm_layout = outputs_layout
        return eqx.filter_shard(out, norm_layout)

    return make_module_prepare_hooks(
        module,
        params_partition_fn=_params_fn,
        prepare_input_fn=_prep_in,
        prepare_output_fn=_prep_out,
    )


def row_parallel(
    module: PrepareableModule,
    axis_name: str,
    mesh: Mesh,
    *,
    inputs_layout: P | tuple[P, ...] | None = None,
    outputs_layout: P | tuple[P, ...] | None = None,
) -> PrepareableModule:
    def _params_fn(m: PrepareableModule) -> PrepareableModule:
        if isinstance(m, nn.Linear):
            return tensor_parallel(
                m, mesh=mesh, axis_name=axis_name, tensor_dim_to_sharded=1
            )
        if isinstance(m, nn.Embedding):
            return tensor_parallel(
                m, mesh=mesh, axis_name=axis_name, tensor_dim_to_sharded=0
            )
        raise ValueError("row_parallel only supports Linear/Embedding, got {m}")

    def _prep_in(_module: PrepareableModule, args: tuple) -> tuple:
        norm_layout = _normalize_layout(inputs_layout, len(args))
        if norm_layout is None:
            return args
        if len(args) != len(norm_layout):
            raise ValueError("input_layouts must match number of positional args")
        return tuple(eqx.filter_shard(args, norm_layout))

    def _prep_out(_module: PrepareableModule, out):
        if isinstance(out, tuple):
            norm_layout = _normalize_layout(outputs_layout, len(out))
            if norm_layout is None:
                return out
            if len(norm_layout) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            return tuple(eqx.filter_shard(out, norm_layout))

        if isinstance(outputs_layout, tuple):
            if len(outputs_layout) != 1:
                raise ValueError("Single output expects P or tuple of length 1")
            norm_layout = outputs_layout[0]
        else:
            norm_layout = outputs_layout
        return eqx.filter_shard(out, norm_layout)

    return make_module_prepare_hooks(
        module,
        params_partition_fn=_params_fn,
        prepare_input_fn=_prep_in,
        prepare_output_fn=_prep_out,
    )


def prepare_input(
    module: PrepareableModule,
    inputs_layout: P | tuple[P, ...] | None,
) -> PrepareableModule:
    def _prep_in(_module: PrepareableModule, args: tuple) -> tuple:
        norm_layout = _normalize_layout(inputs_layout, len(args))
        if norm_layout is None:
            return args
        if len(args) != len(norm_layout):
            raise ValueError("input_layouts must match number of positional args")
        return tuple(eqx.filter_shard(args, norm_layout))

    return make_module_prepare_hooks(
        module,
        params_partition_fn=None,
        prepare_input_fn=_prep_in,
        prepare_output_fn=None,
    )


def prepare_output(
    module: PrepareableModule,
    outputs_layout: P | tuple[P, ...] | None,
) -> PrepareableModule:
    def _prep_out(_module: PrepareableModule, out):
        if isinstance(out, tuple):
            norm_layout = _normalize_layout(outputs_layout, len(out))
            if norm_layout is None:
                return out
            if len(norm_layout) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            return tuple(eqx.filter_shard(out, norm_layout))

        if isinstance(outputs_layout, tuple):
            if len(outputs_layout) != 1:
                raise ValueError("Single output expects P or tuple of length 1")
            norm_layout = outputs_layout[0]
        else:
            norm_layout = outputs_layout
        return eqx.filter_shard(out, norm_layout)

    return make_module_prepare_hooks(
        module,
        params_partition_fn=None,
        prepare_input_fn=None,
        prepare_output_fn=_prep_out,
    )


def prepare_input_output(
    module: PrepareableModule,
    *,
    inputs_layout: P | tuple[P, ...] | None,
    outputs_layout: P | tuple[P, ...] | None,
) -> PrepareableModule:
    def _prep_in(_module: PrepareableModule, args: tuple) -> tuple:
        norm_layout = _normalize_layout(inputs_layout, len(args))
        if norm_layout is None:
            return args
        if len(args) != len(norm_layout):
            raise ValueError("input_layouts must match number of positional args")
        return tuple(eqx.filter_shard(args, norm_layout))

    def _prep_out(_module: PrepareableModule, out):
        if isinstance(out, tuple):
            norm_layout = _normalize_layout(outputs_layout, len(out))
            if norm_layout is None:
                return out
            if len(norm_layout) != len(out):
                raise ValueError("output_layouts must match number of outputs")
            return tuple(eqx.filter_shard(out, norm_layout))

        if isinstance(outputs_layout, tuple):
            if len(outputs_layout) != 1:
                raise ValueError("Single output expects P or tuple of length 1")
            norm_layout = outputs_layout[0]
        else:
            norm_layout = outputs_layout
        return eqx.filter_shard(out, norm_layout)

    return make_module_prepare_hooks(
        module,
        params_partition_fn=None,
        prepare_input_fn=_prep_in,
        prepare_output_fn=_prep_out,
    )

