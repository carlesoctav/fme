import dataclasses as dc
import typing as tp

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax import lax, P


class ModuleWithShardingConstraint(eqx.Module):

    """
    Mixin providing __call__ that applies with_sharding_constraint to inputs
    and outputs. By default, shards the last dimension of any Array found in
    args/kwargs or outputs, controlled by shard_in_last_dim/shard_out_last_dim.

    This mixin must be placed before the original class in MRO so that its
    __call__ overrides the base implementation, and then delegates to super().
    """

    _axis_name: str = eqx.field(static=True, default="tp")
    _tp_dim_index: int = eqx.field(static=True, default=-1)  # which dim to shard; -1 means last
    _tp_shard_in: bool = eqx.field(static=True, default=False)
    _tp_shard_out: bool = eqx.field(static=True, default=False)
    _tp_in_spec_fn: tp.Optional[tp.Callable[[jax.Array], tp.Any]] = eqx.field(
        static=True, default=None
    )
    _tp_out_spec_fn: tp.Optional[tp.Callable[[jax.Array], tp.Any]] = eqx.field(
        static=True, default=None
    )

    def _spec_for(self, arr: jax.Array, shard: bool, is_input: bool):
        if not isinstance(arr, jax.Array):
            return None
        if arr.ndim == 0:
            return None
        # If a custom spec function is provided, prefer it
        spec_fn = self._tp_in_spec_fn if is_input else self._tp_out_spec_fn
        if spec_fn is not None:
            return spec_fn(arr)

        # Helper: try to extract an existing PartitionSpec from the array, if any
        def _existing_parts(a: jax.Array) -> tp.Optional[list]:
            jax.NamedSharding
            a.sharding
            sharding = getattr(a, "sharding", None)
            spec = getattr(sharding, "spec", None)
            if spec is None:
                return None
            parts: tp.Optional[list] = None
            # Attempt common layouts across JAX versions
            for attr in ("partitions", "rules"):
                if hasattr(spec, attr):
                    try:
                        parts = list(getattr(spec, attr))
                        break
                    except Exception:
                        pass
            if parts is None:
                try:
                    parts = list(spec)  # if PartitionSpec is iterable
                except Exception:
                    return None

            # Expand Ellipsis if present (e.g., P(...))
            if any(p is Ellipsis for p in parts):
                # Replace a single Ellipsis with the right number of Nones
                if parts.count(Ellipsis) > 1:
                    return None
                missing = a.ndim - (len(parts) - 1)
                if missing < 0:
                    return None
                expanded: list = []
                for p in parts:
                    if p is Ellipsis:
                        expanded.extend([None] * missing)
                    else:
                        expanded.append(p)
                parts = expanded

            # Sanity-check length
            if len(parts) != a.ndim:
                return None
            return parts

        def _contains_axis(p, axis_name: str) -> bool:
            if p is None:
                return False
            if isinstance(p, (tuple, list)):
                return axis_name in p
            return p == axis_name

        axis = self._axis_name if shard else None

        # Determine target dim, preferring any dim that already carries this axis
        existing = _existing_parts(arr)
        dim_from_existing = None
        if existing is not None and axis is not None:
            for i, p in enumerate(existing):
                if _contains_axis(p, axis):
                    dim_from_existing = i
                    break

        # Choose target dim with support for negatives (like numpy)
        if dim_from_existing is not None:
            dim = dim_from_existing
        else:
            dim = self._tp_dim_index if self._tp_dim_index >= 0 else arr.ndim + self._tp_dim_index

        if dim < 0 or dim >= arr.ndim:
            # Out of range; skip
            return None

        # Start from existing parts if available; otherwise replicate everywhere
        parts = existing[:] if existing is not None else [None] * arr.ndim

        if axis is None:
            # When replicating, override only the chosen dim to None, but keep others
            parts[dim] = None
        else:
            # Merge axis into the chosen dim, de-duplicating
            cur = parts[dim]
            if cur is None:
                parts[dim] = axis
            elif isinstance(cur, (tuple, list)):
                if axis not in cur:
                    parts[dim] = tuple(list(cur) + [axis])
            elif cur != axis:
                parts[dim] = (cur, axis)

        return P(*parts)

    def _constrain_tree(self, tree, shard: bool, *, is_input: bool):
        def _apply(x):
            if isinstance(x, jax.Array):
                ps = self._spec_for(x, shard, is_input)
                if ps is not None:
                    return lax.with_sharding_constraint(x, ps)
            return x

        return jtu.tree_map(_apply, tree)

    def __call__(self, *args, **kwargs):  # type: ignore[override]
        # Apply input constraints
        args = self._constrain_tree(args, self._tp_shard_in, is_input=True)
        kwargs = self._constrain_tree(kwargs, self._tp_shard_in, is_input=True)
        # Delegate to original __call__ via MRO
        out = super().__call__(*args, **kwargs)  # type: ignore[misc]
        # Apply output constraints
        out = self._constrain_tree(out, self._tp_shard_out, is_input=False)
        return out


def make_module_with_sharding_constraint(
    module: tp.Any,
    *,
    axis_name: str,
    dim_index: int,
    shard_in: bool,
    shard_out: bool,
    in_spec_fn: tp.Optional[tp.Callable[[jax.Array], tp.Any]] = None,
    out_spec_fn: tp.Optional[tp.Callable[[jax.Array], tp.Any]] = None,
):
    base_cls = module.__class__
    name = f"ModuleWithShardingConstraint{base_cls.__name__}"
    new_cls = type(name, (ModuleWithShardingConstraint, base_cls), {})
    new_cls = dc.dataclass(new_cls)  # type: ignore[misc]

    obj = object.__new__(new_cls)
    for f in dc.fields(base_cls):
        try:
            setattr(obj, f.name, getattr(module, f.name))
        except Exception:
            pass

    # Set the correct axis name field
    setattr(obj, "_axis_name", axis_name)
    setattr(obj, "_tp_dim_index", dim_index)
    setattr(obj, "_tp_shard_in", shard_in)
    setattr(obj, "_tp_shard_out", shard_out)
    setattr(obj, "_tp_in_spec_fn", in_spec_fn)
    setattr(obj, "_tp_out_spec_fn", out_spec_fn)
    return obj


def as_column_parallel(module: tp.Any, axis_name: str, *, dim_index: int = -1):
    """Replicate inputs; shard outputs on the specified dim (column-parallel).

    By default shards the last activation dim (common for vectors coming out of
    per-token projections under vmap).
    """
    return make_module_with_sharding_constraint(
        module,
        axis_name=axis_name,
        dim_index=dim_index,
        shard_in=False,
        shard_out=True,
    )


def as_row_parallel(
    module: tp.Any,
    axis_name: str,
    *,
    dim_index: int = -1,
    out_sharded: bool = True,
):
    """Shard inputs on specified dim (row-parallel).

    If out_sharded is False, outputs are constrained to replicate, which tends
    to induce an all-reduce at the boundary.
    """
    return make_module_with_sharding_constraint(
        module,
        axis_name=axis_name,
        dim_index=dim_index,
        shard_in=True,
        shard_out=out_sharded,
    )



 - Wrapper to apply name→spec for inputs, handling both args and kwargs:
  import inspect
  import jax
  import jax.tree_util as jtu
  from jax import lax, P

  def _ps_tree_for_value(ps, value):
      # Allow users to provide either a single PartitionSpec or a pytree of specs.
      # If it's not a pytree matching value, broadcast ps across value's leaves.
      try:
          # If ps already matches value structure, this is a no-op map
          return jtu.tree_map(lambda _: ps, value)
      except Exception:
          # Fallback: still broadcast
          return jtu.tree_map(lambda _: ps, value)

  def wrap_inputs_with_specs(orig_fn, name_to_pspec):
      sig = inspect.signature(orig_fn)
      pos_names = [p.name for p in sig.parameters.values()
                   if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                 inspect.Parameter.POSITIONAL_OR_KEYWORD)]
      # For methods, skip 'self' if present
      if pos_names and pos_names[0] == "self":
          pos_names = pos_names[1:]

      def wrapped(self, *args, **kwargs):
          args = list(args)

          def apply(value, ps):
              ps_tree = _ps_tree_for_value(ps, value)
              return lax.with_sharding_constraint(value, ps_tree)

          for name, ps in name_to_pspec.items():
              if name in kwargs:
                  kwargs[name] = apply(kwargs[name], ps)
              elif name in pos_names:
                  idx = pos_names.index(name)
                  if idx < len(args):
                      args[idx] = apply(args[idx], ps)

          return orig_fn(self, *args, **kwargs)

      return wrapped
  Example

  - Given shard_in_pspec = {"x": P("data"), "random_thing": P("data")}, both work:
      - m(x_val, random_thing=kw_val)
      - m(x_val, kw_val)  # if signature is (self, x, random_thing)
  - If an argument is itself a pytree, pass a pytree of PartitionSpecs for fine-grained control, or rely on broadcasting the same P to all array leaves under that argument.
