from __future__ import annotations

import dataclasses
import typing as tp

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray, PyTree
from optax import GradientTransformation, GradientTransformationExtraArgs

from ._filter import apply_transforms, iter_module
from ._utils import first_from
from .distributed import get_partition_spec


M = tp.TypeVar("M", bound=eqx.Module)
T = tp.TypeVar("T")

GradTx = GradientTransformation | GradientTransformationExtraArgs
AxisSpec = bool | tp.Callable[[tp.Any], bool]
Wrt = PyTree[AxisSpec]
Aux = dict[str, tp.Any]
InitFn = tp.Callable[[jax.Array, tuple[int, ...], tp.Any], jax.Array]


class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    wrt: PyTree[AxisSpec] = eqx.field(static=True)
    step: int
    tx: GradientTransformationExtraArgs = eqx.field(static=True)

    def __init__(self, grad_tx: GradTx, model: eqx.Module, *, wrt: Wrt = eqx.is_inexact_array):
        self.tx = grad_tx
        self.wrt = wrt
        self.opt_state = self.tx.init(eqx.filter(model, self.wrt))
        self.step = 0

    def __call__(self, grads: PyTree[Array], model: eqx.Module) -> tuple[eqx.Module, Optimizer]:
        updates, opt_state = self.tx.update(grads, self.opt_state, eqx.filter(model, self.wrt))
        new_model = eqx.apply_updates(model, updates)
        new_self = eqx.tree_at(lambda o: [o.opt_state, o.step], self, [opt_state, self.step + 1])
        return new_model, new_self


@jtu.register_dataclass
@dataclasses.dataclass
class Metric:
    values: float
    counts: int
    mode: str

    def update(self, **kwargs) -> Metric:
        if "values" not in kwargs:
            raise ValueError("values must be present with type of int or float")
        if "counts" not in kwargs:
            raise ValueError("counts must be present with type of int")

        add_value = kwargs.get("values")
        add_count = kwargs.get("counts")
        assert isinstance(add_value, (float, int)), "values must be int or float value"
        assert isinstance(add_count, int), "counts must be int value"

        new_metric = eqx.tree_at(
            lambda m: [m.values, m.counts],
            self,
            [self.values + float(add_value), self.counts + add_count],
        )
        return new_metric

    def compute(self) -> float:
        raise NotImplementedError




def _as_list(x: T | tp.Sequence[T] | None) -> list[T]:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _is_shape_dtype_struct(x: tp.Any) -> bool:
    try:
        from jax import ShapeDtypeStruct

        return isinstance(x, ShapeDtypeStruct)
    except Exception:
        return False

def _reseed(key: jax.Array, tag: str) -> jax.Array:
    return jax.random.fold_in(key, (hash(tag) & 0xFFFFFFFF))


def init_module(
    module: eqx.Module,
    *,
    key: PRNGKeyArray,
    init_weights_plan: tp.Callable[[eqx.Module, jax.Array], eqx.Module] | None = None,
) -> eqx.Module:
    """Initialize module leaves using optional plans or submodule init hooks."""

    root_plan_method = getattr(module, "init_weights_plan", None)

    init_weights_plan = first_from(
            init_weights_plan,
            root_plan_method,
            error_msg="init_weights_plan candidates unexpectedly empty",
        )

    getters: list[tp.Callable[[tp.Any], tp.Any]] = []
    replacements: list[tp.Any] = []

    for path, sub in iter_module(module, include_root=True):
        tag = ".".join(str(p) for p in path) if path else "<root>"
        key, subkey = jax.random.split(key, 2)  

        new_sub = sub
        if callable(init_weights_plan):
            try:
                cand = init_weights_plan(sub, subkey)
                if cand is not None:
                    new_sub = cand
            except TypeError:
                cand = init_weights_plan(sub) 
                if cand is not None:
                    new_sub = cand

        if new_sub is sub and hasattr(sub, "init_weights") and callable(getattr(sub, "init_weights")):
            try:
                cand = sub.init_weights(key=subkey) 
            except TypeError:
                cand = sub.init_weights() 
            new_sub = cand

        if new_sub is not sub:

            def _getter_from_path(pth):
                def get(root):
                    node = root
                    for part in pth:
                        if isinstance(part, int):
                            node = node[part]
                        else:
                            node = getattr(node, part)
                    return node

                return get

            getters.append(_getter_from_path(path))
            replacements.append(new_sub)

    mod = module
    for get, rep in zip(getters, replacements):
        mod = eqx.tree_at(get, mod, rep)

    return mod

def _module_has_abstract_params(m: eqx.Module) -> bool:
    found = False

    def _check(leaf):
        nonlocal found
        if _is_shape_dtype_struct(getattr(leaf, "value", leaf)):
            found = True
        return leaf

    jtu.tree_map(_check, m)
    return found


def _maybe_do_sched(
    fn: tp.Callable[..., tp.Any],
    *,
    curr_step: int,
    every: int | None = None,
    at: int | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
):
    if kwargs is None:
        kwargs = {}
    if at is not None:
        if curr_step == at:
            return fn(*args, **kwargs)
        return None
    if every is None:
        return None
    if every <= 0:
        return None
    if curr_step % every == 0:
        return fn(*args, **kwargs)
    return None


def maybe_do(
    function: tp.Callable[..., tp.Any],
    function_args: tuple,
    function_kwargs: dict | None,
    curr_step: int,
    required_step: int,
):
    """Simple scheduling helper: run function every required_step with given args/kwargs."""
    if function_kwargs is None:
        function_kwargs = {}
    return _maybe_do_sched(
        function,
        curr_step=curr_step,
        every=required_step,
        args=function_args,
        kwargs=function_kwargs,
    )


def _shape_key(x) -> tuple[int, ...] | None:
    try:
        return tuple(x.shape)  # type: ignore[attr-defined]
    except Exception:
        return None


def _infer_opt_state_spec_from_params(opt_state: PyTree, params_spec_tree: PyTree) -> PyTree:
    """Approximate mapping from parameter specs to optimizer state specs by shape."""
    params_specs = list(jtu.tree_leaves(params_spec_tree, is_leaf=lambda _: False))

    from jax import P

    shape_to_spec: dict[tuple[int, ...], P | None] = {}
    for spec in params_specs:
        pval = getattr(spec, "value", spec)
        key = _shape_key(pval)
        if key is None:
            continue
        shape_to_spec[key] = pval

    def _assign_spec(leaf):
        key = _shape_key(leaf)
        if key is None:
            return None
        return shape_to_spec.get(key, None)

    return jtu.tree_map(_assign_spec, opt_state)


def setup_module_opts(
    module: M,
    grad_tx: GradTx,
    mesh: Mesh,
    *,
    wrt: Wrt = eqx.is_inexact_array,
    parallelism_plans: (
        tp.Sequence[dict[str, tp.Callable[[eqx.Module], eqx.Module]]]
        | dict[str, tp.Callable[[eqx.Module], eqx.Module]]
        | None
    ) = None,
    key: PRNGKeyArray | None = None,
) -> tuple[M, Optimizer]:
    if not isinstance(module, eqx.Module):
        raise TypeError("module must be an equinox.Module instance")
    if not isinstance(grad_tx, (GradientTransformation, GradientTransformationExtraArgs)):
        raise TypeError(
            "grad_tx must be an optax.GradientTransformation or GradientTransformationExtraArgs instance"
        )
    if key is None:
        raise ValueError("key must be provided for initialization")

    plans = _as_list(parallelism_plans)

    def _build(
        m: M,
        rng: PRNGKeyArray,
    ) -> tuple[M, Optimizer]:
        if _module_has_abstract_params(m):
            m = init_module(m, key=rng)

        for plan in plans:
            m = apply_transforms(m, plan)

        pspec_tree = get_partition_spec(m)
        m_sharded = eqx.filter_shard(m, pspec_tree)
        opt = Optimizer(grad_tx, m_sharded, wrt=wrt)

        # try:
        #     # opt_state_sharded = eqx.filter_shard(opt, pspec_tree)
        # except Exception:
        #     pass

        return m_sharded, opt

    build = eqx.filter_jit(_build)

    with mesh:
        new_module, new_opt = build(module, key)

    return new_module, new_opt


def make_train_step(
    *,
    loss_function: tp.Callable[[M, Optimizer, tp.Any], tuple[float, Aux]] | None = None,
    train_step: tp.Callable[[M, Optimizer, tp.Any], tuple[M, Optimizer, Aux]] | None = None,
    jit: bool = True,
    default_key: PRNGKeyArray | None = None,
) -> tp.Callable[[M, Optimizer, tp.Any], tuple[M, Optimizer, Aux]]:

    if train_step is None and loss_function is None:
        raise ValueError("Provide either train_step or loss_function")

    if train_step is not None:
        return eqx.filter_jit(train_step) if jit else train_step

    assert loss_function is not None

    def _step(
        module: M,
        optimizer: Optimizer,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[M, Optimizer, Aux]:
        k = key if key is not None else default_key
        if k is None:
            k = jax.random.PRNGKey(0)
        grad_fn = eqx.filter_value_and_grad(loss_function, has_aux=True)
        (_, aux), grads = grad_fn(module, optimizer, batch, key=k)
        new_module, new_opt = optimizer(grads, module)
        return new_module, new_opt, aux

    return eqx.filter_jit(_step) if jit else _step


def compute_metrics(
    metrics: PyTree[Metric] | None,
    aux: dict[str, tp.Any],
    *,
    mode: str = "train",
) -> PyTree[Metric]:
    """Update a Metric PyTree using aux from a step."""

    def _to_metric(v) -> Metric:
        if isinstance(v, Metric):
            return v
        if isinstance(v, tuple) and len(v) == 2:
            val, cnt = v
            return Metric(values=float(val), counts=int(cnt), mode=mode)
        if isinstance(v, (int, float)):
            return Metric(values=float(v), counts=1, mode=mode)
        return Metric(values=0.0, counts=0, mode=mode)

    updates: dict[str, Metric] = {k: _to_metric(v) for k, v in aux.items()}
    if metrics is None:
        return updates

    def _merge(old: Metric | None, new: Metric) -> Metric:
        if old is None:
            return new
        return old.update(values=new.values, counts=new.counts)

    out = dict(metrics)
    for k, v in updates.items():
        out[k] = _merge(out.get(k), v)
    return out


def metrics_to_host(metrics: PyTree[Metric]) -> dict[str, float]:
    """Compute scalar host metrics from Metric PyTree (value/count average)."""
    host: dict[str, float] = {}
    for k, m in metrics.items():
        denom = max(1, int(m.counts))
        host[k] = float(m.values) / float(denom)
    return host


def maybe_write(
    logger: tp.Any,
    metrics: PyTree[Metric],
    step: int,
    *,
    every: int = 1,
    mode: str = "train",
) -> None:
    if logger is None:
        return
    _maybe_do_sched(
        logger.log_host_metrics,
        curr_step=step,
        every=every,
        args=(metrics_to_host(metrics), step),
        kwargs={"mode": mode},
    )


def maybe_checkpoint(
    checkpoint_manager: tp.Any,
    state: PyTree,
    step: int,
    *,
    every: int = 0,
    at: int | None = None,
) -> None:
    if checkpoint_manager is None:
        return

    def _save():
        checkpoint_manager.save(step, args=state)

    _maybe_do_sched(_save, curr_step=step, every=every, at=at)
