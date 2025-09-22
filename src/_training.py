from __future__ import annotations

import typing as tp
from typing import Protocol

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray, PyTree
from optax import GradientTransformation, GradientTransformationExtraArgs

from ._filter import apply_transforms, iter_module
from ._metrics import MetricsAgg
from ._utils import first_from
from .callbacks import Callback, CallbackManager
from .distributed import get_partition_spec
from .loggers import Logger


_M = tp.TypeVar("_M", bound=eqx.Module)
_O = tp.TypeVar("_O", bound="Optimizer") 

_GradTx = GradientTransformation | GradientTransformationExtraArgs
_AxisSpec = bool | tp.Callable[[tp.Any], bool]
_Wrt = PyTree[_AxisSpec]
_Aux = dict[str, tp.Any]
_Loss = float
_Batch = tp.Any

_LossFn = tp.Callable[[_M, _O, _Batch], tuple[_Loss, _Aux]]
_ParallelismPlans = dict[str, tp.Callable[[_M], _M]] | tp.Sequence[dict[str, tp.Callable[[_M], _M]]]


_ModuleInput = tp.TypeVar("_ModuleInput", _M, tp.Sequence[_M])
_OptimizerInput = tp.TypeVar("_OptimizerInput", _O, tp.Sequence[_O])

class _TrainStepCallable(Protocol[_ModuleInput, _OptimizerInput]):
    def __call__(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        batch: tp.Any,
        key: PRNGKeyArray | None = None,
    ) -> tuple[_ModuleInput, _OptimizerInput, _Aux]:
        ...


class _EvalStepCallable(Protocol[_ModuleInput]):
    def __call__(
        self,
        module: _ModuleInput,
        batch: tp.Any,
        key: PRNGKeyArray | None = None,
    ) -> _Aux:
        ...


class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    wrt: PyTree[_AxisSpec] = eqx.field(static=True)
    step: int
    tx: GradientTransformationExtraArgs = eqx.field(static=True)

    def __init__(self, grad_tx: _GradTx, model: eqx.Module, *, wrt: _Wrt = eqx.is_inexact_array):
        self.tx = grad_tx
        self.wrt = wrt
        self.opt_state = self.tx.init(eqx.filter(model, self.wrt))
        self.step = 0

    def __call__(self, grads: PyTree[Array], model: eqx.Module) -> tuple[eqx.Module, Optimizer]:
        updates, opt_state = self.tx.update(grads, self.opt_state, eqx.filter(model, self.wrt))
        new_model = eqx.apply_updates(model, updates)
        new_self = eqx.tree_at(lambda o: [o.opt_state, o.step], self, [opt_state, self.step + 1])
        return new_model, new_self


_T = tp.TypeVar("_T")
def _as_list(x: _T | tp.Sequence[_T] | None) -> list[_T]:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _ensure_manager(
    callbacks: CallbackManager | tp.Sequence[Callback] | None,
) -> CallbackManager | None:
    if callbacks is None:
        return None
    if isinstance(callbacks, CallbackManager):
        return callbacks
    return CallbackManager(list(callbacks))


def init_module(
    module: eqx.Module,
    *,
    key: PRNGKeyArray,
    init_weights_plan: tp.Callable[[eqx.Module, jax.Array], eqx.Module] | None = None,
) -> eqx.Module:
    """Initialize module (abstract module ) leaves using optional plans or submodule init hooks."""

    root_plan_method = getattr(module, "init_weights_plan", None)

    init_weights_plan = first_from(
            init_weights_plan,
            root_plan_method,
            error_msg="init_weights_plan candidates unexpectedly empty",
        )

    getters: list[tp.Callable[[tp.Any], tp.Any]] = []
    replacements: list[tp.Any] = []

    for path, sub in iter_module(module, include_root=True):

        key, subkey = jax.random.split(key, 2)  

        new_sub = None

        if callable(init_weights_plan):
            try:
                cand = init_weights_plan(sub, subkey)
                if cand is not None:
                    new_sub = cand
            except TypeError:
                cand = init_weights_plan(sub) 
                if cand is not None:
                    new_sub = cand

        if new_sub is None and hasattr(sub, "init_weights") and callable(getattr(sub, "init_weights")):
            try:
                cand = sub.init_weights(key=subkey) 
            except TypeError:
                cand = sub.init_weights() 
            new_sub = cand

        if new_sub is not None:
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

    for get, rep in zip(getters, replacements):
        module = eqx.tree_at(get, module, rep)

    return module

#TODO: maybe add AbstractModule and do typecheck instead of tree_map scan
def _module_has_abstract_params(m: eqx.Module) -> bool:
    found = False

    def _is_shape_dtype_struct(x: tp.Any) -> bool:
        try:
            from jax import ShapeDtypeStruct

            return isinstance(x, ShapeDtypeStruct)
        except Exception:
            return False

    def _check(leaf):
        nonlocal found
        if _is_shape_dtype_struct(getattr(leaf, "value", leaf)):
            found = True
        return leaf

    jtu.tree_map(_check, m)
    return found

def setup_module_opts(
    module: _M,
    grad_tx: _GradTx,
    mesh: Mesh,
    *,
    wrt: _Wrt = eqx.is_inexact_array,
    parallelism_plans: _ParallelismPlans | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[_M, Optimizer]:
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
        m: _M,
        rng: PRNGKeyArray,
    ) -> tuple[_M, Optimizer]:
        if _module_has_abstract_params(m):
            m = init_module(m, key=rng)

        for plan in plans:
            m = apply_transforms(m, plan)

        pspec_tree = get_partition_spec(m)
        m_sharded = eqx.filter_shard(m, pspec_tree)
        opt = Optimizer(grad_tx, m_sharded, wrt=wrt)
        opt = eqx.filter_shard(opt, pspec_tree)

        return m_sharded, opt

    build = eqx.filter_jit(_build)

    with mesh:
        new_module, new_opt = build(module, key)

    return new_module, new_opt


@tp.overload
def make_train_step(
    *, 
    loss_function: _LossFn,
    jit: bool = True,
    default_key: PRNGKeyArray | None = None,
)-> _TrainStepCallable[_M, Optimizer]:
    ...

@tp.overload
def make_train_step(
    *, 
    train_step: _TrainStepCallable[_M, Optimizer],
    jit: bool = True,
    default_key: PRNGKeyArray | None = None,
)-> _TrainStepCallable[_M, Optimizer]:
    ...
@tp.overload
def make_train_step(
    *, 
    train_step: _TrainStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]],
    jit: bool = True,
    default_key: PRNGKeyArray | None = None,
    )-> _TrainStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]:
    ...
def make_train_step(
    *,
    loss_function: _LossFn | None = None,
    train_step: _TrainStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]] | _TrainStepCallable[_M, Optimizer] | None = None,
    jit: bool = True,
    default_key: PRNGKeyArray | None = None,
) -> _TrainStepCallable[_M, Optimizer] | _TrainStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]:

    if train_step is None and loss_function is None:
        raise ValueError("Provide either train_step or loss_function")

    if train_step is not None:
        return eqx.filter_jit(train_step) if jit else train_step

    assert loss_function is not None

    def _step(
        module: _M,
        optimizer: Optimizer,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[_M, Optimizer, _Aux]:
        grad_fn = eqx.filter_value_and_grad(loss_function, has_aux=True)
        (_, aux), grads = grad_fn(module, optimizer, batch, key=key)
        new_module, new_opt = optimizer(grads, module)
        return new_module, new_opt, aux

    return eqx.filter_jit(_step) if jit else _step

@tp.overload
def make_eval_step(
  *,
  loss_function: _LossFn, 
  jit: bool = True,
  default_key: PRNGKeyArray | None = None,
) -> _EvalStepCallable[_M]:
  ...

@tp.overload
def make_eval_step(
  *,
  eval_step: _EvalStepCallable[_M],
  jit: bool = True,
  default_key: PRNGKeyArray | None = None,
) -> _EvalStepCallable[_M]:
  ...

@tp.overload
def make_eval_step(
  *,
  eval_step: _EvalStepCallable[tp.Sequence[_M]],
  jit: bool = True,
  default_key: PRNGKeyArray | None = None,
) -> _EvalStepCallable[tp.Sequence[_M]]:
  ...

def make_eval_step(
  *,
  loss_function: _LossFn | None = None, 
  eval_step: _EvalStepCallable[tp.Sequence[_M]] | _EvalStepCallable[_M] | None = None,
  jit: bool = True,
  default_key: PRNGKeyArray | None = None,
) -> _EvalStepCallable[_M] | _EvalStepCallable[tp.Sequence[_M]]:
    if eval_step is None and loss_function is None:
        raise ValueError("Provide either eval_step or loss_function")

    if eval_step is not None:
        return eqx.filter_jit(eval_step) if jit else eval_step

    assert loss_function is not None

    def _step(
      module: _M,
      batch: tp.Any,
      *,
      key: PRNGKeyArray | None = None,
    ) -> _Aux:
        _, aux = loss_function(module, batch, key=key)
        return aux

    return eqx.filter_jit(_step) if jit else _step

def train_loop(
    *,
    num_train_steps: int,
    num_eval_steps: int,
    eval_interval: int,
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    data_loader: tp.Iterable[tp.Any],
    train_step: _TrainStepCallable[_ModuleInput, _OptimizerInput],
    eval_loader: tp.Iterable[tp.Any] | None = None,
    eval_step: _EvalStepCallable[_ModuleInput] | None = None,
    callbacks: CallbackManager | tp.Sequence[Callback] | None = None,
    loggers: Logger | tp.Sequence[Logger] | None = None,
    logging_enabled: bool = True,
    train_metrics: MetricsAgg | None = None,
    eval_metrics: MetricsAgg | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[_ModuleInput, _OptimizerInput, dict[str, float], dict[str, float]]:
    if num_train_steps <= 0:
        raise ValueError("num_train_steps must be positive")
    if eval_step is None and eval_loader is not None:
        raise ValueError("eval_step must be provided when eval_loader is not None")
    if eval_loader is None:
        eval_step = None

    callback_manager = _ensure_manager(callbacks)
    if isinstance(loggers, Logger):
        logger_list = [loggers]
    elif loggers is None:
        logger_list = []
    else:
        logger_list = list(loggers)

    if not logging_enabled:
        logger_list = []

    if train_metrics is None:
        train_metrics = MetricsAgg()
    if eval_metrics is None:
        eval_metrics = MetricsAgg()

    rng = key
    step = 0


    if callback_manager:
        callback_manager.call("on_training_start", module=module, optimizer=optimizer, step=step)

    train_iter = iter(data_loader)
    eval_iter = iter(eval_loader) if eval_loader is not None else None

    last_train_log: dict[str, float] | None = None
    last_eval_log: dict[str, float] | None = None

    for step in range(1, num_train_steps + 1):
        if rng is not None:
            rng, step_key = jax.random.split(rng)
        else:
            step_key = None

        batch = next(train_iter)

        if callback_manager:
            callback_manager.call(
                "on_train_step",
                module=module,
                optimizer=optimizer,
                batch=batch,
                step=step,
                key=step_key,
            )

        module, optimizer, aux = train_step(module, optimizer, batch, key=step_key)

        train_metrics.update(aux)
        current_metrics = train_metrics.compute(reset=False)

        reset_train = False
        for lg in logger_list:
            if lg.log_metrics(current_metrics, step=step, mode="train"):
                reset_train = True
                last_train_log = dict(current_metrics)
        if reset_train:
            train_metrics.reset()

        if callback_manager:
            callback_manager.call(
                "on_train_step_end",
                module=module,
                optimizer=optimizer,
                batch=batch,
                step=step,
                aux=aux,
                metrics=current_metrics,
            )

        if (
            eval_step is not None
            and eval_interval > 0
            and step % eval_interval == 0
        ):
            if callback_manager:
                callback_manager.call("on_eval_start", module=module, optimizer=optimizer, step=step)
            eval_metrics.reset()

            eval_rng = rng
            for eval_idx in range(1, num_eval_steps + 1):
                if eval_iter is None:
                    raise ValueError("eval_loader must be provided for evaluation")
                if eval_rng is not None:
                    eval_rng, eval_key = jax.random.split(eval_rng)
                else:
                    eval_key = None

                eval_batch = next(eval_iter)
                if callback_manager:
                    callback_manager.call(
                        "on_eval_step",
                        module=module,
                        optimizer=optimizer,
                        batch=eval_batch,
                        step=step,
                        eval_step=eval_idx,
                        key=eval_key,
                    )

                eval_aux = eval_step(module, eval_batch, key=eval_key)

                eval_metrics.update(eval_aux)
                eval_current = eval_metrics.compute(reset=False)

                reset_eval = False
                for lg in logger_list:
                    if lg.log_metrics(eval_current, step=eval_idx, mode="eval"):
                        reset_eval = True
                        last_eval_log = dict(eval_current)
                if reset_eval:
                    eval_metrics.reset()

                if callback_manager:
                    callback_manager.call(
                        "on_eval_step_end",
                        module=module,
                        optimizer=optimizer,
                        batch=eval_batch,
                        step=step,
                        eval_step=eval_idx,
                        aux=eval_aux,
                        metrics=eval_current,
                    )

            if callback_manager:
                callback_manager.call(
                    "on_eval_end",
                    module=module,
                    optimizer=optimizer,
                    step=step,
                    metrics=eval_metrics.compute(reset=False),
                )
            rng = eval_rng

    if callback_manager:
        callback_manager.call("on_training_end", module=module, optimizer=optimizer, step=step)

    train_summary = train_metrics.compute(reset=False)
    if not train_summary and last_train_log is not None:
        train_summary = last_train_log

    eval_summary: dict[str, float] = {}
    if eval_step is not None:
        eval_summary = eval_metrics.compute(reset=False)
        if not eval_summary and last_eval_log is not None:
            eval_summary = last_eval_log

    return module, optimizer, train_summary, eval_summary
