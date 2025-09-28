from __future__ import annotations

import logging
import typing as tp
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray, PyTree
from optax import GradientTransformation, GradientTransformationExtraArgs
from tqdm.auto import tqdm

from ._filter import apply_transforms, iter_module
from ._utils import first_from, rank_zero
from .callbacks import Callback
from .distributed import get_partition_spec
from .loggers import Logger, TensorBoardLogger


LOGGER = logging.getLogger(__name__)


_M = tp.TypeVar("_M", bound=eqx.Module)
_O = tp.TypeVar("_O", bound="Optimizer") 

_GradTx = GradientTransformation | GradientTransformationExtraArgs
_AxisSpec = bool | tp.Callable[[tp.Any], bool]
_Wrt = PyTree[_AxisSpec]
_Aux = dict[str, tp.Any]
_Loss = float
_Batch = tp.Any

_LossFn = tp.Callable[[_M, _O, _Batch, PRNGKeyArray], tuple[_Loss, _Aux]]
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


WallclockFactory = tp.Callable[[str, int | None], AbstractContextManager[tp.Any]]
EvalLoopResult = tuple[dict[str, float], _Aux | None, PRNGKeyArray | None]


class _EvalLoopCallable(Protocol[_ModuleInput, _OptimizerInput]):
    def __call__(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        eval_loader: tp.Iterable[tp.Any],
        eval_step: _EvalStepCallable[_ModuleInput],
        callbacks: tp.Sequence[Callback],
        logger: Logger | None,
        key: PRNGKeyArray | None,
    ) -> EvalLoopResult:
        ...


class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    wrt: PyTree[_AxisSpec] = eqx.field(static=True)
    step: jax.Array
    tx: GradientTransformationExtraArgs = eqx.field(static=True)

    def __init__(self, grad_tx: _GradTx, model: eqx.Module, *, wrt: _Wrt = eqx.is_inexact_array):
        self.tx = grad_tx
        self.wrt = wrt
        self.opt_state = self.tx.init(eqx.filter(model, self.wrt))
        self.step = jnp.array(0, dtype=jnp.int32)

    def __call__(self, grads: PyTree[Array], model: eqx.Module) -> tuple[eqx.Module, Optimizer]:
        updates, opt_state = self.tx.update(grads, self.opt_state, eqx.filter(model, self.wrt))
        new_model = eqx.apply_updates(model, updates)
        new_step = self.step + jnp.array(1, dtype=self.step.dtype)
        new_self = eqx.tree_at(lambda o: [o.opt_state, o.step], self, [opt_state, new_step])
        return new_model, new_self


_T = tp.TypeVar("_T")
def _as_list(x: _T | tp.Sequence[_T] | None) -> list[_T]:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]

def _split_batch_for_accum(batch: tp.Any, steps: int) -> tp.Any:
    def reshape(arr):
        if isinstance(arr, (jax.Array, np.ndarray)):
            if arr.shape[0] % steps != 0:
                raise ValueError("Global batch dimension must be divisible by gradient_accumulation_steps")
            new_shape = (steps, arr.shape[0] // steps) + arr.shape[1:]
            return arr.reshape(new_shape)
        return arr

    return jax.tree_util.tree_map(reshape, batch)

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
            "submodule init",
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

def make_module_opt(
    module: _M,
    grad_tx: _GradTx,
    mesh: Mesh | None = None,
    wrt: _Wrt = eqx.is_inexact_array,
    parallelism_plans: _ParallelismPlans | None = None,
    *,
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

    with mesh if mesh else nullcontext():
        new_module, new_opt = build(module, key)

    return new_module, new_opt


def make_train_step(
    loss_function: _LossFn | None = None,
    train_step: _TrainStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]] | _TrainStepCallable[_M, Optimizer] | None = None,
    *, 
    gradient_accumulation_steps: int = 1,
    jit: bool = True,
) -> _TrainStepCallable[_M, Optimizer] | _TrainStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]:

    if train_step is None and loss_function is None:
        raise ValueError("Provide either train_step or loss_function")

    if train_step is not None:
        fn = eqx.filter_jit(train_step) if jit else train_step
        return fn

    assert loss_function is not None
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    grad_fn = eqx.filter_value_and_grad(loss_function, has_aux=True)


    def _step(
        module: _M,
        optimizer: Optimizer,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[_M, Optimizer, _Aux]:

        def _minibatch_step(batch):
            (_, step_aux), step_grad = grad_fn(module, optimizer, batch, key=key)
            return step_grad, step_aux

        def _scan_step(carry, batch):
            grad, aux = carry
            step_carry = _minibatch_step(batch)
            return jtu.tree_map(jnp.add, carry, step_carry), None

        def _stack_batch(leaf):
            B = leaf.shape[0]
            if B % gradient_accumulation_steps != 0:
                raise ValueError("Global batch dimension must be divisible by gradient_accumulation_steps")
            else:
                return leaf.reshape((gradient_accumulation_steps, B // gradient_accumulation_steps) + leaf.shape[1:])

        if gradient_accumulation_steps == 1:
            (_, aux), grad = _minibatch_step(batch)
            new_module, new_opt = optimizer(grad, module)
            return new_module, new_opt, aux or {}

        grad_shapes, aux_shape = jax.eval_shape(_minibatch_step, 0)
        grad = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grad_shapes)
        aux = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), aux_shape)

        batch = jtu.tree_map(_stack_batch, batch) 

        (grad, aux), _ = jax.lax.scan(_scan_step, (grad, aux), batch, length = gradient_accumulation_steps)

        grad = jax.tree_util.tree_map(lambda g: g / gradient_accumulation_steps, grad)
        new_module, new_opt = optimizer(grad, module)
        return new_module, new_opt, aux or {}

    fn = eqx.filter_jit(_step) if jit else _step
    return fn

def make_eval_step(
  *,
  loss_function: _LossFn | None = None, 
  eval_step: _EvalStepCallable[tp.Sequence[_M]] | _EvalStepCallable[_M] | None = None,
  jit: bool = True,
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


def eval_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    eval_loader: tp.Iterable[tp.Any],
    eval_step: _EvalStepCallable[_ModuleInput],
    logger: Logger | None,
    callbacks: tp.Sequence[Callback],
    enable_wallclock: bool,
    key: PRNGKeyArray | None,
) -> EvalLoopResult:
    callbacks = list(callbacks)
    eval_step_counter = 0
    last_eval_logged = {}
    aggregated_aux = None

    def wallclock(name: str, step: int | None = None):
        if not enable_wallclock: 
            return nullcontext()

        ctx = rank_zero(logger.wc, name) if step is None else rank_zero(logger.wc, name, step=step)
        return ctx if ctx is not None else nullcontext()

    @rank_zero
    def _make_progress(desc: str, total: int | None):
        bar_format = None if total is not None else "{desc}: {n}"
        return tqdm(total=total, desc=desc, leave=False, bar_format=bar_format)

    eval_progress = _make_progress(
        "Train",
        len(eval_loader) if hasattr(eval_loader, "__len__") else None
    )

    try:
        eval_iter = iter(eval_loader)
    except Exception as e:
        raise RuntimeError("Exception creating evaluation data iterator") from e

    for callback in callbacks:
        method = getattr(callback, "on_evaluation_start", None)
        if method is None:
            continue
        callback_name = callback.__class__.__name__
        with wallclock(f"callbacks.{callback_name}.on_evaluation_start", eval_step_counter):
            method(
                module=module,
                optimizer=optimizer,
                step=eval_step_counter,
                logger=logger,
            )

    while True:
        with wallclock("eval/data.next", eval_step_counter + 1):
            try:
                batch = next(eval_iter)
            except StopIteration:
                break

        eval_step_counter += 1
        rank_zero(eval_progress.update, 1)

        key, eval_key = jax.random.split(key) if key is not None else (key, None)

        with wallclock("eval/step", eval_step_counter):
            aux = eval_step(module, batch, key=eval_key)

        aggregated_aux = (
            jax.tree_util.tree_map(jnp.add, aggregated_aux, aux)
            if aggregated_aux is not None
            else aux
        )

        if logger:
            reduced = rank_zero(
                logger.log_metrics,
                aux,
                step=eval_step_counter,
                tag="eval",
            )
            if reduced:
                last_eval_logged = reduced

        for callback in callbacks:
            method = getattr(callback, "on_eval_step", None)
            if method is None:
                continue
            callback_name = callback.__class__.__name__
            with wallclock(f"callbacks.{callback_name}.on_eval_step", eval_step_counter):
                method(
                    module=module,
                    optimizer=optimizer,
                    batch=batch,
                    step=eval_step_counter,
                    metrics=aux,
                    logger=logger,
                )

    for callback in callbacks:
        method = getattr(callback, "on_eval_end", None)
        if method is None:
            continue
        callback_name = callback.__class__.__name__
        with wallclock(f"callbacks.{callback_name}.on_eval_end", global_step):
            method(
                module=module,
                optimizer=optimizer,
                step=global_step,
                metrics=aggregated_aux,
                logger=logger,
            )

    if logger: 
        reduced = rank_zero(
            logger.log_metrics,
            aggregated_aux,
            step=global_step,
            tag="eval",
        )
        if reduced:
            last_eval_logged = reduced

    return last_eval_logged, aggregated_aux, key


def train_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    train_step: _TrainStepCallable[_ModuleInput, _OptimizerInput],
    train_loader: tp.Iterable[tp.Any],
    num_train_steps: int | None = None,
    *,
    logger: Logger | None = None,
    enable_wallclock = True,
    eval_loader: tp.Iterable[tp.Any] | None = None,
    eval_step: _EvalStepCallable[_ModuleInput] | None = None,
    num_eval_steps: int | None = None,
    eval_interval: int | None = None,
    callbacks: tp.Sequence[Callback] | None = None,
    eval_loop_fn: _EvalLoopCallable[_ModuleInput, _OptimizerInput] | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[_ModuleInput, _OptimizerInput, dict[str, float], dict[str, float]]:

    if eval_step is None and eval_loader is not None:
        raise ValueError("eval_step must be provided when eval_loader is not None")

    if eval_loader is None:
        eval_step = None

    callbacks = list(callbacks or [])
    eval_loop_callable = eval_loop_fn or eval_loop


    if logger is None:
        logger = rank_zero(
            TensorBoardLogger,
            log_dir="log/tensorboard",
            experiment_name="train",
            enabled=True,
        )


    train_step_counter = 0

    def wallclock(name: str, step: int | None = None):
        if not enable_wallclock: 
            return nullcontext()

        ctx = rank_zero(logger.wc, name) if step is None else rank_zero(logger.wc, name, step=step)
        return ctx if ctx is not None else nullcontext()

    @rank_zero
    def _make_progress(desc: str, total: int | None):
        bar_format = None if total is not None else "{desc}: {n}"
        return tqdm(total=total, desc=desc, leave=False, bar_format=bar_format)

    train_progress = _make_progress("Train", num_train_steps)

    for callback in callbacks:
        method = getattr(callback, "on_training_start", None)
        if method is None:
            continue
        callback_name = callback.__class__.__name__
        with wallclock(f"callbacks.{callback_name}.on_training_start", train_step_counter):
            method(
                module=module,
                optimizer=optimizer,
                step=train_step_counter,
                logger=logger,
            )

    last_train_logged: dict[str, float] = {}
    last_eval_logged: dict[str, float] = {}

    try:
        train_iter = iter(train_loader)
    except Exception as exc:
        raise RuntimeError("Exception creating data loader iterator") from exc

    try:
        while True:
            if num_train_steps is not None and train_step_counter >= num_train_steps:
                break

            train_step_counter += 1

            with wallclock("train/data.next", train_step_counter):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    LOGGER.info("Training data exhausted, ending training loop.")
                    break

            rank_zero(train_progress.update, 1)

            if key is not None:
                key, step_key = jax.random.split(key)
            else:
                step_key = None

            with wallclock("train/step", train_step_counter):
                module, optimizer, aux = train_step(module, optimizer, batch, key=step_key)

            for callback in callbacks:
                method = getattr(callback, "on_training_step", None)
                if method is None:
                    continue
                callback_name = callback.__class__.__name__
                with wallclock(f"callbacks.{callback_name}.on_training_step", train_step_counter):
                    method(
                        module=module,
                        optimizer=optimizer,
                        batch=batch,
                        step=train_step_counter,
                        metrics=aux,
                        logger=logger,
                    )

            if logger and getattr(logger, "enabled", True):
                reduced = rank_zero(
                    logger.log_metrics,
                    aux,
                    step=train_step_counter,
                    tag="train",
                )
                if reduced:
                    last_train_logged = reduced

            if (
                eval_step is not None
                and eval_interval is not None
                and eval_interval > 0
                and train_step_counter % eval_interval == 0
            ):
                if eval_loader is None:
                    raise ValueError("eval_step and eval_interval provided but eval_loader is None")
                eval_logged, _, key = eval_loop_callable(
                    module,
                    optimizer,
                    eval_loader=eval_loader,
                    eval_step=eval_step,
                    callbacks=callbacks,
                    logger=logger,
                    wallclock=wallclock,
                    key=key,
                    global_step=train_step_counter,
                )
                if eval_logged:
                    last_eval_logged = eval_logged

    except Exception:
        LOGGER.error("Exception during training loop", exc_info=True)
        raise
    finally:
        LOGGER.info("Training loop ended")
        rank_zero(train_progress.close)

    for callback in callbacks:
        method = getattr(callback, "on_training_end", None)
        if method is None:
            continue
        callback_name = callback.__class__.__name__
        with wallclock(f"callbacks.{callback_name}.on_training_end", train_step_counter):
            method(
                module=module,
                optimizer=optimizer,
                step=train_step_counter,
                logger=logger,
                metrics=last_train_logged,
            )

    if logger and hasattr(logger, "finalize"):
        rank_zero(logger.finalize)

    return module, optimizer, last_train_logged, last_eval_logged
