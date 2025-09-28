from __future__ import annotations

import logging
import typing as tp
from contextlib import nullcontext
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
from ._profiler import JaxProfiler
from ._utils import first_from
from .callbacks import Callback, CallbackManager
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


def _ensure_manager(
    callbacks: CallbackManager | tp.Sequence[Callback] | None,
) -> CallbackManager | None:
    if callbacks is None:
        return None
    if isinstance(callbacks, CallbackManager):
        return callbacks
    return CallbackManager(list(callbacks))

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
    print(f"DEBUGPRINT[342]: _training.py:316: gradient_accumulation_steps={gradient_accumulation_steps}")

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

@tp.overload
def make_eval_step(
  *,
  loss_function: _LossFn, 
  jit: bool = True,
) -> _EvalStepCallable[_M]:
  ...

@tp.overload
def make_eval_step(
  *,
  eval_step: _EvalStepCallable[_M],
  jit: bool = True,
) -> _EvalStepCallable[_M]:
  ...

@tp.overload
def make_eval_step(
  *,
  eval_step: _EvalStepCallable[tp.Sequence[_M]],
  jit: bool = True,
) -> _EvalStepCallable[tp.Sequence[_M]]:
  ...

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

def train_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    train_step: _TrainStepCallable[_ModuleInput, _OptimizerInput],
    train_loader: tp.Iterable[tp.Any],
    num_train_steps: int | None = None,
    *,
    logger: Logger | None = None,
    log_wc: bool = True,
    enable_wallclock = True,
    eval_loader: tp.Iterable[tp.Any] | None = None,
    eval_step: _EvalStepCallable[_ModuleInput] | None = None,
    num_eval_steps: int | None = None,
    eval_interval: int | None = None,
    callbacks: CallbackManager | tp.Sequence[Callback] | None = None,
    profiler: JaxProfiler | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[_ModuleInput, _OptimizerInput, dict[str, float], dict[str, float]]:

    if eval_step is None and eval_loader is not None:
        raise ValueError("eval_step must be provided when eval_loader is not None")

    if eval_loader is None:
        eval_step = None

    callback_manager = _ensure_manager(callbacks)

    if logger is None:
        logger = TensorBoardLogger(
            log_dir="log/tensorboard",
            experiment_name="train",
            enabled=jax.process_index() == 0,
        )

    train_step_counter = 0
    prof = profiler if profiler is not None and profiler.enabled() else None
    wc = logger.wc  if log_wc else nullcontext

    def _make_progress(desc: str, total: int | None):
        show_progress = jax.process_index() == 0
        if not show_progress:
            return None
        bar_format = None if total is not None else "{desc}: {n}"
        return tqdm(total=total, desc=desc, leave=False, bar_format=bar_format)
    train_progress = _make_progress("Train", num_train_steps)

    def _callback(
        event: str,
        step: int | None = None,
        **kwargs,
    ) -> None:
        if callback_manager is None:
            return
        with wc(f"callbacks.{event}", step=step):
            callback_manager.call(event, **kwargs, step = step)

    _callback(
        "on_training_start",
        step=train_step_counter,
        module=module,
        optimizer=optimizer,
    )

    last_train_logged: dict[str, float] = {}
    last_eval_logged: dict[str, float] = {}

    try:
        train_iter = iter(train_loader)
        eval_iter = iter(eval_loader) if eval_loader else None
    except Exception as e:
        raise RuntimeError("Exception creating data loader iterator") from e  

    try:
        while True:
            if num_train_steps is not None and train_step_counter >= num_train_steps:
                break


            train_step_counter += 1 
            with wc("train/data.next", step=train_step_counter):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    LOGGER.info("Training data exhausted, ending training loop.")
                    break


            if train_progress is not None:
                train_progress.update(1)

            key, step_key = jax.random.split(key, 2) if key is not None else (key, None)


            if prof is not None:
                prof.maybe_start_train(train_step_counter, metadata={"phase": "train"})

            with wc("train/step", step=train_step_counter):
                module, optimizer, aux = train_step(module, optimizer, batch, key=step_key)

            _callback(
                "on_training_step",
                module=module,
                optimizer=optimizer,
                batch=batch,
                logger = logger,
                step=train_step_counter,
                aux=aux,
            )

            if prof is not None:
                prof.maybe_stop_train(train_step_counter, block_on=aux)

            if logger and getattr(logger, "enabled", True):
                last_train_logged = logger.log_metrics(aux, train_step_counter, tag = "train")

            if (
                eval_step is not None
                and eval_interval is not None
                and eval_interval > 0
                and train_step_counter % eval_interval == 0
            ):
                if eval_loader is None:
                    raise ValueError("eval_step and eval_interval provided but eval_loader is None")

                if prof is not None:
                    prof.on_eval_start(force_stop=True)

                _callback(
                    "on_evaluation _start",
                    mode="eval",
                    module=module,
                    optimizer=optimizer,
                    step=train_step_counter,
                )

                eval_progress = _make_progress("Eval", num_eval_steps)
                eval_step_counter = 0

                try:
                    while True:
                        if num_eval_steps is not None and eval_step_counter >= num_eval_steps:
                            break

                        if eval_iter is None:
                            raise ValueError("eval_loader must be provided for evaluation")

                        eval_step_counter = eval_step_counter + 1
                        with wc("data.eval.next", step=eval_step_counter):
                            try:
                                eval_batch = next(eval_iter)
                            except StopIteration:
                                break

                        if eval_progress is not None:
                            eval_progress.update(1)

                        key, eval_key = jax.random.split(key) if key is not None else (key, None)

                        if prof is not None:
                            prof.maybe_start_eval(
                                eval_step_counter,
                                metadata={"global_step": train_step_counter},
                            )

                        with wc("eval/step", step=eval_step_counter):
                            eval_aux = eval_step(module, eval_batch, key=eval_key)

                        agg_eval_aux = jax.tree_util.tree_map(
                            jnp.add,
                            agg_eval_aux,
                            eval_aux
                        ) if agg_eval_aux is not None else eval_aux

                        if prof is not None:
                            prof.maybe_stop_eval(eval_step_counter, block_on=eval_aux)

                        if logger and getattr(logger, "enabled", True):
                            last_eval_logged = logger.log_metrics(eval_aux, eval_step_counter, tag = "eval")

                        _callback(
                            "on_evaluation_step",
                            module=module,
                            optimizer=optimizer,
                            batch=eval_batch,
                            logger = logger,
                            step=eval_step_counter,
                            aux=eval_aux,
                        )
                finally:
                    if eval_progress is not None:
                        eval_progress.close()

                if logger and getattr(logger, "enabled", True) and agg_eval_aux is not None:
                    logger.log_metrics(
                        agg_eval_aux,
                        train_step_counter,
                        tag = "eval",
                    )

                _callback(
                    "on_evaluation_end",
                    module=module,
                    optimizer=optimizer,
                    metrics=last_eval_logged,
                )

                if prof is not None:
                    prof.stop()

    except Exception:
        LOGGER.error("Exception during training loop", exc_info=True)
        raise
    finally:
        LOGGER.info("Training loop ended")
        if train_progress is not None:
            train_progress.close()

    _callback(
        "on_training_end",
        module=module,
        optimizer=optimizer,
        step=train_step_counter,
    )

    if (
        wc is not None
        and logger
        and getattr(logger, "enabled", True)
        and hasattr(logger, "log_histogram")
    ):
        summary = wc.summary()
        for key in summary:
            if "." in key:
                mode, name = key.split(".", 1)
            else:
                mode, name = None, key
            samples = wc.histogram(name, mode=mode)
            if samples:
                logger.log_histogram(
                    f"time.{name}",
                    samples,
                    step=train_step_counter,
                    mode=mode or "setup",
                )

    if logger and hasattr(logger, "finalize"):
        logger.finalize()

    return module, optimizer, last_train_logged, last_eval_logged
