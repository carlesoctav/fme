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
from ._wallclock import ProgramWallClock
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


def _combine_aux(acc: tp.Any | None, aux: tp.Any) -> tp.Any:
    if aux is None:
        return acc
    if acc is None:
        return aux
    if isinstance(aux, dict):
        result: dict[str, tp.Any] = {}
        for key, value in aux.items():
            acc_value = acc.get(key) if isinstance(acc, dict) else None
            result[key] = _combine_aux(acc_value, value)
        return result
    if isinstance(aux, tuple) and isinstance(acc, tuple):
        return tuple(_combine_aux(a, b) for a, b in zip(acc, aux))
    return acc + aux


def _finalize_aux(aux: tp.Any, steps: int) -> tp.Any:
    if aux is None:
        return {}
    if isinstance(aux, dict):
        return {k: _finalize_aux(v, steps) for k, v in aux.items()}
    if isinstance(aux, tuple):
        return tuple(_finalize_aux(v, steps) if isinstance(v, dict) else v for v in aux)
    return aux / steps


def _split_batch_for_accum(batch: tp.Any, steps: int) -> tp.Any:
    def reshape(arr):
        if isinstance(arr, (jax.Array, np.ndarray)):
            if arr.shape[0] % steps != 0:
                raise ValueError("Global batch dimension must be divisible by gradient_accumulation_steps")
            new_shape = (steps, arr.shape[0] // steps) + arr.shape[1:]
            return arr.reshape(new_shape)
        return arr

    return jax.tree_util.tree_map(reshape, batch)


def _select_microbatch(batched: tp.Any, index: int) -> tp.Any:
    return jax.tree_util.tree_map(lambda x: x[index], batched)


def _to_host(value: tp.Any) -> tp.Any:
    if isinstance(value, dict):
        return {k: _to_host(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_host(v) for v in value)
    if isinstance(value, jax.Array):
        host = jax.device_get(value)
        if np.isscalar(host):
            return float(host)
        if hasattr(host, "item") and host.shape == ():
            return float(host.item())
        return host
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover - defensive
            return value
    return value


def _flatten_metrics(metrics: dict[str, tp.Any], prefix: str = "") -> dict[str, tp.Any]:
    flat: dict[str, tp.Any] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, name))
        else:
            flat[name] = value
    return flat


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
    wall_clock: ProgramWallClock | None = None,
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
        measurement = wall_clock.measure("module.build", mode="setup") if wall_clock else nullcontext()
        with measurement:
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


        grad_shapes, aux_shape = jax.eval_shape(_minibatch_step, 0)
        grad = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grad_shapes)
        aux = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), aux_shape)

        batch = jtu.tree_map(_stack_batch, batch) 

        (grad, aux), _ = jax.lax.scan(_scan_step, (grad, aux), batch, length = gradient_accumulation_steps)

        grad = jax.tree_util.tree_map(lambda g: g / gradient_accumulation_steps, grad)
        new_module, new_opt = optimizer(grad, module)
        return new_module, new_opt, aux or {}

    setattr(_step, "_gradient_accumulation_steps", gradient_accumulation_steps)
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
    eval_loader: tp.Iterable[tp.Any] | None = None,
    eval_step: _EvalStepCallable[_ModuleInput] | None = None,
    num_eval_steps: int | None = None,
    eval_interval: int | None = None,
    callbacks: CallbackManager | tp.Sequence[Callback] | None = None,
    wall_clock: ProgramWallClock | None = None,
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
    wc = wall_clock if wall_clock is not None and wall_clock.enabled() else None
    prof = profiler if profiler is not None and profiler.enabled() else None

    def _measure(
        name: str,
        *,
        mode: str | None = None,
        step: int | None = None,
        metadata: dict[str, tp.Any] | None = None,
    ):
        if wc is None:
            return nullcontext()
        return wc.measure(name, mode=mode, step=step, metadata=metadata)

    def _collect(mode: str) -> dict[str, float]:
        if wc is None:
            return {}
        collected = wc.collect_pending_metrics(mode)
        cleaned: dict[str, float] = {}
        for key, value in collected.items():
            parts = key.split("/", 2)
            if len(parts) == 3:
                _, _, remainder = parts
            else:
                remainder = parts[-1]
            cleaned[remainder.replace("/", ".")] = value
        return cleaned

    def _callback(
        event: str,
        *,
        mode: str,
        timer_step: int | None = None,
        eval_step_index: int | None = None,
        **kwargs,
    ) -> None:
        if callback_manager is None:
            return
        metadata = {"eval_step": eval_step_index} if eval_step_index is not None else None
        with _measure(f"callbacks.{event}", mode=mode, step=timer_step, metadata=metadata):
            callback_manager.call(event, **kwargs)

    def _prepare_metrics(aux: tp.Any) -> dict[str, tp.Any]:
        if aux is None:
            return {}
        host = _to_host(aux)
        if isinstance(host, dict):
            return _flatten_metrics(host)
        return {"value": host}

    def _log(mode: str, metrics: dict[str, tp.Any], step: int) -> None:
        if not logger or not getattr(logger, "enabled", True):
            return
        logger.log_metrics(metrics, step=step, mode=mode)

    if callback_manager:
        _callback(
            "on_training_start",
            mode="setup",
            timer_step=train_step_counter,
            module=module,
            optimizer=optimizer,
            step=train_step_counter,
        )

    with _measure("iterators", mode="setup"):
        print(f"DEBUGPRINT[341]: _training.py:525: train_loader={train_loader}")
        train_iter = iter(train_loader)
        eval_iter = iter(eval_loader) if eval_loader is not None else None

    show_progress = jax.process_index() == 0

    def _make_progress(desc: str, total: int | None):
        if not show_progress:
            return None
        bar_format = None if total is not None else "{desc}: {n}"
        return tqdm(total=total, desc=desc, leave=False, bar_format=bar_format)

    train_progress = _make_progress("Train", num_train_steps)

    last_train_logged: dict[str, float] = {}
    last_eval_logged: dict[str, float] = {}

    try:
        while True:
            if num_train_steps is not None and train_step_counter >= num_train_steps:
                break

            next_step = train_step_counter + 1

            with _measure("data", mode="train", step=next_step):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    LOGGER.info("Training data exhausted, ending training loop.")
                    break

            train_step_counter = next_step

            if train_progress is not None:
                train_progress.update(1)

            key, step_key = jax.random.split(key, 2) if key is not None else (key, None)

            _callback(
                "on_train_step",
                mode="train",
                timer_step=train_step_counter,
                module=module,
                optimizer=optimizer,
                batch=batch,
                step=train_step_counter,
                key=step_key,
            )

            if prof is not None:
                prof.maybe_start_train(train_step_counter, metadata={"phase": "train"})

            with _measure("step", mode="train", step=train_step_counter):
                module, optimizer, aux = train_step(module, optimizer, batch, key=step_key)

            if prof is not None:
                prof.maybe_stop_train(train_step_counter, block_on=aux)

            metrics = _prepare_metrics(aux)
            timings = _collect("train")
            if timings:
                metrics.update({f"time.{name}": value for name, value in timings.items()})

            _log("train", metrics, train_step_counter)
            last_train_logged = metrics

            _callback(
                "on_train_step_end",
                mode="train",
                timer_step=train_step_counter,
                module=module,
                optimizer=optimizer,
                batch=batch,
                step=train_step_counter,
                aux=aux,
                metrics=metrics,
            )

            if (
                eval_step is not None
                and eval_interval is not None
                and eval_interval > 0
                and train_step_counter % eval_interval == 0
            ):
                if prof is not None:
                    prof.on_eval_start(force_stop=True)

                _callback(
                    "on_eval_start",
                    mode="eval",
                    timer_step=train_step_counter,
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

                        next_eval_step = eval_step_counter + 1
                        with _measure("data", mode="eval", step=next_eval_step):
                            try:
                                eval_batch = next(eval_iter)
                            except StopIteration:
                                break

                        eval_step_counter = next_eval_step

                        if eval_progress is not None:
                            eval_progress.update(1)

                        key, eval_key = jax.random.split(key) if key is not None else (key, None)

                        _callback(
                            "on_eval_step",
                            mode="eval",
                            timer_step=eval_step_counter,
                            eval_step_index=eval_step_counter,
                            module=module,
                            optimizer=optimizer,
                            batch=eval_batch,
                            step=train_step_counter,
                            key=eval_key,
                        )

                        if prof is not None:
                            prof.maybe_start_eval(
                                eval_step_counter,
                                metadata={"global_step": train_step_counter},
                            )

                        with _measure("step", mode="eval", step=eval_step_counter):
                            eval_aux = eval_step(module, eval_batch, key=eval_key)

                        if prof is not None:
                            prof.maybe_stop_eval(eval_step_counter, block_on=eval_aux)

                        metrics = _prepare_metrics(eval_aux)
                        timings = _collect("eval")
                        if timings:
                            metrics.update({f"time.{name}": value for name, value in timings.items()})

                        _log("eval", metrics, eval_step_counter)
                        last_eval_logged = metrics

                        _callback(
                            "on_eval_step_end",
                            mode="eval",
                            timer_step=eval_step_counter,
                            eval_step_index=eval_step_counter,
                            module=module,
                            optimizer=optimizer,
                            batch=eval_batch,
                            step=train_step_counter,
                            aux=eval_aux,
                            metrics=metrics,
                        )
                finally:
                    if eval_progress is not None:
                        eval_progress.close()

                _callback(
                    "on_eval_end",
                    mode="eval",
                    timer_step=train_step_counter,
                    module=module,
                    optimizer=optimizer,
                    step=train_step_counter,
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

    if callback_manager:
        _callback(
            "on_training_end",
            mode="setup",
            timer_step=train_step_counter,
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
