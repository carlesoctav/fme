from __future__ import annotations

import logging
import time
import typing as tp
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
from jax.sharding import Mesh
from jaxtyping import Array, PRNGKeyArray, PyTree
from optax import GradientTransformation, GradientTransformationExtraArgs
from tqdm.auto import tqdm

from ._filter import apply_transforms, iter_module
from ._logger import Logger
from ._metrics import SufficientMetric
from ._utils import first_from, wallclock
from .callbacks import Callback
from .distributed import get_partition_spec


LOGGER = logging.getLogger("distributed_logger")


_M = tp.TypeVar("_M", bound=eqx.Module)
_O = tp.TypeVar("_O", bound="Optimizer")

_GradTx = GradientTransformation | GradientTransformationExtraArgs
_AxisSpec = bool | tp.Callable[[tp.Any], bool]
_Wrt = PyTree[_AxisSpec]
_Aux = dict[str, tp.Any]
_Loss = float
_Batch = tp.Any

_LossFn = tp.Callable[[_M, _O, _Batch, PRNGKeyArray], tuple[_Loss, _Aux]]
_ParallelismPlans = (
    dict[str, tp.Callable[[_M], _M]] | tp.Sequence[dict[str, tp.Callable[[_M], _M]]]
)


_ModuleInput = tp.TypeVar("_ModuleInput", _M, tp.Sequence[_M])
_OptimizerInput = tp.TypeVar("_OptimizerInput", _O, tp.Sequence[_O])


class _TrainStepCallable(Protocol[_ModuleInput, _OptimizerInput]):
    def __call__(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        batch: tp.Any,
        key: PRNGKeyArray | None = None,
    ) -> tuple[_ModuleInput, _OptimizerInput, _Aux]: ...


class _EvalStepCallable(Protocol[_ModuleInput, _OptimizerInput]):
    def __call__(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> _Aux: ...


class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    wrt: PyTree[_AxisSpec] = eqx.field(static=True)
    tx: GradientTransformationExtraArgs = eqx.field(static=True)

    def __init__(
        self, grad_tx: _GradTx, model: eqx.Module, *, wrt: _Wrt = eqx.is_inexact_array
    ):
        self.tx = grad_tx
        self.wrt = wrt
        self.opt_state = self.tx.init(eqx.filter(model, self.wrt))

    def __call__(
        self, grads: PyTree[Array], model: eqx.Module
    ) -> tuple[eqx.Module, Optimizer]:
        updates, opt_state = self.tx.update(
            grads, self.opt_state, eqx.filter(model, self.wrt)
        )
        new_model = eqx.apply_updates(model, updates)
        new_self = eqx.tree_at(lambda o: [o.opt_state], self, [opt_state])
        return new_model, new_self


_T = tp.TypeVar("_T")


def _split_batch_for_accum(batch: tp.Any, steps: int) -> tp.Any:
    def reshape(arr):
        if isinstance(arr, (jax.Array, np.ndarray)):
            if arr.shape[0] % steps != 0:
                raise ValueError(
                    "Global batch dimension must be divisible by gradient_accumulation_steps"
                )
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

        if (
            new_sub is None
            and hasattr(sub, "init_weights")
            and callable(getattr(sub, "init_weights"))
        ):
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


# TODO: maybe add AbstractModule and do typecheck instead of tree_map scan
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
    if not isinstance(
        grad_tx, (GradientTransformation, GradientTransformationExtraArgs)
    ):
        raise TypeError(
            "grad_tx must be an optax.GradientTransformation or GradientTransformationExtraArgs instance"
        )
    if key is None:
        raise ValueError("key must be provided for initialization")

    plans = (
        list(parallelism_plans or [])
        if isinstance(parallelism_plans, (list, tuple))
        else ([parallelism_plans] if parallelism_plans else [])
    )

    def _build(
        m: _M,
        rng: PRNGKeyArray,
    ) -> tuple[_M, Optimizer]:
        for plan in plans:
            m = apply_transforms(m, plan)

        pspec_tree = get_partition_spec(m)
        m_sharded = eqx.filter_shard(m, pspec_tree)
        opt = Optimizer(grad_tx, m_sharded, wrt=wrt)

        return m_sharded, opt

    if _module_has_abstract_params(module):
        module = init_module(module, key=key)

    build = eqx.filter_jit(_build)

    with mesh if mesh else nullcontext():
        new_module, new_opt = build(module, key)

    return new_module, new_opt


def make_train_step(
    loss_function: _LossFn | None = None,
    train_step: _TrainStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]
    | _TrainStepCallable[_M, Optimizer]
    | None = None,
    *,
    gradient_accumulation_steps: int = 1,
    jit: bool = True,
) -> (
    _TrainStepCallable[_M, Optimizer]
    | _TrainStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]
):
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
        def _minibatch_step(batch, step_key):
            (_, step_aux), step_grad = grad_fn(module, optimizer, batch, key=step_key)
            return step_grad, step_aux

        def _scan_step(carry, batch):
            grad, aux, scan_key = carry
            scan_key, step_key = jr.split(scan_key)
            step_grad, step_aux = _minibatch_step(batch, step_key)
            new_grad = jtu.tree_map(jnp.add, grad, step_grad)
            new_aux = jtu.tree_map(jnp.add, aux, step_aux)
            return (new_grad, new_aux, scan_key), None

        def _stack_batch(leaf):
            B = leaf.shape[0]
            if B % gradient_accumulation_steps != 0:
                raise ValueError(
                    "Global batch dimension must be divisible by gradient_accumulation_steps"
                )
            else:
                return leaf.reshape(
                    (gradient_accumulation_steps, B // gradient_accumulation_steps)
                    + leaf.shape[1:]
                )

        if gradient_accumulation_steps == 1:
            (_, aux), grad = grad_fn(module, optimizer, batch, key=key)
            new_module, new_opt = optimizer(grad, module)
            return new_module, new_opt, aux or {}
        else:
            grad_shapes, aux_shape = jax.eval_shape(_minibatch_step, 0, key)
            grad = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), grad_shapes)
            aux = jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), aux_shape)

            batch = jtu.tree_map(_stack_batch, batch)

            (grad, aux, _), _ = jax.lax.scan(
                _scan_step, (grad, aux, key), batch, length=gradient_accumulation_steps
            )

            grad = jax.tree_util.tree_map(
                lambda g: g / gradient_accumulation_steps, grad
            )
            new_module, new_opt = optimizer(grad, module)
            return new_module, new_opt, aux or {}

    fn = eqx.filter_jit(_step) if jit else _step
    return fn


@dataclass
class Eval:
    name: str
    dataset: tp.Iterable[tp.Any]
    eval_step: _EvalStepCallable[_ModuleInput, _OptimizerInput] | None = None
    loss_function: _LossFn | None = None
    eval_metric: SufficientMetric | None = None
    jit: bool = True
    _compiled_eval_step: _EvalStepCallable[_ModuleInput, _OptimizerInput] | None = (
        field(default=None, init=False, repr=False)
    )

    def __post_init__(self) -> None:
        if self.eval_step is not None and self.loss_function is not None:
            raise ValueError("provide either eval_step or loss_function, not both")
        if self.eval_step is None and self.loss_function is None:
            raise ValueError("eval requires eval_step or loss_function")

        if self.eval_step is None and self.loss_function is not None:
            self._compiled_eval_step = make_eval_step(
                loss_function=self.loss_function,
                jit=self.jit,
            )
        elif self.eval_step is not None:
            self._compiled_eval_step = (
                eqx.filter_jit(self.eval_step) if self.jit else self.eval_step
            )

    def run(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        logger: Logger | None = None,
        enable_wallclock: bool = True,
        train_step_idx: int | None = None,
        key: PRNGKeyArray | None = None,
    ) -> SufficientMetric:
        eval_step_idx = 0
        eval_metric = (
            self.eval_metric
            if self.eval_metric
            else SufficientMetric(name=self.name, log_every_n_step=None)
        )

        try:
            iterator = iter(self.dataset)
        except TypeError as e:
            raise RuntimeError("eval dataset is not iterable") from e

        progress_bar = tqdm(
            desc=f"Evaluating {self.name}",
            disable=jax.process_index() != 0,
            leave=False,
        )

        try:
            for batch in iterator:
                eval_step_idx += 1

                key, eval_key = (
                    jax.random.split(key, 2) if key is not None else (key, None)
                )

                aux = self._compiled_eval_step(
                    module,
                    optimizer,
                    batch,
                    key=eval_key,
                )

                progress_bar.update()

                eval_metric += aux

            logs = eval_metric.per_N_metrics(eval_step_idx, skip_check=True)
            logger.log(logs, train_step_idx)

            progress_bar.close()
            return eval_metric, logs
        except Exception as e:
            raise e


def make_eval_step(
    loss_function: _LossFn | None = None,
    eval_step: _EvalStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]
    | _EvalStepCallable[_M, Optimizer]
    | None = None,
    *,
    jit: bool = True,
) -> (
    _EvalStepCallable[_M, Optimizer]
    | _EvalStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]
):
    if eval_step is None and loss_function is None:
        raise ValueError("Provide either eval_step or loss_function")

    if eval_step is not None:
        return eqx.filter_jit(eval_step) if jit else eval_step

    assert loss_function is not None

    def _step(
        module: _M,
        optimizer: Optimizer,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> _Aux:
        _, aux = loss_function(module, optimizer, batch, key=key)
        return aux

    return eqx.filter_jit(_step) if jit else _step


def eval_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    evals: list[Eval],
    logger: Logger,
    callbacks: tp.Sequence[Callback] | None = None,
    train_step_idx: int | None = None,
    enable_wallclock: bool = False,
    *,
    key: PRNGKeyArray | None = None,
) -> dict[str, SufficientMetric]:
    callbacks = list(callbacks or [])
    eval_metrics: dict[str, SufficientMetric] = {}
    eval_logs: dict[str, float] = {}

    for callback in callbacks:
        method = getattr(callback, "on_validation_start", None)
        if method is not None:
            method(module, optimizer, logger, train_step_idx)

    for eval_obj in evals:
        with wallclock(f"eval{eval_obj.name}", logger =logger):
            eval_metric, logs = eval_obj.run(
                module,
                optimizer,
                key=key,
                logger=logger,
                enable_wallclock=enable_wallclock,
                train_step_idx=train_step_idx,
            )
            eval_metrics[eval_obj.name] = eval_metric
            eval_logs = {**eval_logs, **logs} 

    for callback in callbacks:
        method = getattr(callback, "on_validation_end", None)
        if method is not None:
            method(
                module,
                optimizer,
                eval_logs,
                logger,
                train_step_idx,
            )

    return eval_metrics


def train_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    train_step_fn: _TrainStepCallable[_ModuleInput, _OptimizerInput],
    train_loader: tp.Iterable[tp.Any],
    logger: Logger,
    train_metric: SufficientMetric | None = None,
    num_train_steps: int | None = None,
    enable_wallclock: bool = True,
    stop_train_wallclock_after_step: int = 1000,
    callbacks: tp.Sequence[Callback] | None = None,
    evals: tp.Sequence[Eval] | None = None,
    eval_interval: int | None = None,
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[_ModuleInput, _OptimizerInput, dict[str, tp.Any], dict[str, tp.Any]]:
    callbacks = list(callbacks or [])
    evals = list(evals or [])

    train_metric = (
        SufficientMetric(name="train", log_every_n_step=None)
        if train_metric is None
        else train_metric
    )

    if logger is None:
        raise ValueError("logger is required")

    train_step_idx = 0
    eval_metrics: dict[str, SufficientMetric] = {}

    try:
        train_iterator = iter(train_loader)
    except TypeError as e:
        raise RuntimeError("train_loader is not iterable") from e

    progress_bar = tqdm(
        total=num_train_steps,
        desc="Training",
        disable=jax.process_index() != 0,
        leave=False,
        position=0,
    )

    for cb in callbacks:
        method = getattr(cb, "on_training_start", None)
        if method is not None:
            method(module, optimizer, logger)

    try:
        while num_train_steps is None or train_step_idx < num_train_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                print("DEBUGPRINT[5]: _training.py:547 (after except StopIteration:)")
                LOGGER.info("Train data loader exhausted, ending training loop.")
                break

            train_step_idx += 1

            key, step_key = jax.random.split(key, 2) if key is not None else (key, None)

            with wallclock(
                "train_step",
                logger,
                train_step_idx,
                noop=train_step_idx > stop_train_wallclock_after_step
                or not enable_wallclock,
            ):
                with jax.profiler.StepTraceAnnotation("train_step", step = train_step_idx):
                    module, optimizer, aux = train_step_fn(
                        module, optimizer, batch, key=step_key
                    )

            progress_bar.update()

            aux = aux or {}
            train_metric += aux

            step_metrics = train_metric.step_metrics()
            per_N_metrics = train_metric.per_N_metrics(step=train_step_idx)

            logs = {**step_metrics, **per_N_metrics}

            logger.log(logs, step=train_step_idx)

            for cb in callbacks:
                method = getattr(cb, "on_training_step", None)
                if method is not None:
                    method(module, optimizer, batch, logs, logger, train_step_idx)

            should_eval = (
                eval_interval is not None
                and eval_interval > 0
                and train_step_idx % eval_interval == 0
            )

            if should_eval:
                key, eval_key = (
                    jax.random.split(key, 2) if key is not None else (key, None)
                )
                eval_metrics = eval_loop(
                    module,
                    optimizer,
                    evals,
                    logger,
                    callbacks,
                    train_step_idx,
                    enable_wallclock,
                    key=eval_key,
                )

    except Exception:
        LOGGER.error("Exception during training loop", exc_info=True)
        raise
    finally:
        LOGGER.info("Training loop ended")

    for cb in callbacks:
        method = getattr(cb, "on_training_end", None)
        if method is not None:
            method(module, optimizer, logs, logger, train_step_idx)

    return module, optimizer, train_metric, eval_metrics if evals else {}


def benchmark_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    train_step_fn: _TrainStepCallable[_ModuleInput, _OptimizerInput],
    train_loader: tp.Iterable[tp.Any],
    logger: Logger,
    num_steps: int = 100,
    theoretical_flops_per_step: float | None = None,
    trace_steps: tuple[int, int] | None = None,
    trace_dir: str = "./benchmark_traces",
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[_ModuleInput, _OptimizerInput, dict[str, tp.Any]]:
    import time
    
    if logger is None:
        raise ValueError("logger is required")
    step_idx = 0
    train_step_times = []
    next_batch_times = []
    try:
        train_iterator = iter(train_loader)
    except TypeError as e:
        raise RuntimeError("train_loader is not iterable") from e
    progress_bar = tqdm(
        total=num_steps,
        desc="Benchmarking",
        disable=jax.process_index() != 0,
        leave=True,
    )
    try:
        while step_idx < num_steps:
            batch_start = time.perf_counter()
            try:
                batch = next(train_iterator)
            except StopIteration:
                LOGGER.info("Train data loader exhausted, ending benchmark loop.")
                break
            batch_end = time.perf_counter()
            step_idx += 1
            if (
                trace_steps is not None
                and step_idx == trace_steps[0]
                and jax.process_index() == 0
            ):
                from pathlib import Path
                trace_path = Path(trace_dir)
                trace_path.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Starting JAX profiler trace at step {step_idx}")
                jax.profiler.start_trace(str(trace_path))
            key, step_key = jax.random.split(key, 2) if key is not None else (key, None)
            step_start = time.monotonic()
            with jax.profiler.StepTraceAnnotation("train_step", step = step_idx):
                module, optimizer, aux = train_step_fn(
                    module, optimizer, batch, key=step_key
                )
            jtu.tree_map(
                lambda x: x.block_until_ready()
                if hasattr(x, "block_until_ready")
                else x,
                module,
            )
            jtu.tree_map(
                lambda x: x.block_until_ready()
                if hasattr(x, "block_until_ready")
                else x,
                optimizer,
            )
            step_end = time.monotonic()

            if (
                trace_steps is not None
                and step_idx == trace_steps[1]
                and jax.process_index() == 0
            ):
                LOGGER.info(f"Stopping JAX profiler trace at step {step_idx}")
                jax.profiler.stop_trace()
            train_step_times.append(step_end - step_start)
            next_batch_times.append(batch_end - batch_start)
            progress_bar.update()
        progress_bar.close()
        train_step_times = np.array(train_step_times)
        next_batch_times = np.array(next_batch_times)
        if len(train_step_times) == 0:
            LOGGER.warning("No timing data collected (all steps were skipped)")
            return module, optimizer, {}
        for i, t in enumerate(train_step_times, ):
            logger.log({f"benchmark/train_step_time": t}, step=i)
        for i, t in enumerate(next_batch_times):
            logger.log({f"benchmark/next_batch_time": t}, step=i)
        if theoretical_flops_per_step is not None:
            for i, step_time in enumerate(train_step_times):
                measured_flops_per_sec = (
                    theoretical_flops_per_step / step_time if step_time > 0 else 0
                )
                logger.log({f"benchmark/flops_per_sec": measured_flops_per_sec}, step=i)
                logger.log(
                    {
                        f"benchmark/mfu": measured_flops_per_sec
                        / theoretical_flops_per_step
                        if theoretical_flops_per_step > 0
                        else 0
                    },
                    step=i,
                )
        for i, step_time in enumerate(train_step_times):
            batches_per_sec = 1.0 / step_time if step_time > 0 else 0
            logger.log({f"benchmark/batches_per_sec": batches_per_sec}, step=i)
        if jax.process_index() == 0:
            LOGGER.info("=" * 30)
            LOGGER.info(" Benchmark Results ".center(30, "="))
            LOGGER.info("=" * 30)
            LOGGER.info(f"Train Step Time (avg): {train_step_times.mean():.4f}s")
            LOGGER.info(f"Train Step Time (median): {np.median(train_step_times):.4f}s")
            LOGGER.info(f"Train Step Time (std): {train_step_times.std():.4f}s")
            LOGGER.info(f"Next Batch Time (avg): {next_batch_times.mean():.4f}s")
            LOGGER.info(f"Batches/sec (avg): {1.0 / train_step_times.mean():.2f}")
            if theoretical_flops_per_step is not None:
                avg_flops = theoretical_flops_per_step / train_step_times.mean()
                LOGGER.info(f"FLOPs/sec (avg): {avg_flops:.2e}")
                LOGGER.info(f"MFU (avg): {avg_flops / theoretical_flops_per_step:.4f}")
            LOGGER.info("=" * 30)
        stats = {
            "train_step_time_mean": float(train_step_times.mean()),
            "train_step_time_median": float(np.median(train_step_times)),
            "train_step_time_std": float(train_step_times.std()),
            "next_batch_time_mean": float(next_batch_times.mean()),
            "batches_per_sec": float(1.0 / train_step_times.mean()),
        }
        if theoretical_flops_per_step is not None:
            avg_flops = theoretical_flops_per_step / train_step_times.mean()
            stats["flops_per_sec"] = float(avg_flops)
            stats["mfu"] = float(avg_flops / theoretical_flops_per_step)
    except Exception:
        LOGGER.error("Exception during benchmark loop", exc_info=True)
        raise
    finally:
        LOGGER.info("Benchmark loop ended")
    eturn module, optimizer, stats
