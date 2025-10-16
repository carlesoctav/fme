from __future__ import annotations

import logging
import time
import typing as tp
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
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
import time

from .filter import apply_transforms, iter_module
from .logger import Logger
from .metrics import SufficientMetric
from .utils import first_from, wallclock
from .callbacks import Callback
from .distributed import get_partition_spec, unbox_params


LOGGER = logging.getLogger("distributed_logger")


M = tp.TypeVar("_M", bound=eqx.Module)
O = tp.TypeVar("_O", bound="Optimizer")

GradTx = GradientTransformation | GradientTransformationExtraArgs
AxisSpec = bool | tp.Callable[[tp.Any], bool]
Wrt = PyTree[AxisSpec]
Aux = dict[str, tp.Any]
Loss = float
Batch = tp.Any

LossFn = tp.Callable[[M, O, Batch, PRNGKeyArray], tuple[Loss, Aux]]
ParallelismPlans = (
    dict[str, tp.Callable[[M], M]] | tp.Sequence[dict[str, tp.Callable[[M], M]]]
)


ModuleInput = tp.TypeVar("_ModuleInput", M, tp.Sequence[M])
OptimizerInput = tp.TypeVar("_OptimizerInput", O, tp.Sequence[O])


class TrainStepCallable(Protocol[ModuleInput, OptimizerInput]):
    def __call__(
        self,
        module: ModuleInput,
        optimizer: OptimizerInput,
        batch: tp.Any,
        key: PRNGKeyArray | None = None,
    ) -> tuple[ModuleInput, OptimizerInput, Aux]: ...


class EvalStepCallable(Protocol[ModuleInput, OptimizerInput]):
    def __call__(
        self,
        module: ModuleInput,
        optimizer: OptimizerInput,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Aux: ...


@dataclass
class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    wrt: PyTree[AxisSpec] = eqx.field(static=True)
    tx: GradTx = eqx.field(static=True)



    def __init__(
        self,
        model: M,
        tx: GradTx
        wrt: Wrt,
    ):
        self.tx = tx
        self.wrt = wrt
        self.opt_state= tx.init(eqx.filter(model, self.wrt))

    def __call__(
        self, grads: PyTree[Array], model: eqx.Module
    ) -> tuple[eqx.Module, Optimizer]:

        updates, opt_state = self.tx.update(
            grads,
            self.opt_state
        ) 
        new_model = eqx.apply_updates(model, updates)
        new_self = eqx.tree_at(lambda x: x.opt_state, self, opt_state)
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
    module: M,
    tx: GradTx,
    mesh: Mesh | None = None,
    wrt: Wrt = eqx.is_inexact_array,
    parallelism_plans: ParallelismPlans | None = None,
    *,
    key: PRNGKeyArray | None = None,
) -> tuple[M, Optimizer]:
    if not isinstance(module, eqx.Module):
        raise TypeError("module must be an equinox.Module instance")
    if not isinstance(
        tx, (GradientTransformation, GradientTransformationExtraArgs)
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

    need_init =  _module_has_abstract_params(module):

    def _build(
        m: M,
        rng: PRNGKeyArray,
    ) -> tuple[M, Optimizer]:

        if need_init:
            m = init_module(m, key=key)

        for plan in plans:
            m = apply_transforms(m, plan)

        m = unbox_params(m)
        opt = Optimizer(m, tx, wrt)

        return m_sharded, opt


    _build = eqx.filter_jit(_build)

    with mesh: 
        new_module, new_opt = _build(module, key)

    return new_module, new_opt


def make_train_step(
    loss_function: LossFn | None = None,
    train_step: TrainStepCallable[tp.Sequence[M], tp.Sequence[Optimizer]]
    | TrainStepCallable[M, Optimizer]
    | None = None,
    *,
    gradient_accumulation_steps: int = 1,
    jit: bool = True,
) -> TrainStepCallable[ModuleInput, OptimizerInput]:

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
        module: M,
        optimizer: Optimizer,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> tuple[M, Optimizer, Aux]:
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
    eval_step: EvalStepCallable[ModuleInput, OptimizerInput] | None = None
    loss_function: LossFn | None = None
    eval_metric: SufficientMetric | None = None
    jit: bool = True
    _compiled_eval_step: EvalStepCallable[ModuleInput, OptimizerInput] | None = (
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
        module: ModuleInput,
        optimizer: OptimizerInput,
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
    loss_function: LossFn | None = None,
    eval_step: EvalStepCallable[tp.Sequence[M], tp.Sequence[Optimizer]]
    | EvalStepCallable[M, Optimizer]
    | None = None,
    *,
    jit: bool = True,
) -> (
    EvalStepCallable[M, Optimizer]
    | EvalStepCallable[tp.Sequence[M], tp.Sequence[Optimizer]]
):
    if eval_step is None and loss_function is None:
        raise ValueError("Provide either eval_step or loss_function")

    if eval_step is not None:
        return eqx.filter_jit(eval_step) if jit else eval_step

    assert loss_function is not None

    def _step(
        module: M,
        optimizer: Optimizer,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Aux:
        _, aux = loss_function(module, optimizer, batch, key=key)
        return aux

    return eqx.filter_jit(_step) if jit else _step


def eval_loop(
    module: ModuleInput,
    optimizer: OptimizerInput,
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
        with wallclock(f"eval{eval_obj.name}", logger=logger):
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
    module: ModuleInput,
    optimizer: OptimizerInput,
    train_step_fn: TrainStepCallable[ModuleInput, OptimizerInput],
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
) -> tuple[ModuleInput, OptimizerInput, dict[str, tp.Any], dict[str, tp.Any]]:
    callbacks = list(callbacks or [])
    evals = list(evals or [])
    first_step = True

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
                if first_step:
                    timing = time.monotonic()
                    with jax.profiler.StepTraceAnnotation(
                        "train_step", step=train_step_idx
                    ):
                        module, optimizer, aux = train_step_fn(
                            module, optimizer, batch, key=step_key
                        )

                    diff = time.monotonic() - timing
                    first_step = False
                else:
                    with jax.profiler.StepTraceAnnotation(
                        "train_step", step=train_step_idx
                    ):
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


