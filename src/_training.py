from __future__ import annotations

import logging
import typing as tp
from contextlib import nullcontext
from dataclasses import dataclass, field
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
from ._metrics import SufficientMetric


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


class _EvalStepCallable(Protocol[_ModuleInput, _OptimizerInput]):
    def __call__(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> _Aux:
        ...



@dataclass
class Eval:
    name: str
    dataset: tp.Iterable[tp.Any] | None = None
    load_dataset_fn: tp.Callable[[], tp.Iterable[tp.Any]] | None = None
    unload_dataset_fn: tp.Callable[[tp.Iterable[tp.Any]], None] | None = None
    eval_step: _EvalStepCallable[_ModuleInput, _OptimizerInput] | None = None
    loss_function: _LossFn | None = None
    enable_wallclock: bool | None = None
    reduce_fn: tp.Callable[[tp.Any, int], tp.Any] | None = None
    jit: bool = True

    _compiled_eval_step: _EvalStepCallable[_ModuleInput, _OptimizerInput] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.dataset is not None and self.load_dataset_fn is not None:
            raise ValueError("Provide either dataset or load_dataset_fn, not both")
        if self.eval_step is not None and self.loss_function is not None:
            raise ValueError("Provide either eval_step or loss_function, not both")
        if self.eval_step is None and self.loss_function is None:
            raise ValueError("Eval requires eval_step or loss_function")

        if self.eval_step is None and self.loss_function is not None:
            self._compiled_eval_step = make_eval_step(
                loss_function=self.loss_function,
                jit=self.jit,
            )
        elif self.eval_step is not None:
            self._compiled_eval_step = (
                eqx.filter_jit(self.eval_step) if self.jit else self.eval_step
            )

    def on_eval_start(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        *,
        logger: Logger | None = None,
    ) -> None:
        del module, optimizer, logger

    def on_eval_end(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        logs: tp.Any,
        *,
        logger: Logger | None = None,
    ) -> None:
        del module, optimizer, logs, logger

    def on_eval_step_start(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        batch: tp.Any,
        step: int,
        *,
        logger: Logger | None = None,
    ) -> None:
        del module, optimizer, batch, step, logger

    def on_eval_step_end(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        batch: tp.Any,
        aux: tp.Any,
        logs: tp.Any,
        step: int,
        *,
        logger: Logger | None = None,
        callback: list[Callback] | None = None
    ) -> None:
        del module, optimizer, batch, logs, step, logger

    def _resolve_eval_step(self) -> _EvalStepCallable[_ModuleInput, _OptimizerInput]:
        if self._compiled_eval_step is None:
            raise RuntimeError("Eval step not initialised")
        return self._compiled_eval_step

    def _resolve_dataset(self) -> tuple[tp.Iterable[tp.Any], tp.Any]:
        if self.dataset is not None:
            return self.dataset, None
        assert self.load_dataset_fn is not None
        loaded = self.load_dataset_fn()
        return loaded, loaded

    def eval(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        logger: Logger, 
        callback: list[callback],
        enable_wallclock: bool = True,
        global_step_idx: int | None = None,
        key: PRNGKeyArray | None = None,
    ) -> tuple[dict[str, float]]:

        eval_step_fn = self._resolve_eval_step()
        dataset, handle = self._resolve_dataset()
        aggregated = None
        step_idx = 0
        eval_metrics = SufficientMetric(name = self.name, log_every_n_steps=None, reduce_fn=self.reduce_fn)

        def wallclock(name: str, idx: int | None = None):
            use_wallclock = self.enable_wallclock if self.enable_wallclock is not None else enable_wallclock
            if not use_wallclock or eval_logger is None:
                return nullcontext()
            if idx is None:
                ctx = rank_zero(eval_logger.wc, name)
            else:
                ctx = rank_zero(eval_logger.wc, name, idx)
            return ctx if ctx is not None else nullcontext()

        try:
            iterator = iter(dataset)
        except TypeError as exc:
            raise RuntimeError("Eval dataset is not iterable") from exc

        try:
            self.on_eval_start(module, optimizer, logger=logger, callback = callback)

            for batch in iterator:
                step_idx += 1

                self.on_eval_step_start(
                    module,
                    optimizer,
                    batch,
                    step = step_idx,
                    logger =logger,
                    callback = callback,
                )

                key, eval_key = jax.random.split(key, 2) if key is not None else (key, None)

                with wallclock(f"eval.{self.name}.step", step_idx):
                    aux = eval_step_fn(
                        module,
                        optimizer,
                        batch,
                        key=eval_key,
                    )

                eval_metrics += aux
                logs = eval_metrics.step_metrics()
                rank_zero(logger.log, self.name, logs, step=step_idx)

                self.on_eval_step_end(
                    module,
                    optimizer,
                    batch,
                    aux,
                    logs,
                    step = step_idx,
                    logger=logger,
                    callback = callback,
                )


            logs = eval_metrics.per_N_metrics(step_idx, skip_check=True)
            rank_zero(logger.log, self.name, logs, step_idx)

            self.on_eval_end(
                module,
                optimizer,
                aux,
                logs,
                logger=logger,
                callback = callback,
            )

            return eval_metrics, reduce
        finally:
            if self.unload_dataset_fn is not None and handle is not None:
                self.unload_dataset_fn(handle)

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
            (_, aux), grad = grad_fn(module, optimizer, batch, key=key)
            new_module, new_opt = optimizer(grad, module)
            return new_module, new_opt, aux or {}
        else:
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
    eval_step: _EvalStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]
    | _EvalStepCallable[_M, Optimizer]
    | None = None,
    jit: bool = True,
) -> _EvalStepCallable[_M, Optimizer] | _EvalStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]:
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


@tp.overload
def eval_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    *,
    eval_loader: tp.Iterable[tp.Any],
    eval_step: _EvalStepCallable[_ModuleInput, _OptimizerInput],
    evals: None = None,
    logger: Logger | None,
    callbacks: tp.Sequence[Callback],
    enable_wallclock: bool,
    key: PRNGKeyArray | None,
    global_step_idx: int | None = None,
) -> SufficientMetric:
    ...

@tp.overload
def eval_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    *,
    eval_loader: None = None,
    eval_step: None = None,
    evals: tp.Sequence[Eval],
    logger: Logger | None,
    callbacks: tp.Sequence[Callback],
    enable_wallclock: bool,
    key: PRNGKeyArray | None,
    global_step_idx: int | None = None,
) -> dict[str, SufficientMetric]:
    ...
def eval_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    *,
    eval_loader: tp.Iterable[tp.Any] | None = None,
    eval_step: _EvalStepCallable[_ModuleInput, _OptimizerInput] | None = None,
    evals: tp.Sequence[Eval] | None = None,
    logger: Logger | None,
    callbacks: tp.Sequence[Callback],
    enable_wallclock: bool,
    key: PRNGKeyArray | None,
    global_step: int | None = None,
) -> SufficientMetric | dict[str, SufficientMetric]:
    callbacks_list = list(callbacks)

    if evals is not None and (eval_loader is not None or eval_step is not None):
        raise ValueError("Provide either evals or eval_loader/eval_step, not both")

    def wallclock(name: str, idx: int | None = None):
        if not enable_wallclock or logger is None:
            return nullcontext()
        ctx = rank_zero(logger.wc, name) if idx is None else rank_zero(logger.wc, name, idx)
        return ctx if ctx is not None else nullcontext()

    if evals:
        eval_logs: dict[str, tp.Any] = {}
        eval_metrics: dict[str, SufficientMetric] = {}
        for callback in callbacks_list:
            method = getattr(callback, "on_validation_start", None)
            if method is not None:
                method(module, optimizer, global_step)

        for eval_obj in evals:
            eval_aux, eval_log, eval_metric = eval_obj.eval(
                module,
                optimizer,
                key=key,
                logger=logger,
                enable_wallclock=enable_wallclock,
                global_step_idx=global_step,
            )
            eval_logs[eval_obj.name] = eval_log
            eval_metrics[eval_obj.name] = eval_metric

        for callback in callbacks_list:
            method = getattr(callback, "on_validation_end", None)
            if method is not None:
                method(
                    module,
                    optimizer,
                    eval_aux,
                    eval_logs,
                    global_step,
                )

        return eval_metrics

    elif eval_loader is None or eval_step is None:
        return {}, None
    else:
        eval_step = 0
        aggregated_aux = None
        eval_metric = SufficientMetric(name = "eval" log_every_n_steps=None)

        @rank_zero
        def _make_progress(desc: str, total: int | None):
            bar_format = None if total is not None else "{desc}: {n}"
            return tqdm(total=total, desc=desc, leave=False, bar_format=bar_format)

        eval_progress = _make_progress(
            "eval",
            len(eval_loader) if hasattr(eval_loader, "__len__") else None,
        )

        try:
            try:
                eval_iter = iter(eval_loader)
            except Exception as exc:
                raise RuntimeError("Exception creating evaluation data iterator") from exc

            for callback in callbacks_list:
                method = getattr(callback, "on_validation_start", None)
                if method is not None:
                    method(module, optimizer)

            while True:
                with wallclock("eval/data.next", eval_step + 1):
                    try:
                        batch = next(eval_iter)
                    except StopIteration:
                        break

                eval_step += 1
                rank_zero(eval_progress.update, 1)

                for callback in callbacks_list:
                    method = getattr(callback, "on_validation_step_start", None)
                    if method is not None:
                        method(
                            module,
                            optimizer,
                            batch,
                            step = eval_step
                        )

                if key is not None:
                    key, eval_key = jax.random.split(key)
                else:
                    eval_key = None

                with wallclock("eval/step", eval_step):
                    eval_aux = eval_step(module, optimizer, batch, key=eval_key)

                aggregated_aux = _merge_metrics(aggregated_aux, eval_aux)
                eval_metric += eval_aux
                eval_logs = eval_metric.step_metrics()
                rank_zero(log_method, "eval", eval_logs, step=eval_step)

                for callback in callbacks_list:
                    method = getattr(callback, "on_validation_step_end", None)
                    if method is not None:
                        method(
                            module,
                            optimizer,
                            batch,
                            eval_aux,
                            eval_logs
                            step=eval_step
                        )

            eval_logs = eval_metric.per_N_metrics(skip_check = True)
            rank_zero(logger.log, "eval", eval_logs, eval_step, )

                for callback in callbacks_list:
                    method = getattr(callback, "on_validation_step_end", None)
                    if method is not None:
                        method(module, optimizer, None, aux=reduced, step_idx=eval_step)

            for callback in callbacks_list:
                method = getattr(callback, "on_validation_end", None)
                if method is not None:
                    method(
                        module,
                        optimizer,
                        aux=aggregated_aux,
                        logs=reduced,
                        global_step
                    )

            return reduced, key

        finally:
            rank_zero(eval_progress.close)



@tp.overload
def train_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    train_step_fn: _TrainStepCallable[_ModuleInput, _OptimizerInput],
    train_loader: tp.Iterable[tp.Any],
    num_train_steps: int | None = None,
    *,
    logger: Logger | None = None,
    enable_wallclock: bool = True,
    eval_loader: tp.Iterable[tp.Any] | None = None,
    eval_step_fn: _EvalStepCallable[_ModuleInput, _OptimizerInput] | None = None,
    evals: None = None,
    eval_interval: int | None = None,
    callbacks: tp.Sequence[Callback] | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[_ModuleInput, _OptimizerInput, dict[str, tp.Any], dict[str, tp.Any]]:
    ...


@tp.overload
def train_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    train_step_fn: _TrainStepCallable[_ModuleInput, _OptimizerInput],
    train_loader: tp.Iterable[tp.Any],
    num_train_steps: int | None = None,
    *,
    logger: Logger | None = None,
    enable_wallclock: bool = True,
    eval_loader: None = None,
    eval_step_fn: None = None,
    evals: tp.Sequence[Eval],
    eval_interval: int | None = None,
    callbacks: tp.Sequence[Callback] | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[_ModuleInput, _OptimizerInput, dict[str, tp.Any], dict[str, tp.Any]]:
    ...


def train_loop(
    module: _ModuleInput,
    optimizer: _OptimizerInput,
    train_step_fn: _TrainStepCallable[_ModuleInput, _OptimizerInput],
    train_loader: tp.Iterable[tp.Any],
    num_train_steps: int | None = None,
    *,
    logger: Logger | None = None,
    log_every_n_steps: int | None = None,
    train_metric: SufficientMetric | None = None,
    enable_wallclock: bool = True,
    eval_loader: tp.Iterable[tp.Any] | None = None,
    eval_step_fn: _EvalStepCallable[_ModuleInput, _OptimizerInput] | None = None,
    evals: tp.Sequence[Eval] | None = None,
    eval_metric: SufficientMetric | None = None,
    eval_interval: int | None = None,
    callbacks: tp.Sequence[Callback] | None = None,
    key: PRNGKeyArray | None = None,
) -> tuple[_ModuleInput, _OptimizerInput, dict[str, tp.Any], dict[str, tp.Any]]:
    if eval_loader is not None and eval_step_fn is None:
        raise ValueError("eval_step must be provided when eval_loader is set")
    if evals is not None and (eval_loader is not None or eval_step_fn is not None):
        raise ValueError("Provide either evals or eval_loader/eval_step, not both")

    callbacks_list = list(callbacks or [])
    evals_list = list(evals or [])
    train_metric = SufficientMetric(name = "train", log_every_n_steps=log_every_n_steps) if train_metric is None else train_metric

    if logger is None:
        logger = rank_zero(
            TensorBoardLogger,
            log_dir="log/tensorboard",
            experiment_name="train",
            enabled=True,
        )

    train_step = 0

    def wallclock(name: str, step_idx: int | None = None):
        if not enable_wallclock:
            return nullcontext()
        ctx = rank_zero(logger.wc, name) if step_idx is None else rank_zero(logger.wc, name, step=step_idx)
        return ctx if ctx is not None else nullcontext()

    @rank_zero
    def _make_progress(desc: str, total: int | None):
        bar_format = None if total is not None else "{desc}: {n}"
        return tqdm(total=total, desc=desc, leave=False, bar_format=bar_format)

    train_progress = _make_progress("Train", num_train_steps)

    for callback in callbacks_list:
        method = getattr(callback, "on_training_start", None)
        if method is not None:
            method(module, optimizer, train_step)


    try:
        train_iter = iter(train_loader)
    except Exception as exc:
        raise RuntimeError("Exception creating data loader iterator") from exc

    try:
        while True:
            if num_train_steps is not None and train_step >= num_train_steps:
                break

            train_step += 1

            with wallclock("train/data.next", train_step):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    LOGGER.info("Training data exhausted, ending training loop.")
                    break

            rank_zero(train_progress.update, 1)

            for callback in callbacks_list:
                method = getattr(callback, "on_training_step_start", None)
                if method is not None:
                    method(module, optimizer, batch, step=train_step)

            key, step_key = jax.random.split(key, 2) if key is not None else (key, None)

            with wallclock("train/step", train_step):
                module, optimizer, aux = train_step_fn(module, optimizer, batch, key=step_key)

            aux = aux or {}

            train_metric += aux
            step_metrics = train_metric.step_metrics()
            per_n_metrics = train_metric.per_N_metrics(step=train_step)
            logs = {**step_metrics, **per_n_metrics}


            rank_zero(logger.log, "train_step", step_metrics, step = train_step)
            rank_zero(logger.log, "per_N_train_step", per_n_metrics, step = train_step)

            should_eval = (
                eval_interval is not None
                and eval_interval > 0
                and train_step % eval_interval == 0
            )

            for callback in callbacks_list:
                method = getattr(callback, "on_training_step_end", None)
                if method is not None:
                    method(
                        module,
                        optimizer,
                        batch,
                        aux,
                        logs,
                        train_step,
                    )

            if should_eval:
                key, eval_key = jax.random.split(key, 2) if key is not None else (key, None)
                eval_metric = eval_loop(
                    module,
                    optimizer,
                    eval_loader=eval_loader,
                    eval_step=eval_step_fn,
                    evals=evals_list,
                    eval_metric = 
                    logger=logger,
                    callbacks=callbacks_list,
                    enable_wallclock=enable_wallclock,
                    key=eval_key,
                    global_step_idx=train_step,
                )
            if eval_logs:
                last_eval_logs = eval_logs

    except Exception:
        LOGGER.error("Exception during training loop", exc_info=True)
        raise
    finally:
        LOGGER.info("Training loop ended")
        rank_zero(train_progress.close)

    per_N_metrics = train_metric.per_N_metrics(step = train_step, skip_check = True)

    for callback in callbacks_list:
        method = getattr(callback, "on_training_end", None)
        if method is not None:
            method(
                module,
                optimizer,
                aux,
                per_N_metrics,
                train_step
            )

    rank_zero(logger.finalize)

    return module, optimizer, train_metric, eval_metric 
