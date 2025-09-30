from __future__ import annotations

import logging
import typing as tp
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Protocol

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray, PyTree
from optax import GradientTransformation, GradientTransformationExtraArgs

from ._metrics import SufficientMetric
from ._training import Optimizer
from ._utils import first_from, wallclock
from .callbacks import Callback
from .loggers import Logger


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
_ParallelismPlans = (
    dict[str, tp.Callable[[_M], _M]] | tp.Sequence[dict[str, tp.Callable[[_M], _M]]]
)


_ModuleInput = tp.TypeVar("_ModuleInput", _M, tp.Sequence[_M])
_OptimizerInput = tp.TypeVar("_OptimizerInput", _O, tp.Sequence[_O])


class _EvalStepCallable(Protocol[_ModuleInput, _OptimizerInput]):
    def __call__(
        self,
        module: _ModuleInput,
        optimizer: _OptimizerInput,
        batch: tp.Any,
        *,
        key: PRNGKeyArray | None = None,
    ) -> _Aux: ...


@dataclass
class Eval:
    name: str
    dataset: tp.Iterable[tp.Any]
    eval_step: _EvalStepCallable[_ModuleInput, _OptimizerInput] | None = None
    loss_function: _LossFn | None = None
    eval_metric: SufficientMetric | None = None
    logger: Logger | None = None
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
        global_step_idx: int | None = None,
        key: PRNGKeyArray | None = None,
    ) -> SufficientMetric:

        logger = first_from(self.logger, logger, error_msg="logger required")

        with wallclock() if enable_wallclock else nullcontext():
            eval_step_fn = self._resolve_eval_step()
            dataset, handle = self._resolve_dataset()
            step_idx = 0
            eval_metric = self.eval_metric if self.eval_metric else SufficientMetric(name=self.name, log_every_n_steps=None)

            try:
                iterator = iter(dataset)
            except TypeError as e:
                raise RuntimeError("eval dataset is not iterable") from e

            try:
                for batch in iterator:
                    step_idx += 1

                    key, eval_key = (
                        jax.random.split(key, 2) if key is not None else (key, None)
                    )

                    aux = eval_step_fn(
                        module,
                        optimizer,
                        batch,
                        key=eval_key,
                    )

                    eval_metric.add(aux)


                logs = eval_metric.per_N_metrics(step_idx, skip_check=True)
                logger.log(self.name, logs, global_step_idx)
                return eval_metric
            except Exception as e:
                raise e


def make_eval_step(
    *,
    loss_function: _LossFn | None = None,
    eval_step: _EvalStepCallable[tp.Sequence[_M], tp.Sequence[Optimizer]]
    | _EvalStepCallable[_M, Optimizer]
    | None = None,
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
    train_step: int | None = None,
    enable_wallclock: bool = False,
    *,
    key: PRNGKeyArray | None = None,
) -> dict[str, SufficientMetric]:
    callbacks = list(callbacks or [])
    eval_metrics: dict[str, SufficientMetric] = {}

    for callback in callbacks:
        method = getattr(callback, "on_validation_start", None)
        if method is not None:
            method(module, optimizer, logger, train_step)

    for eval_obj in evals:
        eval_metric = eval_obj.run(
            module,
            optimizer,
            key=key,
            logger=logger,
            enable_wallclock=enable_wallclock,
            global_step_idx=train_step,
            callbacks=callbacks,
        )
        eval_metrics[eval_obj.name] = eval_metric

    for callback in callbacks:
        method = getattr(callback, "on_validation_end", None)
        if method is not None:
            method(
                module,
                optimizer,
                eval_metrics,
                logger,
                train_step,
            )

    return eval_metrics
