import logging
import typing as tp
from time import time

import equinox as eqx
import jax
import jax.tree_util as jtu
import optax
import orbax.checkpoint as ocp
from jaxtyping import PRNGKeyArray, PyTree
from tqdm import tqdm

import src.distributed as dst
from src import Darray, Optimizer


LOGGER = logging.getLogger(__name__)


AbstractModule = tp.Any
AxisSpec = bool | tp.Callable[[tp.Any], bool]
Wrt = PyTree[AxisSpec]

Loss = float
Aux = dict[str, int | float]

M = tp.TypeVar("M", eqx.Module, AbstractModule)
GradTx = optax.GradientTransformation | optax.GradientTransformationExtraArgs

A = tp.TypeVar("A")

def _as_list(x :A | tp.Sequence[A]) -> list[A]:
    return x if isinstance(x, tp.Sequence) else [x]

def _is_shape_dtype_struct(x)  -> bool:
    return isinstance(x, jax.ShapeDtypeStruct)

def is_darray(x) -> bool:
    return isinstance(x, Darray)

def _is_abstract_module(tree) -> bool:
    found = False

    def _check(leaf):
        nonlocal found
        if _is_shape_dtype_struct(leaf):
            found = True
        return leaf

    jtu.tree_map(_check, tree, is_leaf = is_darray)
    return found


class TrainerModule:

    def __init__(
        self,
        module: M | tp.Sequence[M],
        grad_tx: GradTx | tp.Sequence[GradTx],
        *,
        parallel_config: tp.Any | None = None,
        wrt_filters: Wrt | tp.Sequence[Wrt] | None = None,
        logger: tp.Any | None = None,
        checkpoint_manager: ocp.CheckpointManager,
        key: PRNGKeyArray | None = None,
        jit = True,
    ):
        self.modules= _as_list(module) 
        self.grad_txs = _as_list(grad_tx)
        assert len(self.grad_txs) in (1, len(self.modules)), "grad_txs, mus be single or per  per-module list"
        if len(self.grad_txs) == 1 and len(self.modules) > 1:
            self.grad_txs = len(self.modules) * self.grad_txs

        if wrt_filters is None:
            self._wrt_filters = [eqx.is_inexact_array] * len(self.modules)
        else:
            self._wrt_filters = _as_list(wrt_filters) 

            if len(self._wrt_filters) ==1 and len(self.modules) > 1:
                self._wrt_filters = self._wrt_filters * len(self.modules)

            assert len(self._wrt_filters) == len(self.modules), "wrt_filters must match modules"

        self.optmizers = None
        self.logger = logger

        self._jit = jit
        self._compiled_train_step = None
        self._compiled_eval_step = None
        self._step = 0
        self._first_step = False
        self._single_module = len(self.modules)

        def loss_function(
            self, 
            module: M,
            optmizer: Optimizer,
            batch: tp.Any,
            *,
            key: PRNGKeyArray,
        )-> tuple[Loss, Aux]:
            raise NotImplementedError

        def train_step(
            self, 
            module: M | tp.Sequence[M],
            optmizer: Optimizer | tp.Sequence[Optimizer],
            batch: tp.Any,
            *,
            key: PRNGKeyArray,
        )-> tuple[M, Optimizer, Aux] | tuple[tp.Sequence[M], tp.Seqence[Optimizer], Aux]:
            raise NotImplementedError

        def eval_step(
            self, 
            module: M | tp.Sequence[M],
            optmizer: Optimizer | tp.Sequence[Optimizer],
            batch: tp.Any,
            *,
            key: PRNGKeyArray,
        ) -> Aux: 
            raise NotImplementedError

    def parallelism_plan(self, modules: M | tp.Sequence[M],) -> M | tp.Sequence[M]:
        raise NotImplementedError

    def _compile_modules_opts_if_needed(self):
        has_parallelism_plan = self._is_overidden(self.__class__, "parallelism_plan")

        def _shard_module_opts(modules, grad_txs, wrts):
            modules = self.parallelism_plan(self.modules) 
            partition_specs = dst.get_partition_spec(modules)
            modules = eqx.filter_shard(modules, partition_specs)
            optimizers = [Optimizer(grad_tx, module, wrt) for grad_tx, module, wrt in zip(grad_txs, modules, wrts)]
            optimizers = eqx.filter_shard(modules, partition_specs) 
            return modules, optimizers

        def _re_init_and_shard_modules_opts(modules, grad_txs, wrts):
            pass

        if has_parallelism_plan:
            if not _is_abstract_module(self.modules):
                self.modules, self.optimizers = eqx.filter_jit(_shard_module_opts)(self.modules, self.grad_txs, self._wrt_filters)
            else:
                self.modules, self.optimizers = eqx.filter_jit(_re_init_and_shard_modules_opts)(self.modules, self.grad_txs, self._wrt_filters)
        else:
            pass


    def _compile_steps_if_needed(self):
        has_train_step = self._is_overidden(self.__class__, "train_step")
        has_loss_function = self._is_overidden(self.__class__, "loss_function")
        has_eval = self.is_overidden(self.__class_, "eval_step")

        if not has_train_step and not has_loss_function:
            raise ValueError("when subclassing TrainerModule, you must override either train_step or loss_function. "
                             "For single module training, it's recommended to override loss_function. "
                             "For multi-module training, it's recommended to override train_step.")
        
        if len(self.modules) == 1 and has_train_step:
            self.compiled_train_step = eqx.filter_jit(self.train_step) if self.jit else self.train_step
        elif len(self.modules) == 1 and has_loss_function:
            def _step_single_loss(
                module, 
                optimizer: Optimizer, 
                batch,
                *, 
                key,
            ):
                grad_fn = eqx.filter_value_and_grad(self.loss_function, has_aux = True)
                (_, aux), grad = grad_fn(module, optimizer, batch, key = key) 
                new_module, new_optimizer = optimizer(grad, module)
                return [new_module], [new_optimizer], aux
            self._compiled_train_step = eqx.filter_jit(_step_single_loss) if self.jit else _step_single_loss
        elif len(self.modules) > 1 and has_train_step:
            self.compiled_train_step = eqx.filter_jit(self.train_step) if self.jit else self.train_step
        else:
            if has_loss_function:
                raise ValueError("For multi-module training, you must override train_step, not loss_function.")

        if has_eval:
            self._compiled_eval_step = eqx.filter_jit(self.eval_step) if self._jit else self.eval_step
        elif has_loss_function and len(self.modules) == 1:
            def _eval_single_step(*args, **kwargs):
                _, aux = self.loss_function(*args, **kwargs)
                return aux
            self._compiled_eval_step = eqx.filter_jit(_eval_single_step) if self._jit else _eval_single_step 
        else:
            self.comiled_eval_step = None

    def on_training_start(self):
        if self._callbacks is not None:
            try:
                for callback in self.callbacks:
                    callback.on_traning_start()
            except Exception as e:
                raise ValueError("Error in on_training_start callback") from e

    def tracker(self, iterable: tp.Iterable, **kwargs) -> tp.Iterable:
        if jax.process_index() == 0:
            return tqdm(iterable, **kwargs)
        else:
            return iterable


    def _is_overidden(
        self,
        cls,
        method,
    ):
        base_fn = getattr(TrainerModule, method, None)
        sub_fn = getattr(cls, method, None)
        return  (sub_fn is not None) and (sub_fn is not base_fn)

    def train(
        self, 
        train_dataloader: tp.Iterable,
        metrics: List[Metrics],
        eval_dataloader: tp.Iterable | None = None,
        desired_train_step: int | None = None,
        eval_step_interval: int | None = None, 
        desired_eval_step: int | None = None,
        *,
        key: PRNGKeyArray | None = None,
    ):

        if desired_train_step is None:
            desired_train_steps = len(train_dataloader)

        self._compile_modules_opts_if_needed()
        self._compile_steps_if_needed()

        stop_training_on_error = False

        while self.step < desired_train_steps and not stop_training_on_error:
            key, step_key = jax.random.split(key)
            batch = next(train_dataloader)

            if self.first_step:
                LOGGER.info("Compiling train_step...")
                start_time = time.time()
                if self._single_module: 
                    self.modules, self.optimizers, step_aux = self._compiled_train_step(self.modules[0], self.optimizers[0], batch, key = step_key) 
                else:
                    self.modules, self.optimizers, step_aux = self._compiled_train_step(self.modules, self.optimizers, batch, key = step_key) 

                LOGGER.info(
                    f"Successfully completed train_step compilation in {time.time() - start_time:.2f} seconds."
                )
                self.first_step = False
            else:
                with jax.profiler.StepTraceAnnotation(name = f"train step", step_num = self.step):
                    if self._single_module: 
                        self.modules, self.optimizers, step_aux = self._compiled_train_step(self.modules[0], self.optimizers[0], batch, key = step_key) 
                    else:
                        self.modules, self.optimizers, step_aux = self._compiled_train_step(self.modules, self.optimizers, batch, key = step_key) 


            self._logger.log_step(self.step, step_metrics, namespace = "train")
            self.maybe_checkpoint(self.modules, self.optimizers, step = self.step)

            if self.step % eval_step_interval == 0:
                eval_step = 0
                while eval_step < desired_eval_step:
                    eval_batch = next(eval_dataloader)
                    if self._single_module: 
                        eval_step_aux = self._compiled_eval_step(self.modules[0], self.optimizers[0], batch, key = step_key) 
                    else:
                        eval_step_aux = self._compiled_train_step(self.modules, self.optimizers, batch, key = step_key) 

                    self._logger.log_step(eval_step, eval_step_aux, namespace = "eval")
                    self._logger.maybe_write(eval_step, namespace = "eval")
                    eval_step +=1

                self._logger.maybe_write(eval_step, namespace = "eval")
                self.maybe_checkpoint(self.modules, self.optimizers, step = self.step)
            self.step += 1

            return self._logger.metrics



