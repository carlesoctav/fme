from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jaxtyping import Array, PyTree


try:
    import orbax.checkpoint as ocp
except Exception:  
    ocp = None 

from ._darray import Darray
from ._training import Optimizer
from .distributed._params import get_partition_spec
from .logger import Logger, LoggerConfig


ModuleT = eqx.Module
GradTx = optax.GradientTransformation | optax.GradientTransformationExtraArgs
PRNGKey = jax.Array


def _is_shape_dtype_struct(x: Any) -> bool:
    return isinstance(x, jax.ShapeDtypeStruct)


def _any_is_shape_dtype_struct(tree: Any) -> bool:
    found = False

    def _check(x):
        nonlocal found
        if _is_shape_dtype_struct(x):
            found = True
        return x

    jtu.tree_map(_check, tree, is_leaf=_is_shape_dtype_struct)
    return found


def _apply_with_sharding_constraint(tree: Any, spec_tree: Any) -> Any:
    def _apply(x, pspec):
        try:
            if isinstance(x, jax.Array) and pspec is not None:
                return jax.lax.with_sharding_constraint(x, pspec)
        except Exception:
            return x
        return x

    return jtu.tree_map(_apply, tree, spec_tree, is_leaf=lambda x: False)


def _as_list(x: ModuleT | Sequence[ModuleT]) -> list[ModuleT]:
    return list(x) if isinstance(x, Sequence) else [x]


def _as_list_tx(x: GradTx | Sequence[GradTx]) -> list[GradTx]:
    return list(x) if isinstance(x, Sequence) else [x]


@dataclass
class ParallelConfig:
    config: Any | None = None


class TrainerModule:

    def __init__(
        self,
        modules: ModuleT | Sequence[ModuleT],
        grad_fns: GradTx | Sequence[GradTx],
        *,
        parallel_config: Any | None = None,
        module_init_fns: Callable[[PRNGKey, ModuleT], ModuleT] | Sequence[Callable[[PRNGKey, ModuleT], ModuleT]] | None = None,
        wrt_filters: Any | Sequence[Any] | None = None,
        logger: Logger | None = None,
        logger_config: LoggerConfig | None = None,
        # Checkpointing
        checkpoint_dir: str | None = None,
        # JIT behavior
        jit: bool = True,
    ):
        self.parallel_config = ParallelConfig(parallel_config)
        self.orig_modules: list[ModuleT] = _as_list(modules)
        self.modules: list[ModuleT] = list(self.orig_modules)
        self.orig_grad_fns: list[GradTx] = _as_list_tx(grad_fns)
        assert len(self.orig_grad_fns) in (1, len(self.modules)), "grad_fns must be single or per-module list"
        if len(self.orig_grad_fns) == 1 and len(self.modules) > 1:
            self.orig_grad_fns = self.orig_grad_fns * len(self.modules)

        # wrt filters for optimizers
        if wrt_filters is None:
            self.wrt_filters = [eqx.is_inexact_array] * len(self.modules)
        else:
            self.wrt_filters = list(wrt_filters) if isinstance(wrt_filters, Sequence) else [wrt_filters]
            if len(self.wrt_filters) == 1 and len(self.modules) > 1:
                self.wrt_filters = self.wrt_filters * len(self.modules)
            assert len(self.wrt_filters) == len(self.modules), "wrt_filters must match modules"

        # JIT behavior
        self._jit = jit

        # Abstract-module materializers
        if module_init_fns is None:
            self.module_init_fns: list[Callable[[PRNGKey, ModuleT], ModuleT] | None] = [None] * len(self.modules)
        else:
            self.module_init_fns = list(module_init_fns) if isinstance(module_init_fns, Sequence) else [module_init_fns]
            if len(self.module_init_fns) == 1 and len(self.modules) > 1:
                self.module_init_fns = self.module_init_fns * len(self.modules)
            assert len(self.module_init_fns) == len(self.modules), "module_init_fns must match modules or be single"

        # Initialize (materialize) modules if abstract.
        # Defer rng; user should call materialize_modules explicitly with rng if needed.
        self._materialized = False

        # Build optimizers lazily, after materialization/resharding.
        self.optimizers: list[Optimizer] | None = None

        # Compile steps if provided
        self._compiled_train_step = None
        self._compiled_eval_step = None

        # Logging
        if logger is not None:
            self._logger: Logger | None = logger
        elif logger_config is not None:
            self._logger = Logger(logger_config)
        else:
            self._logger = None

        # Checkpointing
        self._ckpt_dir = checkpoint_dir

        # Single vs multi checks are deferred until step creation; this is an abstract base meant for subclassing.

    # -------- Methods to override in subclasses --------
    def loss_function(
        self,
        module: ModuleT,
        optimizer: Optimizer,
        batch: Any,
        rngs: dict[str, PRNGKey] | PRNGKey,
        *,
        train: bool = True,
    ) -> tuple[Array, dict]:
        raise NotImplementedError

    def train_step(
        self,
        *args: Any,
        batch: Any,
        rngs: dict[str, PRNGKey] | PRNGKey,
    ) -> tuple[Sequence[ModuleT], Sequence[Optimizer], dict | None]:
        raise NotImplementedError

    def eval_step(
        self,
        *args: Any,
        batch: Any,
        rngs: dict[str, PRNGKey] | PRNGKey,
    ) -> dict | None:
        # Optional; default uses loss_function for single-module, else no-op.
        raise NotImplementedError

    # -------- Public API --------
    def materialize_modules(self, rng: PRNGKey | dict[str, PRNGKey] | None = None):
        """
        If any module contains jax.ShapeDtypeStruct leaves, use `module_init_fns[i]` or a module `init(rng, abstract)`
        method to create a concrete module. Otherwise, no-op.
        """
        new_modules: list[ModuleT] = []
        rng_seq: Iterable[PRNGKey]
        if isinstance(rng, dict):  # allow dict of rngs, take a stable order
            rng_seq = list(rng.values())
        elif rng is None:
            rng_seq = [jax.random.PRNGKey(0) for _ in self.modules]
        else:
            rng_seq = [rng]

        # Make per-module RNGs
        rngs: list[PRNGKey] = list(rng_seq)
        if len(rngs) < len(self.modules):
            base = rngs[0] if rngs else jax.random.PRNGKey(0)
            rngs = list(jax.random.split(base, len(self.modules)))

        for i, m in enumerate(self.modules):
            if _any_is_shape_dtype_struct(m):
                # First, try user-provided module_init_fn
                init_fn = self.module_init_fns[i]
                if init_fn is not None:
                    new_modules.append(init_fn(rngs[i], m))
                    continue
                # Else, try to materialize Darray leaves using module’s own initializers when available
                new_modules.append(self._materialize_darrays_in_module(m, rngs[i]))
            else:
                new_modules.append(m)
        self.modules = new_modules
        self._materialized = True

    def build_optimizers(self):
        if not self._materialized:
            # Materialize with default RNG if user forgot.
            self.materialize_modules(jax.random.PRNGKey(0))
        opts: list[Optimizer] = []
        for m, tx, wrt in zip(self.modules, self.orig_grad_fns, self.wrt_filters):
            opts.append(Optimizer(tx, m, wrt=wrt))
        self.optimizers = opts

    def parallelism_plan(self, *modules: ModuleT) -> Sequence[ModuleT]:
        """Hook for subclasses to transform modules for parallelism (e.g., tensor/fused sharding)."""
        return modules

    def get_partition_specs(self, modules: Sequence[ModuleT]) -> Sequence[PyTree]:
        return [get_partition_spec(m) for m in modules]

    def reshard_modules_and_opts(self):
        """
        Apply `parallelism_plan`, compute partition specs, and apply sharding constraints to modules and optimizer
        states. Rebuild optimizers against reshaped modules (new opt_state).
        """
        if not self._materialized:
            self.materialize_modules(jax.random.PRNGKey(0))

        # Plan
        planned_modules = self.parallelism_plan(*self.modules)
        if isinstance(planned_modules, tuple):
            planned_modules = list(planned_modules)
        else:
            planned_modules = list(planned_modules)

        # Partition specs
        specs = self.get_partition_specs(planned_modules)

        # Apply constraints to module parameters
        new_modules: list[ModuleT] = []
        for m, s in zip(planned_modules, specs):
            new_modules.append(_apply_with_sharding_constraint(m, s))
        self.modules = new_modules

        # Rebuild optimizers based on new modules and original grad transformations
        self.build_optimizers()

        # Optionally apply constraints to opt states if helpful (best-effort)
        if self.optimizers is not None:
            new_opts: list[Optimizer] = []
            for opt, spec in zip(self.optimizers, specs):
                try:
                    sharded_state = _apply_with_sharding_constraint(opt.opt_state, spec)
                    opt = eqx.tree_at(lambda o: o.opt_state, opt, sharded_state)
                except Exception:
                    pass
                new_opts.append(opt)
            self.optimizers = new_opts

    def reshard_module(
        self,
        *args: ModuleT | Optimizer,
    ) -> tuple[list[ModuleT], list[Optimizer]]:
        """
        Functional-style reshard that accepts a flat list `(*module_args, *opt_args)` and returns
        `(module_args, opt_args)` reshaped and with optimizer states rebuilt.

        This mirrors the user-facing API described in the request.
        """
        # Split inputs into modules then optimizers by counting
        mods: list[ModuleT] = []
        opts: list[Optimizer] = []
        for a in args:
            if isinstance(a, eqx.Module):
                mods.append(a)
            else:
                opts.append(a)  # type: ignore[arg-type]
        planned = self.parallelism_plan(*mods)
        planned = list(planned) if isinstance(planned, (list, tuple)) else [planned]
        specs = [get_partition_spec(m) for m in planned]
        new_mods = [_apply_with_sharding_constraint(m, s) for m, s in zip(planned, specs)]
        # Build new optimizers based on new modules and original grad_fns
        new_opts: list[Optimizer] = []
        gfs = self.orig_grad_fns if len(self.orig_grad_fns) == len(new_mods) else self.orig_grad_fns * len(new_mods)
        wrts = self.wrt_filters if len(self.wrt_filters) == len(new_mods) else self.wrt_filters * len(new_mods)
        for m, tx, wrt in zip(new_mods, gfs, wrts):
            new_opts.append(Optimizer(tx, m, wrt=wrt))
        # Best-effort sharding of opt state to match param specs
        tmp_opts: list[Optimizer] = []
        for opt, spec in zip(new_opts, specs):
            try:
                sharded_state = _apply_with_sharding_constraint(opt.opt_state, spec)
                opt = eqx.tree_at(lambda o: o.opt_state, opt, sharded_state)
            except Exception:
                pass
            tmp_opts.append(opt)
        return new_mods, tmp_opts

    # -------- Step creation --------
    def _compile_steps_if_needed(self):
        if self._compiled_train_step is not None:
            return

        # Determine which methods are overridden by subclass
        has_train = self._is_overridden(self.__class__, "train_step")
        has_loss = self._is_overridden(self.__class__, "loss_function")
        has_eval = self._is_overridden(self.__class__, "eval_step")

        if len(self.modules) == 1:
            if has_train:
                def _step_single(mod: ModuleT, opt: Optimizer, batch: Any, rngs: dict[str, PRNGKey] | PRNGKey):
                    out = self.train_step(mod, opt, batch=batch, rngs=rngs)
                    if len(out) == 2:
                        (nm,), (no,) = out
                        metrics = None
                    else:
                        (nm,), (no,), metrics = out
                    return nm, no, metrics

                self._compiled_train_step = eqx.filter_jit(_step_single) if self._jit else _step_single
            elif has_loss:
                def _step_single_loss(mod: ModuleT, opt: Optimizer, batch: Any, rngs: dict[str, PRNGKey] | PRNGKey):
                    def _loss_only(m: ModuleT):
                        l, aux = self.loss_function(m, opt, batch, rngs, train=True)
                        return l, aux

                    (loss, aux), grads = eqx.filter_value_and_grad(_loss_only, has_aux=True)(mod)
                    new_mod, new_opt = opt(grads, mod)
                    metrics = {"loss": loss}
                    if isinstance(aux, dict):
                        metrics.update(aux)
                    return new_mod, new_opt, metrics

                self._compiled_train_step = eqx.filter_jit(_step_single_loss) if self._jit else _step_single_loss
            else:
                raise ValueError("Subclass must implement either train_step or loss_function for single-module mode.")
        else:
            if not has_train:
                raise ValueError("For multiple modules, subclass must implement train_step.")

            def _step_multi(*args):
                # Expect (*modules, *opts, batch, rngs)
                out = self.train_step(*args)
                if len(out) == 2:
                    new_modules, new_opts = out
                    metrics = None
                else:
                    new_modules, new_opts, metrics = out
                return new_modules, new_opts, metrics

            self._compiled_train_step = eqx.filter_jit(_step_multi) if self._jit else _step_multi

        # Eval step
        if has_eval:
            def _eval_bound(*args):
                return self.eval_step(*args)

            self._compiled_eval_step = eqx.filter_jit(_eval_bound) if self._jit else _eval_bound
        elif has_loss and len(self.modules) == 1:
            def _eval_single(mod: ModuleT, opt: Optimizer, batch: Any, rngs: dict[str, PRNGKey] | PRNGKey):
                loss, aux = self.loss_function(mod, opt, batch, rngs, train=False)
                metrics = {"loss": loss}
                if isinstance(aux, dict):
                    metrics.update(aux)
                return metrics

            self._compiled_eval_step = eqx.filter_jit(_eval_single) if self._jit else _eval_single
        else:
            self._compiled_eval_step = None

    # ----- Training lifecycle hooks (subclasses may override) -----
    def on_training_start(self):
        if self._logger is not None:
            try:
                self._logger.on_training_start()
            except Exception:
                pass

    def on_training_end(self):
        if self._logger is not None:
            try:
                self._logger.on_training_end()
            except Exception:
                pass

    def on_training_epoch_start(self, epoch_idx: int):
        if self._logger is not None:
            try:
                self._logger.on_training_epoch_start(epoch_idx)
            except Exception:
                pass

    def on_training_epoch_end(self, epoch_idx: int, metrics: dict | None = None):
        if self._logger is not None:
            try:
                self._logger.on_training_epoch_end(metrics, epoch_idx)
            except Exception:
                pass

    # -------- Training Loop --------
    def _log_metrics(self, metrics: dict | None, step: int, mode: str = "train"):
        if metrics is None:
            return
        if self._logger is not None:
            try:
                self._logger.log_host_metrics(metrics, step=step, mode=mode)
            except Exception:
                pass

    def train(
        self,
        train_loader: Iterable,
        *,
        num_steps: int | None = None,
        num_epochs: int | None = None,
        steps_per_epoch: int | None = None,
        rng: PRNGKey | dict[str, PRNGKey] | None = None,
        log_every: int = 10,
        eval_loader: Iterable | None = None,
        eval_every: int = -1,
        save_every: int = -1,
        start_step: int = 0,
    ):
        """Simple training loop.

        - If using single-module+loss_function, we call the derived step.
        - If using multi-module, the user-provided train_step must return (modules, optimizers, metrics?)
        """
        if self.optimizers is None:
            self.build_optimizers()

        self._compile_steps_if_needed()

        step = start_step
        # RNG handling
        if rng is None:
            rng = jax.random.PRNGKey(0)
        base_rng = rng

        # Epoch semantics
        if num_epochs is not None:
            if steps_per_epoch is None:
                steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else None
                if steps_per_epoch is None:
                    raise ValueError("steps_per_epoch required when num_epochs is set and loader has no __len__().")

        # Fire training start
        self.on_training_start()
        if self._logger is not None:
            try:
                self._logger.on_training_start()
            except Exception:
                pass

        # Helper for epoch metric aggregation
        def _agg_add(acc_sums: dict, acc_cnts: dict, m: dict | None):
            if not m:
                return
            for k, v in m.items():
                try:
                    vv = jax.device_get(v)
                except Exception:
                    vv = v
                if hasattr(vv, "item"):
                    try:
                        vv = vv.item()
                    except Exception:
                        pass
                if isinstance(vv, (int, float)):
                    acc_sums[k] = acc_sums.get(k, 0.0) + float(vv)
                    acc_cnts[k] = acc_cnts.get(k, 0) + 1

        # Epoch-driven loop
        if num_epochs is not None and steps_per_epoch is not None:
            for epoch_idx in range(num_epochs):
                self.on_training_epoch_start(epoch_idx)
                if self._logger is not None:
                    try:
                        self._logger.start_epoch(epoch_idx, step, mode="train")
                    except Exception:
                        pass
                it = iter(train_loader)
                epoch_sums: dict[str, float] = {}
                epoch_cnts: dict[str, int] = {}
                for _ in range(steps_per_epoch):
                    try:
                        batch = next(it)
                    except StopIteration:
                        it = iter(train_loader)
                        batch = next(it)

                    # Split RNGs
                    if isinstance(base_rng, dict):
                        rngs = {k: jax.random.fold_in(v, step) for k, v in base_rng.items()}
                    else:
                        base_rng, step_rng = jax.random.split(base_rng)
                        rngs = step_rng

                    if len(self.modules) == 1 and isinstance(self._compiled_train_step, Callable):
                        new_mod, new_opt, metrics = self._compiled_train_step(
                            self.modules[0], self.optimizers[0], batch, rngs
                        )
                        self.modules[0], self.optimizers[0] = new_mod, new_opt
                    else:
                        assert isinstance(self._compiled_train_step, Callable)
                        # Pack as tuple for user fn: (*modules, *opts, batch, rngs)
                        args = tuple(self.modules) + tuple(self.optimizers or []) + (batch, rngs)
                        out = self._compiled_train_step(*args)
                        if isinstance(out, tuple) and len(out) == 3:
                            new_modules, new_opts, metrics = out
                        elif isinstance(out, tuple) and len(out) == 2:
                            new_modules, new_opts = out
                            metrics = None
                        else:
                            raise TypeError("train_step must return (modules, optimizers[, metrics])")
                        self.modules = list(new_modules)
                        self.optimizers = list(new_opts)

                    _agg_add(epoch_sums, epoch_cnts, metrics)
                    if (step % log_every) == 0:
                        self._log_metrics(metrics, step)

                    if eval_loader is not None and self._compiled_eval_step is not None and eval_every > 0:
                        if (step % eval_every) == 0:
                            try:
                                eval_batch = next(iter(eval_loader))
                            except Exception:
                                eval_batch = None
                            if eval_batch is not None:
                                if len(self.modules) == 1:
                                    eval_metrics = self._compiled_eval_step(
                                        self.modules[0], self.optimizers[0], eval_batch, rng
                                    )
                                else:
                                    args = tuple(self.modules) + tuple(self.optimizers or []) + (eval_batch, rng)
                                    eval_metrics = self._compiled_eval_step(*args)
                                if eval_metrics is not None:
                                    self._log_metrics({f"eval/{k}": v for k, v in eval_metrics.items()}, step)

                    if self._ckpt_dir is not None and save_every > 0 and (step % save_every) == 0:
                        self.save_checkpoint(step)

                    step += 1

                # End of epoch
                epoch_metrics = {k: (epoch_sums[k] / max(1, epoch_cnts.get(k, 1))) for k in epoch_sums}
                self.on_training_epoch_end(epoch_idx, epoch_metrics if epoch_metrics else None)
                if self._logger is not None:
                    try:
                        self._logger.end_epoch(epoch_metrics if epoch_metrics else {}, step)
                    except Exception:
                        pass

            # End of training
            self.on_training_end()
            if self._logger is not None:
                try:
                    self._logger.finalize("success")
                except Exception:
                    pass
            return

        # Step-driven loop
        it = iter(train_loader)
        current_epoch_idx = 0
        steps_per_epoch_detected = steps_per_epoch or (len(train_loader) if hasattr(train_loader, "__len__") else None)
        if steps_per_epoch_detected:
            self.on_training_epoch_start(current_epoch_idx)
            if self._logger is not None:
                try:
                    self._logger.start_epoch(current_epoch_idx, step, mode="train")
                except Exception:
                    pass
            epoch_sums: dict[str, float] = {}
            epoch_cnts: dict[str, int] = {}
        else:
            epoch_sums = {}
            epoch_cnts = {}
        while num_steps is None or step < num_steps:
            try:
                batch = next(it)
            except StopIteration:
                # Restart epoch
                it = iter(train_loader)
                try:
                    batch = next(it)
                except StopIteration:
                    break

            # Split RNGs
            if isinstance(base_rng, dict):
                rngs = {k: jax.random.fold_in(v, step) for k, v in base_rng.items()}  # deterministic per-step
            else:
                base_rng, step_rng = jax.random.split(base_rng)
                rngs = step_rng

            if len(self.modules) == 1 and isinstance(self._compiled_train_step, Callable):
                new_mod, new_opt, metrics = self._compiled_train_step(
                    self.modules[0], self.optimizers[0], batch, rngs
                )
                self.modules[0], self.optimizers[0] = new_mod, new_opt
            else:
                assert isinstance(self._compiled_train_step, Callable)
                # Pack as tuple for user fn: (*modules, *opts, batch, rngs)
                args = tuple(self.modules) + tuple(self.optimizers or []) + (batch, rngs)
                out = self._compiled_train_step(*args)
                # Normalize output to (mods, opts, metrics)
                if isinstance(out, tuple) and len(out) == 3:
                    new_modules, new_opts, metrics = out
                elif isinstance(out, tuple) and len(out) == 2:
                    new_modules, new_opts = out
                    metrics = None
                else:
                    raise TypeError("train_step must return (modules, optimizers[, metrics])")
                self.modules = list(new_modules)
                self.optimizers = list(new_opts)

            if (step % log_every) == 0:
                self._log_metrics(metrics, step)
            _agg_add(epoch_sums, epoch_cnts, metrics)

            if eval_loader is not None and self._compiled_eval_step is not None and eval_every > 0:
                if (step % eval_every) == 0:
                    try:
                        eval_batch = next(iter(eval_loader))
                    except Exception:
                        eval_batch = None
                    if eval_batch is not None:
                        if len(self.modules) == 1:
                            eval_metrics = self._compiled_eval_step(
                                self.modules[0], self.optimizers[0], eval_batch, rng
                            )
                        else:
                            args = tuple(self.modules) + tuple(self.optimizers or []) + (eval_batch, rng)
                            eval_metrics = self._compiled_eval_step(*args)
                        if eval_metrics is not None:
                            # Log as validation mode for tools
                            self._log_metrics(eval_metrics, step, mode="val")

            # Checkpointing
            if self._ckpt_dir is not None and save_every > 0 and (step % save_every) == 0:
                self.save_checkpoint(step)

            step += 1

            # Epoch boundary in step-driven mode
            if steps_per_epoch_detected and (step % steps_per_epoch_detected == 0):
                epoch_metrics = {k: (epoch_sums[k] / max(1, epoch_cnts.get(k, 1))) for k in epoch_sums}
                self.on_training_epoch_end(current_epoch_idx, epoch_metrics if epoch_metrics else None)
                if self._logger is not None:
                    try:
                        self._logger.end_epoch(epoch_metrics if epoch_metrics else {}, step)
                    except Exception:
                        pass
                current_epoch_idx += 1
                self.on_training_epoch_start(current_epoch_idx)
                if self._logger is not None:
                    try:
                        self._logger.start_epoch(current_epoch_idx, step, mode="train")
                    except Exception:
                        pass
                epoch_sums.clear()
                epoch_cnts.clear()

        # If we started an epoch but ended mid-epoch, signal epoch end
        if steps_per_epoch_detected and (epoch_sums or epoch_cnts):
            epoch_metrics = {k: (epoch_sums[k] / max(1, epoch_cnts.get(k, 1))) for k in epoch_sums}
            self.on_training_epoch_end(current_epoch_idx, epoch_metrics if epoch_metrics else None)
            if self._logger is not None:
                try:
                    self._logger.end_epoch(epoch_metrics if epoch_metrics else {}, step)
                except Exception:
                    pass

        # Training end
        self.on_training_end()
        if self._logger is not None:
            try:
                self._logger.finalize("success")
            except Exception:
                pass

    # -------- Checkpointing --------
    def _module_params_only(self, m: ModuleT) -> PyTree:
        return eqx.filter(m, eqx.is_inexact_array)

    def _apply_module_params(self, m: ModuleT, params: PyTree) -> ModuleT:
        static = eqx.filter(m, lambda x: not eqx.is_inexact_array(x))
        return eqx.combine(params, static)

    def _ckpt_payload(self):
        modules_params = [self._module_params_only(m) for m in self.modules]
        optim_states = [opt.opt_state for opt in (self.optimizers or [])]
        return {"modules_params": modules_params, "optim_states": optim_states}

    def save_checkpoint(self, step: int):
        if ocp is None:
            raise RuntimeError("orbax not available; cannot save checkpoint.")
        if self._ckpt_dir is None:
            raise ValueError("checkpoint_dir not set.")
        ckptr = ocp.PyTreeCheckpointer()
        ckptr.save(self._ckpt_dir, self._ckpt_payload(), step=step)

    def load_checkpoint(self, step: int | None = None):
        if ocp is None:
            raise RuntimeError("orbax not available; cannot load checkpoint.")
        if self._ckpt_dir is None:
            raise ValueError("checkpoint_dir not set.")
        ckptr = ocp.PyTreeCheckpointer()
        payload = ckptr.restore(self._ckpt_dir, step=step)
        modules_params: list[PyTree] = payload.get("modules_params", [])
        optim_states: list[PyTree] = payload.get("optim_states", [])
        # Rebuild modules with same static structure
        if len(modules_params) != len(self.modules):
            raise ValueError("Checkpoint modules count mismatch.")
        self.modules = [self._apply_module_params(m, p) for m, p in zip(self.modules, modules_params)]
        # Rebuild optimizers if needed
        if self.optimizers is None:
            self.build_optimizers()
        if len(optim_states) == len(self.optimizers):
            self.optimizers = [eqx.tree_at(lambda o: o.opt_state, o, s) for o, s in zip(self.optimizers, optim_states)]

    # -------- Helpers --------
    def _is_overridden(self, cls, name: str) -> bool:
        # True if subclass overrides a method from base class
        base_fn = getattr(TrainerModule, name, None)
        sub_fn = getattr(cls, name, None)
        return (sub_fn is not None) and (sub_fn is not base_fn)

    def _materialize_darrays_in_module(self, module: ModuleT, rng: PRNGKey) -> ModuleT:
        """
        Best-effort materialization: for any Darray with a ShapeDtypeStruct or None value, try to initialize using
        commonly stored initializers on the module (e.g., `initializer` for weights) or simple defaults.
        """
        # We iterate fields and reconstruct a new instance if needed
        fields = getattr(module, "__dataclass_fields__", None)
        if fields is None:
            return module

        updates: dict[str, Any] = {}

        for fname in fields:
            try:
                val = getattr(module, fname)
            except Exception:
                continue
            if isinstance(val, Darray):
                sds = val.value
                if sds is None:
                    continue
                if _is_shape_dtype_struct(sds):
                    shape = sds.shape
                    dtype = sds.dtype
                    # Choose initializer
                    init = None
                    # Prefer attribute named "initializer"
                    init_attr = getattr(module, "initializer", None)
                    if callable(init_attr):
                        init = init_attr
                    # Bias heuristic
                    if init is None and ("bias" in fname.lower()):
                        def init(key, shape, dtype):
                            return jnp.zeros(shape, dtype)
                    # Default: normal(0.02)
                    if init is None:
                        def init(key, shape, dtype):
                            return jax.nn.initializers.normal(stddev=0.02)(key, shape, dtype)
                    subkey = jax.random.fold_in(rng, hash(fname) & 0xFFFFFFFF)
                    new_value = init(subkey, shape, dtype)
                    updates[fname] = Darray(value=new_value, pspec=val.pspec)

        if not updates:
            return module
        new_mod = module
        for k, v in updates.items():
            try:
                object.__setattr__(new_mod, k, v)
            except Exception:
                pass
        return new_mod
