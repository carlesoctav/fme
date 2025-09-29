from __future__ import annotations

import math
import typing as tp
from collections.abc import Mapping
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from ._callbacks import Callback


try:
    import orbax.checkpoint as ocp
    from orbax.checkpoint import checkpoint_managers as ocp_cm
except Exception as exc:
    ocp = None
    ocp_cm = None
    _ORBAX_IMPORT_ERROR = exc
else:
    _ORBAX_IMPORT_ERROR = None

if tp.TYPE_CHECKING:
    from .._training import Optimizer as _Optimizer
    from orbax.checkpoint import CheckpointHandler as _CheckpointHandler
else:
    _Optimizer = tp.Any
    _CheckpointHandler = tp.Any


_M = tp.TypeVar("_M", bound=eqx.Module)
_O = tp.TypeVar("_O", bound=_Optimizer)
_Mode = tp.Literal["min", "max"]


class ModelCheckpoint(Callback):
    """Checkpoint callback that stores model and optimizer state via Orbax."""

    def __init__(
        self,
        directory: str | Path,
        *,
        monitor: str | None = None,
        mode: _Mode = "min",
        save_interval_steps: int | None = None,
        save_on: tp.Literal["train", "eval"] = "eval",
        save_on_end: bool = True,
        max_to_keep: int | None = 1,
        keep_every_n_steps: int | None = None,
        save_optimizer: bool = True,
        save_metrics: bool = True,
        metadata: Mapping[str, tp.Any] | None = None,
        use_ocdbt: bool = True,
        use_zarr3: bool = True,
        enable_async_checkpointing: bool = True,
        item_prefix: str | None = "checkpoint",
        preservation_policy: tp.Any | None = None,
        save_decision_policy: tp.Any | None = None,
    ) -> None:
        if ocp is None or ocp_cm is None:  # pragma: no cover - exercised when orbax missing
            raise ImportError(
                "orbax.checkpoint is required for ModelCheckpoint"
            ) from _ORBAX_IMPORT_ERROR

        if mode not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")
        if save_on not in {"train", "eval"}:
            raise ValueError("save_on must be 'train' or 'eval'")
        if save_interval_steps is not None and save_interval_steps <= 0:
            raise ValueError("save_interval_steps must be positive when provided")
        if max_to_keep is not None and max_to_keep <= 0:
            raise ValueError("max_to_keep must be positive when provided")
        if keep_every_n_steps is not None and keep_every_n_steps <= 0:
            raise ValueError("keep_every_n_steps must be positive when provided")
        if monitor is not None and not save_metrics:
            raise ValueError("save_metrics must be True when monitor is set")
        if save_on == "train" and monitor is not None:
            raise ValueError("Monitoring metrics is only supported for eval checkpointing")

        self._directory = Path(directory).expanduser().resolve()
        self._directory.mkdir(parents=True, exist_ok=True)

        self.monitor = monitor
        self.mode = mode
        self.save_interval_steps = save_interval_steps
        self.save_on = save_on
        self.save_on_end = save_on_end
        self.max_to_keep = max_to_keep
        self.keep_every_n_steps = keep_every_n_steps
        self.save_optimizer = save_optimizer
        self.save_metrics = save_metrics
        self.metadata = _maybe_to_serialisable(metadata)
        self.use_ocdbt = use_ocdbt
        self.use_zarr3 = use_zarr3
        self.enable_async_checkpointing = enable_async_checkpointing
        self.item_prefix = item_prefix
        self._external_preservation_policy = preservation_policy
        self._save_decision_policy = save_decision_policy

        self._best_metric: float | None = None
        self._best_step: int | None = None
        self._last_saved_step: int | None = None
        self._last_eval_metrics: dict[str, float] | None = None
        self._last_train_metrics: dict[str, float] | None = None
        self._should_minimise = mode == "min"

        self._manager, self._item_names = self._build_manager()

    def on_training_step_end(
        self,
        module: _M,
        optimizer: _O,
        batch: tp.Any,
        aux: Mapping[str, tp.Any],
        logs: Mapping[str, tp.Any],
        step: int,
    ) -> None:
        metrics = _clean_metrics(logs or aux)
        self._last_train_metrics = metrics
        if self.save_on != "train":
            return
        if not self._periodic_trigger(step):
            return
        self._save_step(
            step,
            module,
            optimizer,
            metrics=metrics,
            reason="train",
        )

    def on_validation_step_end(
        self,
        *,
        module: _M,
        optimizer: _O,
        batch: tp.Any,
        aux: Mapping[str, tp.Any] | None = None,
        step_idx: int,
        **_: tp.Any,
    ) -> None:
        del batch
        metrics = _clean_metrics(aux)
        if not metrics:
            return
        self._last_eval_metrics = metrics
        if self.save_on != "eval":
            return
        if self.monitor is None:
            self._save_step(step_idx, module, optimizer, metrics=metrics, reason="eval")
            return
        metric_value = metrics.get(self.monitor)
        if metric_value is None:
            return
        if self._best_metric is None or _is_improvement(
            current=metric_value,
            best=self._best_metric,
            minimise=self._should_minimise,
        ):
            self._best_metric = metric_value
            self._best_step = step_idx
            self._save_step(step_idx, module, optimizer, metrics=metrics, reason="metric")

    def on_validation_end(
        self,
        *,
        module: _M,
        optimizer: _O,
        **_: tp.Any,
    ) -> None:
        del module, optimizer

    def on_training_end(
        self,
        *,
        module: _M,
        optimizer: _O,
        step_idx: int | None = None,
        aux: Mapping[str, tp.Any] | None = None,
        metrics: Mapping[str, tp.Any] | None = None,
        **_: tp.Any,
    ) -> None:
        if not self.save_on_end:
            self._finalize_manager()
            return
        if step_idx is not None and self._last_saved_step == step_idx:
            self._finalize_manager()
            return
        if metrics is None:
            metrics = self._last_eval_metrics if self.save_on == "eval" else self._last_train_metrics
        if metrics is None and aux is not None:
            metrics = _clean_metrics(aux)
        final_step = step_idx if step_idx is not None else (self._last_saved_step or 0)
        self._save_step(final_step, module, optimizer, metrics=metrics, reason="final")
        self._finalize_manager()

    @property
    def manager(self) -> ocp.CheckpointManager:
        return self._manager

    def latest_step(self) -> int | None:
        return self._manager.latest_step()

    def best_step(self) -> int | None:
        return self._best_step

    def restore(
        self,
        *,
        step: int | None = None,
        module: _M | None = None,
        optimizer: _O | None = None,
    ) -> tuple[_M | None, _O | None, int | None, dict[str, float]]:
        restore_args: dict[str, tp.Any] = {}
        if "module" in self._item_names:
            if module is None:
                raise ValueError("module must be provided for restore")
            restore_args["module"] = ocp.args.PyTreeRestore(module)
        if "optimizer" in self._item_names:
            if optimizer is None:
                raise ValueError("optimizer must be provided for restore")
            restore_args["optimizer"] = ocp.args.PyTreeRestore(optimizer)
        if "step" in self._item_names:
            restore_args["step"] = ocp.args.ArrayRestore(jnp.asarray(0, dtype=jnp.int64))
        if "metrics" in self._item_names:
            restore_args["metrics"] = ocp.args.JsonRestore({})
        if "metadata" in self._item_names:
            restore_args["metadata"] = ocp.args.JsonRestore({})

        composite = ocp.args.Composite(**restore_args)
        target_step = step if step is not None else self._manager.latest_step()
        if target_step is None:
            return module, optimizer, None, {}

        restored = self._manager.restore(target_step, args=composite)
        new_module = restored.get("module", module)
        new_optimizer = restored.get("optimizer", optimizer)
        step_value = restored.get("step")
        metrics = restored.get("metrics", {})
        meta = restored.get("metadata")
        if meta is not None:
            self.metadata = meta
        if step_value is not None:
            step_int = int(jnp.asarray(step_value))
        else:
            step_int = target_step
        return new_module, new_optimizer, step_int, _clean_metrics(metrics)

    def _build_manager(self) -> tuple[ocp.CheckpointManager, tuple[str, ...]]:
        item_handlers: dict[str, _CheckpointHandler] = {
            "module": ocp.PyTreeCheckpointHandler(
                use_ocdbt=self.use_ocdbt,
                use_zarr3=self.use_zarr3,
            )
        }
        if self.save_optimizer and "optimizer" not in item_handlers:
            item_handlers["optimizer"] = ocp.PyTreeCheckpointHandler(
                use_ocdbt=self.use_ocdbt,
                use_zarr3=self.use_zarr3,
            )
        if "step" not in item_handlers:
            item_handlers["step"] = ocp.ArrayCheckpointHandler()
        if self.save_metrics and "metrics" not in item_handlers:
            item_handlers["metrics"] = ocp.JsonCheckpointHandler()
        if self.metadata is not None and "metadata" not in item_handlers:
            item_handlers["metadata"] = ocp.JsonCheckpointHandler()

        item_names = list(item_handlers.keys())

        policy = self._external_preservation_policy
        if policy is None:
            policy = self._default_preservation_policy()

        options = ocp.CheckpointManagerOptions(
            save_interval_steps=self.save_interval_steps or 1,
            step_prefix=self.item_prefix,
            enable_async_checkpointing=self.enable_async_checkpointing,
            preservation_policy=policy,
            save_decision_policy=self._save_decision_policy,
        )
        manager = ocp.CheckpointManager(
            directory=self._directory.as_posix(),
            item_names=tuple(item_names),
            item_handlers=item_handlers,
            options=options,
            metadata=self.metadata,
        )
        return manager, tuple(item_names)

    def _default_preservation_policy(self):
        policies: list[tp.Any] = []
        if self.keep_every_n_steps is not None:
            policies.append(ocp_cm.EveryNSteps(self.keep_every_n_steps))
        if self.monitor is not None:
            reverse = self.mode == "max"

            def _metric_fn(metrics: Mapping[str, tp.Any]) -> float:
                value = metrics.get(self.monitor)
                if value is None:
                    raise KeyError(
                        f"Missing monitored metric '{self.monitor}' in checkpoint metrics"
                    )
                return float(value)

            policies.append(
                ocp_cm.BestN(
                    get_metric_fn=_metric_fn,
                    reverse=reverse,
                    n=self.max_to_keep,
                    keep_checkpoints_without_metrics=False,
                )
            )
        if self.max_to_keep is not None:
            policies.append(ocp_cm.LatestN(self.max_to_keep))

        if not policies:
            return None
        if len(policies) == 1:
            return policies[0]
        return ocp_cm.AnyPreservationPolicy(policies)

    def _periodic_trigger(self, step: int) -> bool:
        if self.save_on != "train":
            return False
        interval = self.save_interval_steps
        if interval is None or interval <= 0:
            return False
        return step % interval == 0

    def _save_step(
        self,
        step: int,
        module: tp.Any,
        optimizer: tp.Any,
        *,
        metrics: Mapping[str, float] | None,
        reason: str,
    ) -> None:
        if self._last_saved_step == step:
            return
        clean_metrics = dict(metrics or {})
        items: dict[str, tp.Any] = {
            "module": ocp.args.PyTreeSave(module),
            "step": ocp.args.ArraySave(jnp.asarray(step, dtype=jnp.int64)),
        }
        if self.save_optimizer:
            items["optimizer"] = ocp.args.PyTreeSave(optimizer)
        if self.save_metrics:
            items["metrics"] = ocp.args.JsonSave(clean_metrics)
        if self.metadata is not None:
            items["metadata"] = ocp.args.JsonSave(self.metadata)

        if self.monitor is not None and self.monitor not in clean_metrics:
            monitored_value = self._best_metric if reason == "metric" else None
            if monitored_value is not None:
                clean_metrics = dict(clean_metrics)
                clean_metrics[self.monitor] = monitored_value

        self._manager.save(step, args=ocp.args.Composite(**items), metrics=clean_metrics)
        self._last_saved_step = step

    def _finalize_manager(self) -> None:
        try:
            self._manager.wait_until_finished()
        finally:
            self._manager.close()


def _is_improvement(*, current: float, best: float, minimise: bool) -> bool:
    if math.isnan(current):
        return False
    if math.isnan(best):
        return True
    return current < best if minimise else current > best


def _clean_metrics(metrics: Mapping[str, tp.Any] | None) -> dict[str, float]:
    if not metrics:
        return {}
    result: dict[str, float] = {}
    for key, value in metrics.items():
        converted = _to_float(value)
        if converted is None:
            continue
        result[str(key)] = converted
    return result


def _to_float(value: tp.Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (jax.Array, jnp.ndarray)) and value.ndim == 0:
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _maybe_to_serialisable(metadata: Mapping[str, tp.Any] | None) -> dict[str, tp.Any] | None:
    if metadata is None:
        return None
    serialisable: dict[str, tp.Any] = {}
    for key, value in metadata.items():
        if isinstance(value, Mapping):
            nested = _maybe_to_serialisable(value)  # type: ignore[arg-type]
            serialisable[key] = nested
        elif isinstance(value, (str, int, float, bool)) or value is None:
            serialisable[key] = value
        else:
            serialisable[key] = str(value)
    return serialisable


__all__ = ["ModelCheckpoint"]
