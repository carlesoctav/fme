from __future__ import annotations

import logging
import typing as tp
import warnings
from collections.abc import Mapping

import equinox as eqx
import orbax.checkpoint as ocp
from etils import epath
from orbax.checkpoint import checkpoint_managers as ocp_cm

if tp.TYPE_CHECKING:
    from ..training_utils import Optimizer


LOGGER = logging.getLogger(__name__)


_M = tp.TypeVar("_M")
_O = tp.TypeVar("_O")
_Mode = tp.Literal["min", "max"]


class ModelCheckpoint:
    """Checkpoint callback that stores model and optimizer state via Orbax."""

    def __init__(
        self,
        directory: str | epath.Path,
        run_name: str = "",
        save_interval_steps: int | None = None,
        preservation_policy: ocp_cm.PreservationPolicy | None = None,
        save_decision_policy: ocp_cm.SaveDecisionPolicy | None = None,
        save_on: tp.Literal["train", "eval"] = "train",
        save_on_training_end: bool = True,
        save_optimizer: bool = True,
        save_hparams: bool = False,
        save_data: bool = False,
    ) -> None:
        if save_on not in {"train", "eval"}:
            raise ValueError("save_on must be 'train' or 'eval'")

        if save_interval_steps is not None and save_decision_policy is not None:
            warnings.warn(
                "Both save_interval_steps and save_decision_policy are set; save_decision_policy will be used."
            )
            save_interval_steps = None
            if save_interval_steps is not None and save_interval_steps <= 0:
                raise ValueError("save_interval_steps must be positive when provided")

        self._directory = (
            directory if isinstance(directory, epath.Path) else epath.Path(directory)
        )

        self._directory = self._directory / run_name if run_name else self._directory

        self._directory.mkdir(parents=True, exist_ok=True)

        self._save_interval_steps = save_interval_steps
        self._save_decision_policy = save_decision_policy

        self.save_on = save_on
        self.save_on_training_end = save_on_training_end

        self.save_optimizer = save_optimizer
        self.save_hparams = save_hparams
        self.save_data = save_data

        self._preservation_policy = preservation_policy
        self._manager = self._build_manager()

    def on_training_step(self, module, optimizer, batch, logs, logger, step) -> None:
        if self.save_on != "train":
            return

        self.save(module, optimizer, logs, step)

    def on_validation_end(
        self,
        module,
        optimizer,
        logs,
        logger,
        step,
    ) -> None:
        if self.save_on != "eval":
            return

        self.save(module, optimizer, logs, step)

    def on_training_end(self, module, optimizer, logs, logger, step) -> None:
        if not self.save_on_training_end:
            self.finish()
            return

        self.save(module, optimizer, logs, 9999999, force=True)
        self.finish()

    def _build_manager(self) -> tuple[ocp.CheckpointManager, tuple[str, ...]]:
        options = ocp.CheckpointManagerOptions(
            save_decision_policy=self._save_decision_policy,
            preservation_policy=self._preservation_policy,
            save_interval_steps=self._save_interval_steps,
        )

        manager = ocp.CheckpointManager(
            directory=self._directory,
            options=options,
        )
        return manager

    def save(
        self,
        module: tp.Any,
        optimizer: Optimizer,
        logs: Mapping[str, float] | None,
        step: int,
        force=False,
    ) -> None:
        items: dict[str, tp.Any] = {
            "module": ocp.args.PyTreeSave(eqx.filter(module, eqx.is_array)),
            "metrics": ocp.args.JsonSave(logs),
        }

        if self.save_optimizer:
            items["optimizer"] = ocp.args.PyTreeSave(optimizer.opt_state)

        if self.save_data:
            raise NotImplementedError("not yet implemented to store the iterator state")
            # items["data"] = grain.checkpoint.CheckpointSave(self.iterator)

        self._manager.save(
            step, args=ocp.args.Composite(**items), metrics=logs, force=force
        )

    def finish(self) -> None:
        try:
            self._manager.wait_until_finished()
        finally:
            self._manager.close()


__all__ = ["ModelCheckpoint"]
