from __future__ import annotations

import types
import typing as tp

import pytest

from src.callbacks import model_checkpoint as mc


class _DummyHandler:
    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _DummyComposite(dict):
    def __init__(self, **items: tp.Any) -> None:
        super().__init__(items)


class _DummyArgs:
    @staticmethod
    def PyTreeSave(value: tp.Any) -> tp.Any:
        return value

    @staticmethod
    def ArraySave(value: tp.Any) -> tp.Any:
        return value

    @staticmethod
    def JsonSave(value: tp.Any) -> dict[str, tp.Any]:
        return dict(value)

    @staticmethod
    def PyTreeRestore(template: tp.Any) -> tp.Any:
        return template

    @staticmethod
    def ArrayRestore(value: tp.Any) -> tp.Any:
        return value

    @staticmethod
    def JsonRestore(value: tp.Any) -> dict[str, tp.Any]:
        return dict(value)

    Composite = _DummyComposite


class _DummyOptions:
    def __init__(self, **kwargs: tp.Any) -> None:
        self.kwargs = kwargs


class _DummyCheckpointManager:
    STORE: dict[str, dict[int, dict[str, tp.Any]]] = {}

    def __init__(
        self,
        directory: str,
        item_names: tuple[str, ...],
        item_handlers: dict[str, tp.Any],
        options: _DummyOptions,
        metadata: dict[str, tp.Any] | None = None,
    ) -> None:
        self.directory = directory
        self.item_names = item_names
        self.item_handlers = item_handlers
        self.options = options
        self.metadata = metadata
        self.save_calls: list[tuple[int, dict[str, float]]] = []
        self.wait_called = False
        self.closed = False
        self.STORE.setdefault(directory, {})

    def save(self, step: int, args: _DummyComposite, metrics: dict[str, float]) -> None:
        data = {name: value for name, value in args.items()}
        data["__metrics__"] = dict(metrics)
        self.STORE[self.directory][int(step)] = data
        self.save_calls.append((int(step), dict(metrics)))

    def latest_step(self) -> int | None:
        store = self.STORE[self.directory]
        return max(store) if store else None

    def restore(self, step: int, args: _DummyComposite) -> dict[str, tp.Any]:
        store = self.STORE[self.directory]
        state = store.get(int(step), {})
        result: dict[str, tp.Any] = {}
        for name in args.keys():
            if name == "metrics":
                result[name] = dict(state.get("__metrics__", {}))
            else:
                result[name] = state.get(name)
        return result

    def wait_until_finished(self) -> None:
        self.wait_called = True

    def close(self) -> None:
        self.closed = True


_DUMMY_ORBAX = types.SimpleNamespace(
    args=_DummyArgs,
    CheckpointManagerOptions=_DummyOptions,
    CheckpointManager=_DummyCheckpointManager,
    PyTreeCheckpointHandler=_DummyHandler,
    ArrayCheckpointHandler=_DummyHandler,
    JsonCheckpointHandler=_DummyHandler,
)


class _DummyPolicy:
    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        self.args = args
        self.kwargs = kwargs


class _DummyAnyPolicy(_DummyPolicy):
    def __init__(self, policies: tp.Sequence[_DummyPolicy]):
        super().__init__(policies=list(policies))
        self.policies = list(policies)


_DUMMY_OCM = types.SimpleNamespace(
    LatestN=lambda n=None: _DummyPolicy(n=n),
    EveryNSteps=lambda n: _DummyPolicy(n=n),
    BestN=lambda **kwargs: _DummyPolicy(**kwargs),
    AnyPreservationPolicy=lambda policies: _DummyAnyPolicy(policies),
)


@pytest.fixture(autouse=True)
def _patch_orbax(monkeypatch: pytest.MonkeyPatch) -> None:
    _DummyCheckpointManager.STORE.clear()
    monkeypatch.setattr(mc, "ocp", _DUMMY_ORBAX, raising=False)
    monkeypatch.setattr(mc, "ocp_cm", _DUMMY_OCM, raising=False)
    monkeypatch.setattr(mc, "_ORBAX_IMPORT_ERROR", None, raising=False)


def test_model_checkpoint_saves_and_restores(tmp_path) -> None:
    callback = mc.ModelCheckpoint(
        directory=tmp_path,
        save_interval_steps=2,
        monitor="val_loss",
        metadata={"run": "demo"},
    )

    module = {"weights": [1, 2, 3]}
    optimizer = {"state": 0}

    assert isinstance(callback.manager.options.kwargs["preservation_policy"], _DummyPolicy)
    assert callback.manager.metadata == {"run": "demo"}

    callback.on_training_step_end(module=module, optimizer=optimizer, step=1, metrics={"loss": 1.0})
    callback.on_training_step_end(module=module, optimizer=optimizer, step=2, metrics={"loss": 0.9})
    callback.on_validation_end(module=module, optimizer=optimizer, step=2, metrics={"val_loss": 0.5})
    callback.on_validation_end(module=module, optimizer=optimizer, step=3, metrics={"val_loss": 0.6})
    callback.on_training_end(module=module, optimizer=optimizer, step=5)

    assert [s for s, _ in callback.manager.save_calls] == [2, 5]
    assert callback.manager.wait_called
    assert callback.manager.closed
    assert callback.best_step() == 2

    restore_callback = mc.ModelCheckpoint(
        directory=tmp_path,
        save_on_end=False,
    )
    restored_module, restored_opt, step, metrics = restore_callback.restore(
        step=2,
        module={"weights": None},
        optimizer={"state": None},
    )

    assert restored_module == module
    assert restored_opt == optimizer
    assert step == 2
    assert metrics["val_loss"] == 0.5

    restore_callback.manager.close()


def test_model_checkpoint_train_mode(tmp_path) -> None:
    callback = mc.ModelCheckpoint(
        directory=tmp_path,
        save_interval_steps=2,
        save_on="train",
        metadata={"mode": "train"},
    )

    module = {"weights": [1, 2, 3]}
    optimizer = {"state": 0}

    for step in range(1, 5):
        metrics = {"train_loss": 1.0 / step}
        callback.on_training_step_end(
            module=module,
            optimizer=optimizer,
            step=step,
            metrics=metrics,
        )

    callback.on_training_end(module=module, optimizer=optimizer, step=4)

    saved_steps = [s for s, _ in callback.manager.save_calls]
    assert saved_steps == [2, 4]
    assert callback.manager.metadata == {"mode": "train"}
    assert callback.best_step() is None

    callback.manager.close()
