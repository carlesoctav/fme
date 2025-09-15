from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import jax


HostMetrics = dict[str, Any]


@dataclass(kw_only=True)
class LoggerConfig:
    log_every_n_steps: int = 1
    log_path: Path | None = None
    log_tools: list["LoggerToolsConfig"] = field(default_factory=list)
    cmd_logging_name: str = "output"

    @property
    def log_dir(self) -> str | None:
        return None if self.log_path is None else self.log_path.as_posix()


@dataclass(kw_only=True)
class LoggerToolsConfig:
    def create(self, logger: "Logger") -> "LoggerTool":
        raise NotImplementedError


class Logger:
    def __init__(self, config: LoggerConfig, metric_postprocess_fn: Callable[[HostMetrics], HostMetrics] | None = None):
        self.config = config
        self.log_path = config.log_path
        if self.log_path is not None:
            self.log_path.mkdir(parents=True, exist_ok=True)
        self.metric_postprocess_fn = metric_postprocess_fn or (lambda x: x)
        # Single-process default; optionally replicate per-process
        if jax.process_index() == 0:
            self.log_tools = [tool_config.create(self) for tool_config in config.log_tools]
        else:
            self.log_tools = []
        self.epoch = 0
        self._mode_stack: list[Literal["default","train","val","test"]] = []
        self._epoch_time_stack: list[float] = []

    @property
    def mode(self) -> Literal["default","train","val","test"]:
        return self._mode_stack[-1] if self._mode_stack else "default"

    def log_config(self, config: dict | Any):
        # Optional: try to serialize
        try:
            obj = config
            if hasattr(config, "to_dict"):
                obj = config.to_dict()
            elif isinstance(config, dict):
                obj = {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in config.items()}
            txt = json.dumps(obj, indent=2, default=str)
        except Exception:
            txt = str(config)
        for tool in self.log_tools:
            tool.log_config(obj if "obj" in locals() else config)

    def on_training_start(self):
        for tool in self.log_tools:
            tool.setup()

    def start_epoch(self, epoch: int, step: int, mode: Literal["train","val","test"] = "train"):
        del step
        self.epoch = epoch
        self._mode_stack.append(mode)
        self._epoch_time_stack.append(time.time())

    def log_host_metrics(self, host_metrics: HostMetrics, step: int, mode: str | None = None):
        host_metrics = self.metric_postprocess_fn(host_metrics)
        for tool in self.log_tools:
            tool.log_metrics(host_metrics, step, self.epoch, mode or self.mode)

    def end_epoch(self, metrics: HostMetrics | None, step: int):
        del step
        epoch_time = None
        if self._epoch_time_stack:
            start = self._epoch_time_stack.pop()
            epoch_time = time.time() - start
        if metrics is None:
            metrics = {}
        if epoch_time is not None:
            metrics = dict(metrics)
            metrics["epoch_time"] = epoch_time
        mode = self._mode_stack.pop() if self._mode_stack else "default"
        for tool in self.log_tools:
            tool.log_metrics(metrics, step, self.epoch, mode)

    def finalize(self, status: str = "success"):
        for tool in self.log_tools:
            tool.finalize(status)


class LoggerTool:
    def log_config(self, config: dict | Any):
        pass

    def setup(self):
        pass

    def log_metrics(self, metrics: HostMetrics, step: int, epoch: int, mode: str):
        raise NotImplementedError

    def finalize(self, status: str):
        pass

