from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv

from .base_logger import Logger, LoggerTool, LoggerToolsConfig


HostMetrics = dict[str, Any]


@dataclass(kw_only=True)
class FileLoggerConfig(LoggerToolsConfig):
    log_step_key: str = "step"
    log_epoch_key: str = "epoch"
    config_format: str = "json"  # json only for now
    log_dir: str = "file_logs"

    def create(self, logger: Logger) -> "LoggerTool":
        return FileLogger(self, logger)


class FileLogger(LoggerTool):
    def __init__(self, config: FileLoggerConfig, logger: Logger):
        self.config = config
        self.logger = logger
        base: Path | None = self.logger.log_path
        if base is None:
            base = Path("./logs")
        self.log_path = base / self.config.log_dir
        self.log_path.mkdir(parents=True, exist_ok=True)
        self._buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def log_config(self, config: dict | Any):
        obj = config
        try:
            if hasattr(config, "to_dict"):
                obj = config.to_dict()
            elif isinstance(config, dict):
                obj = {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in config.items()}
        except Exception:
            obj = config
        if self.config.config_format == "json":
            with open(self.log_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=str)

    def setup(self):
        self._buffers.clear()

    def log_metrics(self, metrics: HostMetrics, step: int, epoch: int, mode: str):
        rec = dict(metrics)
        rec[self.config.log_step_key] = step
        rec[self.config.log_epoch_key] = epoch
        self._buffers[mode].append(rec)

    def finalize(self, status: str):
        del status
        for mode, rows in self._buffers.items():
            if not rows:
                continue
            keys = sorted({k for r in rows for k in r.keys()})
            path = self.log_path / f"metrics_{mode}.csv"
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for r in rows:
                    writer.writerow({k: r.get(k) for k in keys})
