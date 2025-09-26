from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import Logger


@dataclass
class CSVLogger(Logger):
    log_dir: str | Path
    experiment_name: str | Path
    log_interval: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        Logger.__init__(self, enabled=self.enabled)
        if not self.enabled:
            return
        self.log_dir = str(self.log_dir)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir, self.experiment_name).mkdir(parents=True, exist_ok=True)
        self._path = Path(self.log_dir) / self.experiment_name
        self.first_step = {}

    def log_metrics(self, metrics: dict[str, float], *, step: int, mode: str) -> bool:
        if not self.enabled:
            return False
        if not metrics:
            return False
        if self.log_interval <= 0 or step % self.log_interval != 0:
            return False

        filename = self._path / f"{mode}.csv"

        if self.first_step.get(mode, True):
            with filename.open("a", encoding="utf-8") as f:
                headers = ",".join(["step"] + list(metrics.keys()))
                f.write(headers + "\n")
            self.first_step[mode] = False

        line = ",".join([str(step)] + [str(metric) for metric in metrics.values()])
        with filename.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
