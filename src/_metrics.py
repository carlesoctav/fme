from __future__ import annotations

import typing as tp

import logging
import jax.tree_util as jtu

LOGGER = logging.getLogger(__name__)


def _is_tuple_leaf(x: tp.Any) -> bool:
    return isinstance(x, tuple)


class SufficientMetric:
    def __init__(
        self,
        *,
        name: str,
        log_every_n_steps: int | None = None,
    ) -> None:
        self.name = name
        self.log_every_n_steps = log_every_n_steps or 0
        self._buffer_tree: tp.Any = None
        self._last_added: tp.Any = None
        self.per_N_metrics_buffer: dict[int, dict[str, float]] = {}
        self._count = 0
        self._show_log_every_n_steps_warning = False

    def __iadd__(self, other: tp.Any) -> SufficientMetric:
        return self.__add__(other)


    def reduce_fn(self, tree, count) -> tp.Callable[[tp.Any], tp.Any] | None:
        def _reduce_leaf(x: tp.Any) -> tp.Any:
            if isinstance(x, tuple) and len(x) == 2:
                value, normaliser = x
                if normaliser in (0, None):
                    return float(value / count)
                return float(value / normaliser)
            return float(x)
        return jtu.tree_map(_reduce_leaf, tree, is_leaf=_is_tuple_leaf)

    def add(self, other: tp.Any) -> SufficientMetric:
        if other is None:
            return self

        self._last_added = other
        self._count += 1

        if self._buffer_tree is None:
            self._buffer_tree = other
        else:
            self._buffer_tree = jtu.tree_map(
                lambda a, b: a + b,
                self._buffer_tree,
                other,
            )
        return self

    def __add__(self, other: tp.Any) -> SufficientMetric:
        return self.add(other)

    def step_metrics(self) -> dict[str, float]:
        if self._last_added is None:
            return {}
        reduced = self.reduce_fn(self._last_added, count=1)
        reduced = {f"{self.name}/{k}": v for k, v in reduced.items()}
        return reduced 

    def per_N_metrics(self, step: int, *, skip_check: bool = False) -> dict[str, float]:
        if self.per_N_metrics_buffer.get(step) is not None:
            return self.Per_N_metrics_buffer[step]

        if self._buffer_tree is None:
            return {}

        if not skip_check:
            if self.log_every_n_steps <= 0:
                if not self._show_log_every_n_steps_warning:
                    self._show_log_every_n_steps_warning = True
                    LOGGER.warning(
                        f"Metric {self.name} has log_every_n_steps={self.log_every_n_steps}, skipping per_N logging."
                    )
                return {}
            if self._count == 0 or step % self.log_every_n_steps != 0:
                LOGGER.debug(
                    f"Skipping per_{self.log_every_n_steps}_steps logging for metric {self.name} at step {step}."
                )
                return {}

        reduced = self.reduce_fn(self._buffer_tree, count=self._count)
        self.per_N_metrics_buffer[step] = {**reduced, "count": self._count}
        self._buffer_tree = None
        self._count = 0
        reduced = {f"{self.name}_per_N/{k}": v for k, v in reduced.items()}
        return reduced 

    def summary(self) -> dict[str, tp.Any]:
        return {
            "name": self.name,
            "count": self._count,
            "per_N_cache": dict(self.per_N_metrics_buffer),
        }

    def _reduce_tree(self, tree: tp.Any, count = 1) -> tp.Any:
        if tree is None:
            return {}

        return self.reduce_fn(tree)




__all__ = ["SufficientMetric"]
