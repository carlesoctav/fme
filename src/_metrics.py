from __future__ import annotations

import typing as tp

import jax.tree_util as jtu


def _is_tuple_leaf(x: tp.Any) -> bool:
    return isinstance(x, tuple)


class SufficientMetric:
    def __init__(
        self,
        *,
        log_every_n_steps: int | None = None,
        reduce_fn: tp.Callable[[tp.Any], tp.Any] | None = None,
    ) -> None:
        self.log_every_n_steps = log_every_n_steps or 0
        self.reduce_fn = reduce_fn
        self._buffer: tp.Any = None
        self._last_added: tp.Any = None
        self.reduced_buffer: dict[int, dict[str, float]] = {}
        self.count = 0

    def __iadd__(self, other: tp.Any) -> SufficientMetric:
        return self.__add__(other)

    def __add__(self, other: tp.Any) -> SufficientMetric:
        if other is None:
            return self

        other_tree = jtu.tree_map(lambda x: x, other)

        if self._buffer is None:
            self._buffer = jtu.tree_map(lambda x: x, other_tree)
        else:
            self._buffer = jtu.tree_map(lambda a, b: a + b, self._buffer, other_tree)

        if self._total_buffer is None:
            self._total_buffer = jtu.tree_map(lambda x: x, other_tree)
        else:
            self._total_buffer = jtu.tree_map(lambda a, b: a + b, self._total_buffer, other_tree)

        self._last_added = other_tree
        self.count+=1
        return self

    def step_metrics(self) -> dict[str, float]:
        if self._last_added is None:
            return {}
        reduced_tree = self._apply_reduce(self._last_added)
        return reduced_tree 

    def per_N_metrics(self, step: int, skip_check = False) -> dict[str, float]:
        if not skip_check:
            if self.log_every_n_steps <= 0:
                return {}
            if step % self.log_every_n_steps != 0 or self._buffer is None or self._window_steps == 0:
                return {}

        reduced_tree = self._apply_reduce(self._buffer)
        flattened = self._flatten(reduced_tree)
        averaged = {
            f"{key}_per_N": value for key,
            value in flattened.items()
        }
        self.reduced_buffer[step] = averaged
        self._buffer = None
        self._window_steps = 0
        return averaged

    def _apply_reduce(self, tree: tp.Any) -> tp.Any:
        if tree is None:
            return {}

        def _reduce_leaf(x: tp.Any) -> tp.Any:
            if isinstance(x, tuple) and len(x) == 2 and x[1] not in (0, None):
                return x[0] / x[1]
            if isinstance(x, tuple):
                try:
                    return x[0]
                except Exception:  
                    return x
            return x

        if self.reduce_fn is not None:
            reduced = self.reduce_fn(tree)
        else:
            reduced = jtu.tree_map(_reduce_leaf, tree, is_leaf=_is_tuple_leaf)

        return reduced

