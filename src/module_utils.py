from __future__ import annotations

import typing as tp

import equinox as eqx


PrepareInputFn = tp.Callable[[tp.Any, tuple], tuple]
PrepareOutputFn = tp.Callable[[tp.Any, tp.Any], tp.Any]


class PrepareableModule(eqx.Module):
    """Mixin for modules supporting optional prepare_input/output hooks."""

    prepare_input: PrepareInputFn | None 
    prepare_output: PrepareOutputFn | None 

    def maybe_prepare_input(self, args: tuple) -> tuple:
        fn = self.prepare_input
        if fn is None:
            return args

        result = fn(self, args)
        if not isinstance(result, tuple):
            raise TypeError("prepare_input must return a tuple of arguments")
        return result

    def maybe_prepare_output(self, output: tp.Any) -> tp.Any:
        fn = self.prepare_output
        if fn is None:
            return output
        return fn(self, output)


def replace_prepare_hooks(
    module: PrepareableModule,
    *,
    prepare_input: PrepareInputFn | None = None,
    prepare_output: PrepareOutputFn | None = None,
) -> PrepareableModule:
    """Return module with updated prepare hooks via eqx.tree_at."""

    new_module = module
    if prepare_input is not None:
        new_module = eqx.tree_at(lambda m: m.prepare_input, new_module, prepare_input)
    if prepare_output is not None:
        new_module = eqx.tree_at(lambda m: m.prepare_output, new_module, prepare_output)
    return new_module

