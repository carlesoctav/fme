import typing as tp
from collections.abc import Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray, Int


PrepareInputFn = tp.Callable[[tp.Any, tuple], tuple]
PrepareOutputFn = tp.Callable[[tp.Any, tp.Any], tp.Any]

_no_hook = object()


class PrepareableModule(eqx.Module):
    """Mixin for modules supporting optional prepare_input/output hooks."""

    prepare_input: PrepareInputFn | None | object = eqx.field(
        default=_no_hook, static=True
    )
    prepare_output: PrepareOutputFn | None | object = eqx.field(
        default=_no_hook, static=True
    )

    def maybe_prepare_input(self, args: tuple) -> tuple:
        fn = self.prepare_input
        if fn is None or fn is _no_hook:
            return args

        result = fn(self, args)
        if not isinstance(result, tuple):
            raise TypeError("prepare_input must return a tuple of arguments")
        return result

    def maybe_prepare_output(self, output: tp.Any) -> tp.Any:
        fn = self.prepare_output
        if fn is None or fn is _no_hook:
            return output
        return fn(self, output)


def replace_prepare_hooks(
    module: PrepareableModule,
    *,
    prepare_input: PrepareInputFn | None = None,
    prepare_output: PrepareOutputFn | None = None,
) -> PrepareableModule:
    """Return a new module with updated prepare hooks."""

    import copy

    if prepare_input is None and prepare_output is None:
        return module

    new_module = copy.copy(module)

    if prepare_input is not None:
        object.__setattr__(new_module, "prepare_input", prepare_input)
    if prepare_output is not None:
        object.__setattr__(new_module, "prepare_output", prepare_output)

    return new_module


class Rngs(eqx.Module):
    _keys: dict[str, PRNGKeyArray]
    _counters: dict[str, Int[jax.Array, ""]] 

    def __init__(self, **named_keys: PRNGKeyArray):
        if not named_keys:
            raise ValueError("Rngs requires at least one named key, e.g. params=...")

        keys: dict[str, PRNGKeyArray] = {}
        for name, key in named_keys.items():
            if not isinstance(key, jax.Array):
                raise TypeError(
                    "Rngs expects JAX PRNGKey arrays; received type "
                    f"{type(key)!r} for stream '{name}'."
                )
            keys[name] = key


        self._keys = keys
        self._counters = {name: jnp.asarray(0, dtype = jnp.int32) for name in named_keys}


    def make_rng(self, name: str) -> PRNGKeyArray:
        if name not in self._keys:
            available = ", ".join(sorted(self._keys)) or "<none>"
            raise ValueError(
                f"Requested RNG stream '{name}' but available streams are: {available}."
            )
        self._counters[name] = self._counters[name] + 1
        return jax.random.fold_in(self._keys[name], self._counters[name])

    def keys(self) -> Iterable[str]:
        return self._keys.keys()

    def split(self, nums: int) -> tp.Sequence["Rngs"]:
        split_keys: dict[str, jax.Array] = {
            name: jax.random.split(key, nums) for name, key in self._keys.items()
        }

        results: list[Rngs] = []
        for i in range(nums):
            new_keys = {name: value[i] for name, value in split_keys.items()}
            results.append(Rngs(**new_keys))

        return tuple(results)

def split_rngs(rngs: Rngs, nums: int) -> tp.Sequence[Rngs]:
    split_keys: dict[str, jax.Array] = {
        name: jax.random.split(key, nums) for name, key in rngs._keys.items()
    }
    results: list[Rngs] = []
    for i in range(nums):
        new_keys = {name: value[i] for name, value in split_keys.items()}
        results.append(Rngs(**new_keys))

    return tuple(results)
