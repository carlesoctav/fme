import hashlib
from typing import Dict, Iterable

import jax
from jaxtyping import PRNGKeyArray


class Rngs:
    """Simple RNG manager keyed by stream name.

    Users construct it with named base keys, e.g. ``Rngs(params=key)`` or
    ``Rngs(params=key_params, dropout=key_dropout)``. Each call to ``make_rng``
    derives a deterministic child key via ``fold_in`` and increments an internal
    counter for that stream. Names that were not provided up-front raise a
    ``ValueError`` when requested.
    """

    __slots__ = ("_keys", "_counters")

    def __init__(self, **named_keys: PRNGKeyArray):
        if not named_keys:
            raise ValueError("Rngs requires at least one named key, e.g. params=...")
        self._keys: Dict[str, PRNGKeyArray] = dict(named_keys)
        self._counters: Dict[str, int] = {name: 0 for name in named_keys}

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
