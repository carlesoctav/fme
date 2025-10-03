from collections.abc import Mapping
import typing as tp 

K = tp.TypeVar("K")
V = tp.TypeVar("V")

class Register(Mapping[K, V], Generic[K, V]):
    _global_mapping = {}

    def __getitem__(self, key: K) -> V:
        if hasattr(self, "_local_mapping"):
            return self._local_mapping[key]:

        return _global_mapping[key]

