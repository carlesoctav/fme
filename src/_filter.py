import dataclasses as dc
import fnmatch
import typing as tp

import equinox as eqx
import jax
import jax.tree_util as jtu


Path = tuple[tp.Union[str, int], ...]


def iter_module(
    obj: tp.Any,
    *,
    include_root: bool = False,
) -> tp.Iterator[tuple[Path, tp.Any]]:
    """Yield (path, leaf) for eqx.Modules and containers.

    Paths are tuples of attribute names / indices. The root path is empty.
    """

    seen: set[int] = set()

    def _recurse(path: Path, node: tp.Any) -> tp.Iterator[tuple[Path, tp.Any]]:
        oid = id(node)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(node, eqx.Module):
            if path or include_root:
                yield (path, node)

        if dc.is_dataclass(node):
            for f in dc.fields(node):
                try:
                    v = getattr(node, f.name)
                except Exception:
                    continue
                if isinstance(v, eqx.Module):
                    yield from _recurse(path + (f.name,), v)
                elif dc.is_dataclass(v):
                    yield from _recurse(path + (f.name,), v)
                elif isinstance(v, (list, tuple)):
                    for i, subv in enumerate(v):
                        yield from _recurse(path + (f.name, i), subv)
                elif isinstance(v, dict):
                    for k, subv in v.items():
                        yield from _recurse(path + (f.name, k), subv)
        elif isinstance(node, (list, tuple)):
            for i, subv in enumerate(node):
                yield from _recurse(path + (i,), subv)
        elif isinstance(node, dict):
            for k, subv in node.items():
                yield from _recurse(path + (k,), subv)

    yield from _recurse((), obj)


def _path_to_str(path: Path) -> str:
    parts: list[str] = []
    for p in path:
        parts.append(str(p))
    return ".".join(parts)


def _getter_from_path(path: Path):
    def get(root):
        node = root
        for p in path:
            if isinstance(p, int):
                node = node[p]
            else:
                node = getattr(node, p)
        return node

    return get


def apply_transforms(
    module: tp.Any,
    pattern_to_transform: dict[str, tp.Callable[[tp.Any], tp.Any]],
) -> tp.Any:
    """Replace matched submodules using shell-style glob patterns.

    - Keys in pattern_to_transform are glob patterns like "*self.query" or
      "encoder.layer.*.attention.self.query".
    - Values are callables taking the matched submodule and returning replacement.
    """

    getters: list[tp.Callable[[tp.Any], tp.Any]] = []
    replacements: list[tp.Any] = []

    for path, leaf in iter_module(module):
        path_str = _path_to_str(path)
        for pat, transform in pattern_to_transform.items():
            if fnmatch.fnmatchcase(path_str, pat):
                getters.append(_getter_from_path(path))
                replacements.append(transform(leaf))
    if not getters:
        return module

    mod = module
    for get, rep in zip(getters, replacements):
        mod = eqx.tree_at(get, mod, rep)
    return mod
