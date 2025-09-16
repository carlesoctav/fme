import dataclasses as dc
import fnmatch
import typing as tp

import equinox as eqx

Path = tuple[str | int, ...]


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
    """
    Replace matched submodules using shell-style glob patterns.
    Multiple patterns may match the same path; transforms are applied in the order they are inserted into the dictionary. When this happens, only the first match is applied.
    """

    path_to_info: dict[str, tuple[Path, list[tp.Callable[[tp.Any], tp.Any]]]] = {}

    replacements = []
    getters = []

    for path, sub_module in iter_module(module):
        path_str = _path_to_str(path)
        for pattern, transform in pattern_to_transform.items():
            if not fnmatch.fnmatchcase(path_str, pattern):
                continue
            else:
                replacements.append(transform(sub_module))
                getters.append(_getter_from_path(path))
                break

    for where, replacement in zip(getters, replacements):
        module = eqx.tree_at(where, module, replacement)

    return module
