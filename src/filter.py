import dataclasses as dc
import fnmatch
import typing as tp

import equinox as eqx

Path = tuple[str | int, ...]


def iter_module(
    obj: tp.Any,
    *,
    include_root: bool = False,
) -> tp.Iterator[tuple[Path, eqx.Module]]:
    """Yield ``(path, module)`` pairs for nested :class:`eqx.Module` instances.

    Paths mirror ``flax.nnx.Module.iter_modules``: duplicates are skipped and the
    root module is optional. Keys inside the path are attribute names or
    container indices/keys.
    """

    seen: set[int] = set()

    def _is_node(node: tp.Any) -> bool:
        return isinstance(node, eqx.Module) or dc.is_dataclass(node) or isinstance(
            node, (list, tuple, dict)
        )

    def _normalize_key(key: tp.Any) -> str | int:
        if isinstance(key, (str, int)):
            return key
        return str(key)

    def _iter_dataclass(node: tp.Any) -> tp.Iterator[tuple[str, tp.Any]]:
        for field in dc.fields(node):
            try:
                value = getattr(node, field.name)
            except Exception:
                continue
            yield field.name, value

    def _recurse(path: Path, node: tp.Any) -> tp.Iterator[tuple[Path, eqx.Module]]:
        if not _is_node(node):
            return

        oid = id(node)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(node, eqx.Module):
            for name, value in _iter_dataclass(node):
                yield from _recurse(path + (name,), value)
            if path or include_root:
                yield path, node
            return

        if dc.is_dataclass(node):
            for name, value in _iter_dataclass(node):
                yield from _recurse(path + (name,), value)
            return

        if isinstance(node, (list, tuple)):
            for idx, value in enumerate(node):
                yield from _recurse(path + (idx,), value)
            return

        if isinstance(node, dict):
            for key, value in node.items():
                yield from _recurse(path + (_normalize_key(key),), value)
            return

    yield from _recurse((), obj)


def _path_to_str(path: Path) -> str:
    parts: list[str] = []
    for p in path:
        parts.append(str(p))
    return ".".join(parts)

def apply_transforms(
    module: tp.Any,
    pattern_to_transform: dict[str, tp.Callable[[tp.Any], tp.Any]],
) -> tp.Any:
    """
    Replace matched submodules using shell-style glob patterns.
    Multiple patterns may match the same path; transforms are applied in the order they are inserted into the dictionary. When this happens, only the first match is applied.
    """

    matches: list[tuple[Path, tp.Any]] = []

    for path, sub_module in iter_module(module):
        path_str = _path_to_str(path)
        for pattern, transform in pattern_to_transform.items():
            if not fnmatch.fnmatchcase(path_str, pattern):
                continue
            matches.append((path, transform(sub_module)))
            break

    if matches:
        sorted_matches = sorted(matches, key=lambda item: len(item[0]))
        paths = [path for path, _ in sorted_matches]
        replacements = [replacement for _, replacement in sorted_matches]

        def _collect(root):
            values: list[tp.Any] = []
            for path in paths:
                node = root
                for part in path:
                    if isinstance(part, int):
                        node = node[part]
                    else:
                        node = getattr(node, part)
                values.append(node)
            return values

        module = eqx.tree_at(_collect, module, replacements)

    return module
