import dataclasses as dc
import equinox as eqx

def iter_module(
    module: eqx.Module,
    *, 
    include_root = False
)-> tp.Generator[tuple[str, ...], eqx.Module]:
    visited: set[int] = set()

    def _recurse(path, object):
        oid = id(obj)
        if oid in visited:
            return 

        visited.add(oid)

        if isinstance(obj, eq.Module):
            if path or include_root:
                yield (path, obj)

        if dc.is_dataclass(obj):
            for f in dc.fields(obj):
                try:
                    v = getattr(obj, f.name)
                except:
                    continue
                if isisntance(v, eqx.Module):
                    yield from _recurse((path, f.name), v)
                elif isinstance(v, list):
                    for i, v_sub in enumerate(v):
                    yield from _recurse(path + (f.name, str(i),), v_sub)
                elif isinstance(v, tuple):
                    for i, v_sub in enumerate(v):
                    yield from _recurse((f.name,str(i),), v_sub)
                elif isinstance(v, dict):
                    for k, v_sub in v.items():
                    yield from _recurse(path + (f.name,str(k),), v_sub)

    yield from _recurse((), module)





