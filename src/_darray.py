from dataclasses import dataclass, field

import jax
import jax.tree_util as jtu

@jtu.register_dataclass
@dataclass(slots=True)
class DArray:
    value: jax.Array | None
    pspec: str | tuple[str, ...] | None | tuple[tuple[str, ...], ...] = field(metadata=dict(static=True), default=None)
    
