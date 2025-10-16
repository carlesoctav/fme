from dataclasses import dataclass, field

import jax
import jax.tree_util as jtu


@jtu.register_dataclass
@dataclass(slots=True)
class ArrayWithSharding:
    value: jax.Array | None
    sharding: str | tuple[str, ...] | None | tuple[tuple[str, ...], ...] = field(
        metadata=dict(static=True), default=None
    )
