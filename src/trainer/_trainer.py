from abc import ABC
from typing import Any
from dataclasses import dataclass
import equinox as eqx
from optax import GradientTransformation 
from jax import Mesh

@dataclass
class TrainerModule(ABC):
    model: eqx.module | type[eqx.module]
    model_kwargs: dict[str, Any]
    optimizer: GradientTransformation 
    optimizer_kwargs: dict[str, Any]
    mesh: Mesh | None
    parallel_config = "dp"





