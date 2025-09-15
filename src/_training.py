from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import Array, PyTree
from optax import GradientTransformationExtraArgs


AxisSpec = bool | Callable[[Any], bool]

class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    wrt: PyTree[AxisSpec]  = eqx.field(static=True) #thinking more about wrt, so we can freeze some layer, maybe by introducing required_grad in the module?
    step: int 
    tx: GradientTransformationExtraArgs = eqx.field(static=True)

    def __init__(self, grad_txc, model, *, wrt=eqx.is_inexact_array):
        self.tx = grad_txc
        self.wrt = wrt
        self.opt_state = self.tx.init(eqx.filter(model, self.wrt))
        self.step = 0

    def __call__(self, grads: PyTree[Array], model: eqx.Module) -> tuple[eqx.Module, "Optimizer"]:
        updates, opt_state = self.tx.update(
            grads, self.opt_state, eqx.filter(model, self.wrt)
        )
        new_model = eqx.apply_updates(model, updates)
        new_self = eqx.tree_at(lambda o: [o.opt_state, o.step], self, [opt_state, self.step+1])
        return new_model, new_self



@jtu.register_dataclass
class Metric:
    values: float 
    counts: int 
    mode: str

    def update(self, **kwargs) -> "Metric":
        if "values" not in kwargs:
            raise ValueError("values must be present with type of int or float")
        if "counts" not in kwargs:
            raise ValueError("counts must be present with type of int")

        add_value = kwargs.get("values") 
        add_count = kwargs.get("counts") 
        assert isinstance(add_value, (float, int)), f"values must be int or float value, got {add_value.__class__}" 
        assert isinstance(add_count, int), f"counts must be int value, got {add_count.__class__}"

        new_metric = eqx.tree_at(lambda m: [m.values, m.counts], self, [self.values + add_value, self.counts + add_count])
        return new_metric
    
    def compute(self) -> float:
        raise NotImplementedError

Metrics = PyTree[Metric]
