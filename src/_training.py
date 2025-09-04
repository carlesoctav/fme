from collections.abc import Callable
from typing import Any

import equinox as eqx
from jaxtyping import Array, PyTree
from optax import GradientTransformationExtraArgs


AxisSpec = bool | Callable[[Any], bool]

class Optimizer(eqx.Module):
    opt_state: PyTree[Array]
    wrt: PyTree[AxisSpec]  = eqx.field(static=True) #thinking more about wrt, so we can freeze some layer, maybe by introducing required_grad in the module?
    step: int 
    tx: GradientTransformationExtraArgs = eqx.field(static=True)

    def __init__(self, optimizer, model, *, wrt=eqx.is_inexact_array):
        self.tx = optimizer
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
