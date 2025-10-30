"""Test that the Optimizer fix works with adamw."""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from src.training_utils import Optimizer


class SimpleModel(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, key):
        k1, k2 = jr.split(key)
        self.weight = jr.normal(k1, (10, 10))
        self.bias = jr.normal(k2, (10,))

    def __call__(self, x):
        return jnp.dot(x, self.weight) + self.bias


def main():
    key = jr.PRNGKey(0)

    # Create a simple model
    model = SimpleModel(key)

    # Create optimizer with adamw (requires params argument)
    grad_tx = optax.adamw(learning_rate=1e-3, weight_decay=0.01)
    optimizer = Optimizer(model, grad_tx, wrt=eqx.is_array)

    # Create fake gradients
    grads = jax.tree.map(lambda x: jnp.ones_like(x), eqx.filter(model, eqx.is_array))

    # Test the optimizer call - this should work now
    try:
        new_model, new_optimizer = optimizer(grads, model)
        print("✓ Optimizer update with adamw works!")
        print(f"  Old weight sum: {jnp.sum(model.weight):.4f}")
        print(f"  New weight sum: {jnp.sum(new_model.weight):.4f}")
        return 0
    except Exception as e:
        print(f"✗ Optimizer update failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
