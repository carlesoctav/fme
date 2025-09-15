import jax
import jax.numpy as jnp

@jax.jit
def update_metrics(metrics, new_metric):
    # metrics: jnp.array, new_metric: scalar
    print("yah, recmpli")
    return jnp.append(metrics, new_metric)

m1 = jnp.array([1.0, 2.0])
m2 = update_metrics(m1, 3.0)  # Output shape (3,) -- compiles
m3 = update_metrics(m2, 4.0)  # Output shape (4,) -- recompiles
