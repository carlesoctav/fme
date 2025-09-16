import jax
import jax.numpy as jnp


Array = jax.Array


def canonicalize_dtype(*args, dtype: jnp.dtype | None = None, inexact: bool = True) -> jnp.dtype:
    """Infer a common dtype from inputs or use an override.

    - Skips None arguments.
    - If dtype is provided, returns it as a JAX dtype.
    - If inexact is True, ensures the result is a subdtype of jnp.inexact,
      otherwise promotes to at least float32.
    """
    if dtype is not None:
        return jnp.dtype(dtype)

    dtypes = [jnp.asarray(x).dtype for x in args if x is not None]
    if not dtypes:
        inferred = jnp.float32
    else:
        inferred = jnp.result_type(*dtypes)
    if inexact and not jnp.issubdtype(inferred, jnp.inexact):
        inferred = jnp.result_type(inferred, jnp.float32)
    return jnp.dtype(inferred)


def promote_dtype(*args, dtype: jnp.dtype | None = None, inexact: bool = True):
    """Promotes inputs to a specified or inferred dtype and returns cast arrays.

    Returns a tuple of arrays (even for a single input) to allow unpacking.
    """
    target = canonicalize_dtype(*args, dtype=dtype, inexact=inexact)
    arrays = tuple(jnp.asarray(x, target) if x is not None else None for x in args)
    return arrays
