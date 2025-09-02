import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Literal


def softmax_cross_entropy_with_integer_labels(
    logits: Array,
    labels: Array,
    where: Array | None = None,
    reduction: None | Literal["mean", "sum"] = None
) -> Array:
    """Per-example softmax cross-entropy for integer labels.
    Shapes:
    - logits: [..., num_classes]
    - labels: [...]
    - where:  [...], boolean mask; False entries are ignored (loss = 0).

    Returns:
    - loss: [...], per-example loss with zeros at masked positions.

    Notes:
    - Numerically stable via max-subtraction.
    - Safe for masked positions: labels at masked positions are ignored
      (loss set to 0) and do not affect indexing.
    """
    if where is not None:
        safe_labels = jnp.where(where, labels, 0)
    else:
        safe_labels = labels

    logits_max = jnp.max(logits, axis=-1, keepdims=True)
    shifted = logits - jax.lax.stop_gradient(logits_max)

    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted), axis=-1))

    true_logit = jnp.take_along_axis(shifted, safe_labels[..., None], axis=-1)[..., 0]

    loss = log_sum_exp - true_logit

    if where is not None:
        loss = jnp.where(where, loss, 0.0)

    if reduction == "mean":
        denom = jnp.sum(where) if where is not None else loss.size
        loss = jnp.sum(loss) / jnp.maximum(denom, 1)

    return loss

