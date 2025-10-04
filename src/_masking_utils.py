import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ._utils import GeneralInterface


_BlockMask = Array
_MaskImpl = tp.Callable
_MaskFn = tp.Any


def and_masks(mask_fns):
    def mask(b, h, q, kv):
        result = jnp.ones((), dtype=jnp.bool)
        for mask_fn in mask_fns:
            result = mask_fn(b, h, q, kv) & result
        return result

    return mask


def or_masks(mask_fns):
    def mask(b, h, q, kv):
        result = jnp.ones((), dtype=jnp.bool)
        for mask_fn in mask_fns:
            result = mask_fn(b, h, q, kv) | result
        return result

    return mask


def vmap_bhqkv(mask_fn, vmap_bh=True):
    in_axess = [(None, None, None, 0), (None, None, 0, None)]
    if vmap_bh:
        in_axess = in_axess.extend([(None, 0, None, None), (0, None, None, None)])

    fn = mask_fn
    for axes in in_axess:
        fn = jax.vmap(mask_fn, in_axes=axes)

    return fn


def causal_mask_function(b, h, q, kv):
    return q >= kv


def sliding_window_mask_overlay(window_size: int):
    def mask(b, h, q, kv):
        if q - kv >= 0:
            return q - kv <= window_size
        else:
            return kv - q <= window_size

    return mask


def document_mask_overlay(segment_ids: Int[Array, "..."]):
    def mask(b, h, q, kv):
        return segment_ids[b, q] == segment_ids[b, kv]

    return mask


def ignore_padding_overlay(segment_ids: Int[Array, "..."], pad_id: int = 0):
    def mask(b, h, q, kv):
        return segment_ids[b, kv] != pad_id

    return mask


def dummy_mask_function(b, h, q, kv):
    return True


def make_bool_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    mask_function: _MaskFn,
    padding_mask: Bool[Array, "..."],
) -> Bool[Array, "..."]:
    batch_arange = jnp.arange(batch_size, dtype=jnp.int) if batch_size else None
    q_arange = jnp.arange(q_length, dtype=jnp.int)
    kv_arange = jnp.arange(q_length, dtype=jnp.int)
    heads_arange = jnp.arange(1, dtype=jnp.int)

    if padding_mask is not None:
        return (
            vmap_bhqkv(mask_function)(batch_arange, heads_arange, q_arange, kv_arange)
            & padding_mask
        )
    else:
        return vmap_bhqkv(mask_function)(
            batch_arange, heads_arange, q_arange, kv_arange
        )


class AttentionMaskInterface(GeneralInterface[str, _MaskImpl]):
    _global_mapping = {
        "eager": make_bool_mask,
        "sdpa": make_bool_mask,
    }


ALL_MASK_ATTENTION_FUNCTIONS = AttentionMaskInterface()


def makesure_4d_padding_mask(padding_mask: Array, B, T, N) -> Array:
    if padding_mask.shape == (B, T):
        return jnp.broadcast_to(padding_mask[..., None, None], (B, T, N, T))
    elif padding_mask.shape == (B, T, T):
        return jnp.broadcast_to(padding_mask[:, :, None, :], (B, T, N, T))
    elif padding_mask.shape == (B, T, N, T):
        return padding_mask

    raise ValueError(
        f" attention_mask with a shape of (B, T), (B, T, T), or (B, T, N, T), got {padding_mask.ndim} with shape of {padding_mask.shape}"
    )


def make_causal_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T N H"],
    attention_mask: Bool[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
) -> Bool[Array, "B T N T"] | _BlockMask:
    """
    Generates a mask for causal attention.

    Args:
    mask_impl : str
        The type of mask to create. Must be one of the keys in `ALL_MASK_ATTENTION_FUNCTIONS`.
    input_embeds : Float[Array, "*B T N H"]
        Input embeddings to the attention layer, of shape (batch, seq_len, nheads, head_dim).
    attention_mask : Bool[Array] or None, optional
        An attention mask provided by the user. If supplied and has the correct shape, it will be used as is; otherwise, it will be ignored.
    position_ids : Int[Array] or None, optional
        Position IDs of the input embeddings. Used to create segment_ids if segment_ids is not provided.
    segment_ids : Int[Array] or None, optional
        Segment IDs of the input embeddings. If provided, used for document-level masking (sequence packing).

    Returns:
    Bool[Array, "B T N T"] or _BlockMask
        The computed causal attention mask,
    """

    B, T, N, H = input_embeds.shape

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[mask_impl]
    mask_factory_function = causal_mask_function

    padding_mask = None
    if segment_ids is not None:
        mask_factory_function = and_masks(
            mask_factory_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask is not None:
        padding_mask = makesure_4d_padding_mask(attention_mask, B, T, N)

    causal_mask = mask_interface(
        batch_size=B,
        q_length=T,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
    )

    return causal_mask


def make_full_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T N H"],
    attention_mask: Bool[Array, "..."] | None = None,
    segment_ids: Int[Array, "..."] | None = None,
) -> Bool[Array, "B T N T"] | _BlockMask:
    """
    Generates a mask for causal attention.

    Args:
    mask_impl : str
        The type of mask to create. Must be one of the keys in `ALL_MASK_ATTENTION_FUNCTIONS`.
    input_embeds : Float[Array, "B T N H"]
        Input embeddings to the attention layer, of shape (batch, seq_len, nheads, head_dim).
    attention_mask : Bool[Array] or None, optional
        An attention mask provided by the user. If supplied and has the correct shape, it will be used as is; otherwise, it will be ignored.
    segment_ids : Int[Array] or None, optional
        Segment IDs of the input embeddings. If provided, used for document-level masking (sequence packing).

    Returns:
    Bool[Array, "B T N T"] or _BlockMask
        The computed causal attention mask,
    """

    B, T, N, H = input_embeds.shape

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[mask_impl]
    mask_factory_function = dummy_mask_function

    padding_mask = None
    if segment_ids is not None:
        mask_factory_function = and_masks(
            causal_mask_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask is not None:
        padding_mask = makesure_4d_padding_mask(attention_mask, B, T, N)

    full_mask = mask_interface(
        batch_size=B,
        q_length=T,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
    )

    return full_mask


def sliding_window_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T N H"],
    window_size: int,
    attention_mask: Bool[Array, "..."] | None = None,
    segment_ids: Int[Array, "..."] | None = None,
):
    B, T, N, H = input_embeds.shape

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[mask_impl]
    mask_factory_function = sliding_window_mask_overlay(window_size)

    padding_mask = None
    if segment_ids:
        mask_factory_function = and_masks(
            mask_factory_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask:
        padding_mask = makesure_4d_padding_mask(attention_mask)

    full_mask = mask_interface(
        batch_size=B,
        q_length=T,
        nheads=N,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
    )

    return full_mask
