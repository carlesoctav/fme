import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

try:
    from .utils import GeneralInterface
except ImportError:
    from src.utils import GeneralInterface


BlockMask = Array
MaskImpl = tp.Callable
MaskFn = tp.Any


def and_masks(*mask_fns):
    def mask(b, h, q, kv):
        result = jnp.ones((), dtype=jnp.bool)
        for mask_fn in mask_fns:
            result = mask_fn(b, h, q, kv) & result
        return result

    return mask


def or_masks(*mask_fns):
    def mask(b, h, q, kv):
        result = jnp.ones((), dtype=jnp.bool)
        for mask_fn in mask_fns:
            result = mask_fn(b, h, q, kv) | result
        return result

    return mask


def vmap_bhqkv(mask_fn, without_head=False):
    if without_head:
        # Wrap mask_fn to ignore head dimension
        def mask_fn_no_head(b, q, kv):
            return mask_fn(b, None, q, kv)

        in_axess = [(None, None, 0), (None, 0, None), (0, None, None)]
        fn = mask_fn_no_head
    else:
        in_axess = [
            (None, None, None, 0),
            (None, None, 0, None),
            (None, 0, None, None),
            (0, None, None, None),
        ]
        fn = mask_fn

    for axes in in_axess:
        fn = jax.vmap(fn, in_axes=axes)

    return fn


def causal_mask_function(b, h, q, kv):
    return q >= kv


def sliding_window_mask_overlay(window_size: int):
    def mask(b, h, q, kv):
        return jnp.where(q - kv >= 0, q - kv <= window_size, kv - q <= window_size)

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
    mask_function: MaskFn,
    padding_mask: Bool[Array, "..."] | None = None,
    nheads: int | None = None,
) -> Bool[Array, "..."]:
    batch_arange = jnp.arange(batch_size, dtype=jnp.int32) if batch_size else None
    q_arange = jnp.arange(q_length, dtype=jnp.int32)
    kv_arange = jnp.arange(kv_length, dtype=jnp.int32)

    if nheads is None:
        heads_arange = None
        mask_output = vmap_bhqkv(mask_function, without_head=True)(
            batch_arange, q_arange, kv_arange
        )
    else:
        heads_arange = jnp.arange(nheads, dtype=jnp.int32)
        mask_output = vmap_bhqkv(mask_function, without_head=False)(
            batch_arange, heads_arange, q_arange, kv_arange
        )

    if padding_mask is not None:
        if mask_output.ndim == 3:
            expanded_padding_mask = padding_mask[:, None, :]
            return jnp.asarray(mask_output & expanded_padding_mask, dtype=jnp.bool)
        else:
            expanded_padding_mask = padding_mask[:, None, None, :]
            return mask_output & expanded_padding_mask
    else:
        return mask_output


class AttentionMaskInterface(GeneralInterface[str, MaskImpl]):
    _global_mapping = {
        "eager": make_bool_mask,
        "sdpa": make_bool_mask,
    }


ALL_MASK_ATTENTION_FUNCTIONS = AttentionMaskInterface()


def make_causal_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T H"],
    attention_mask: Bool[Array, "B T"] | None = None,
    segment_ids: Int[Array, "B T"] | None = None,
) -> Bool[Array, "B T T"] | BlockMask:
    """
    Generates a mask for causal attention.

    Args:
    mask_impl : str
        The type of mask to create. Must be one of the keys in `ALL_MASK_ATTENTION_FUNCTIONS`.
    input_embeds : Float[Array, "B T H"]
        Input embeddings to the attention layer, of shape (batch, seq_len, head_dim).
    attention_mask : Bool[Array] or None, optional
        An attention mask provided by the user. If supplied and has the correct shape, it will be used as is; otherwise, it will be ignored.
    segment_ids : Int[Array] or None, optional
        Segment IDs of the input embeddings. If provided, used for document-level masking (sequence packing).

    Returns:
    Bool[Array, "B T T"] or _BlockMask
        The computed causal attention mask,
    """

    B, T, H = input_embeds.shape

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[mask_impl]
    mask_factory_function = causal_mask_function

    padding_mask = None
    if segment_ids is not None:
        mask_factory_function = and_masks(
            mask_factory_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask is not None:
        padding_mask = attention_mask

    causal_mask = mask_interface(
        batch_size=B,
        q_length=T,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
        nheads=None,
    )

    return causal_mask


def make_full_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T H"],
    attention_mask: Bool[Array, "..."] | None = None,
    segment_ids: Int[Array, "..."] | None = None,
) -> Bool[Array, "B T T"] | BlockMask:
    """
    Generates a mask for full attention.

    Args:
    mask_impl : str
        The type of mask to create. Must be one of the keys in `ALL_MASK_ATTENTION_FUNCTIONS`.
    input_embeds : Float[Array, "B T H"]
        Input embeddings to the attention layer, of shape (batch, seq_len, head_dim).
    attention_mask : Bool[Array] or None, optional
        An attention mask provided by the user. If supplied and has the correct shape, it will be used as is; otherwise, it will be ignored.
    segment_ids : Int[Array] or None, optional
        Segment IDs of the input embeddings. If provided, used for document-level masking (sequence packing).

    Returns:
    Bool[Array, "B T T"] or _BlockMask
        The computed full attention mask,
    """

    B, T, H = input_embeds.shape

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[mask_impl]
    mask_factory_function = dummy_mask_function

    padding_mask = None
    if segment_ids is not None:
        mask_factory_function = and_masks(
            dummy_mask_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask is not None:
        padding_mask = attention_mask

    full_mask = mask_interface(
        batch_size=B,
        q_length=T,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
        nheads=None,
    )

    return full_mask


def slliding_window_full_mask(
    mask_impl: str,
    input_embeds: Float[Array, "B T H"],
    window_size: int,
    attention_mask: Bool[Array, "..."] | None = None,
    segment_ids: Int[Array, "..."] | None = None,
) -> Bool[Array, "B T T"] | BlockMask:
    """
    Generates a mask for sliding window attention.

    Args:
    mask_impl : str
        The type of mask to create. Must be one of the keys in `ALL_MASK_ATTENTION_FUNCTIONS`.
    input_embeds : Float[Array, "B T H"]
        Input embeddings to the attention layer, of shape (batch, seq_len, head_dim).
    window_size : int
        Size of the sliding window.
    attention_mask : Bool[Array] or None, optional
        An attention mask provided by the user. If supplied and has the correct shape, it will be used as is; otherwise, it will be ignored.
    segment_ids : Int[Array] or None, optional
        Segment IDs of the input embeddings. If provided, used for document-level masking (sequence packing).

    Returns:
    Bool[Array, "B T T"] or _BlockMask
        The computed sliding window attention mask,
    """
    B, T, H = input_embeds.shape

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[mask_impl]
    mask_factory_function = and_masks(
        sliding_window_mask_overlay(window_size), dummy_mask_function
    )

    padding_mask = None
    if segment_ids is not None:
        mask_factory_function = and_masks(
            mask_factory_function, document_mask_overlay(segment_ids)
        )
    elif attention_mask is not None:
        padding_mask = attention_mask

    sliding_mask = mask_interface(
        batch_size=B,
        q_length=T,
        kv_length=T,
        mask_function=mask_factory_function,
        padding_mask=padding_mask,
        nheads=None,
    )

    return sliding_mask
