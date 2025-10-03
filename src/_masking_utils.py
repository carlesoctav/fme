import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ._utils import GeneralInterface


_BlockMask = Array 
_MaskImpl = tp.Callable

def and_masks(mask_fns):
    def mask(b, h, q, kv):
        result = jnp.ones((), dtype = jnp.bool)
        for mask_fn in mask_fns:
            result = mask_fn(b, h, q, kv) & result
        return result

    return mask

def or_masks(mask_fns):
    def mask(b, h, q, kv):

        result = jnp.ones((), dtype = jnp.bool)
        for mask_fn in mask_fns:
            result = mask_fn(b, h, q, kv) | result
        return result

    return mask

def vmap_bhqkv(mask_fn, vmap_bh = True):
    in_axess = [(None, None, None, 0), (None, None, 0, None)]
    if vmap_bh:
        in_axess = in_axess.extend([(None, 0, None, None), (0, None, None, None)])


    fn = mask_fn
    for axes in in_axess:
        fn = jax.vmap(mask_fn, in_axes = axes)

    return fn


def causal_mask_function(b, h, q, kv):
    return q >= kv



def sliding_window_mask_overlay(window_size: int):
    def mask(b, h, q, kv):
        if q-kv >= 0:
            return q-kv <= window_size
        else:
            return kv -q <= window_size
    return mask


def document_mask_overlay(segment_ids: Int[Array]):

    def mask(b, h, q, kv):
        return segment_ids[b, q] == segment_ids[b, kv]
    return mask


def dummy_mask_function(b, h, q, kv):
    return True



def return_bool_mask(
    batch_size,
    q_length ,
    kv_length,
    mask_function,
    attention_mask,
    is_skip_causal,
) -> Bool[Array]:
    batch_arange = jnp.arange(batch_size, dtype = jnp.int) if batch_size else None
    q_arange = jnp.arange(q_length, dtype = jnp.int)
    kv_arange = jnp.arange(q_length, dtype = jnp.int)
    heads_arange = jnp.arange(1, dtype = jnp.int)



    if batch_arange:
        return vmap_bhqkv(mask_function)(batch_arange, heads_arange, q_arange, kv_arange)
    else:
        return vmap_bhqkv(mask_function, vmap_bh = False)(batch_arange, heads_arange, q_arange, kv_arange)



class AttentionMaskInterface(GeneralInterface[str, _MaskImpl]):
    _global_mapping = {
            "eager": return_bool_mask,
            "sdpa": return_bool_mask,
    }

ALL_MASK_ATTENTION_FUNCTIONS = AttentionMaskInterface()

def _preprocess_mask_arguments(
    mask_impl: str,
    input_embeds: Array,
    attention_mask: Array | _BlockMask,
    position_ids: Array,
) -> tuple[bool, Array | _BlockMask | None, int, int]:
    if isinstance(attention_mask, (Array, _BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None

    if mask_impl not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None

    kv_length, kv_offset = input_embeds.shape[1], 0

    packed_sequence_mask = None

    return False, attention_mask, packed_sequence_mask, kv_length, kv_offset

def make_causal_mask(
    mask_impl: str,
    input_embeds: Float[Array, "*B T N H"], 
    attention_mask: Bool[Array] | None = None,
    position_ids: Int[Array] | None = None,
    segment_ids: Int[Array] | None = None,
)-> Bool[Array, "B T N T" ] | _BlockMask:
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

    *B, T, N, H = input_embeds.shape
    B = tuple(B) 

    early_exit, attention_mask, packed_sequence_mask, S, kv_offset = _preprocess_mask_arguments(
        mask_impl, input_embeds, attention_mask, 
    )

    if early_exit:
        return attention_mask 

    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[mask_impl]
    mask_factory_function = causal_mask_function

    if segment_ids: #(batch, seq_len)
        mask_factory_function = and_masks(causal_mask_function, document_mask_overlay(segment_ids))

    causal_mask = mask_interface(
        q_length =  T,
        batch_size= B if B else None,
        nheads = N,
        kv_length=  S,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
    )

    return causal_mask
