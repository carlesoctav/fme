import typing as tp

import jax
import jax.numpy as jnp
from jaxtyping import Array


_MaskFn = tp.Callable[[int, int, int, int]] #(b, h, q, kv)


def and_masks(*mask_fns: list[_MaskFn]) -> _MaskFn:

    def mask(b, h, q, kv):
        res = jnp.ones((), dtype = jnp.bool)
        for mask_fn in mask_fns:
            res = mask_fn(b, h, q, kv) & res
        return res

    return mask


def or_masks(*mask_fns :list[_MaskFn]) -> _MaskFn:
    def mask(b, h, q, kv):
        res = jnp.ones((), dtype = jnp.bool)
        for mask_fn in mask_fns:
            res = mask_fn(b, h, q, kv) | res
        return res

    return mask


def causal_mask_function(b, h, q, kv)-> bool:
    return kv <= q


def sliding_window_overlay(sliding_window: int) -> _MaskFn:
    def mask(b, h, q, kv):
        return q - kv < sliding_window

    return mask


#segmentsegment_ids_mask,)segment_ids_mask
def segment_ids_mask_function(segment_ids: Array) -> _MaskFn:
    def mask(b, h, q, kv):
        return segment_ids[b, q] == segment_ids[b, kv]

    return mask

def _vmap_for_bhqkv(mask_function: _MaskFn, bh_indices = True):
    dimensions = [(None, None, None, 0), (None, None, 0, None)]
    if bh_indices: 
        dimensions.extend([(None, 0, None, None), (0, None, None, None)])

    for dim in dimensions:
        fn = jax.vmap(mask_function, in_axes=dim)

    return fn



def causal_mask():
    pass
