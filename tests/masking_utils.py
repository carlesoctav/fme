import jax.numpy as jnp
import pytest

from src._masking_utils import make_causal_mask


@pytest.fixture
def base_inputs():
    batch = 1
    seq_len = 4
    num_heads = 1
    head_dim = 2
    input_embeds = jnp.zeros((batch, seq_len, num_heads, head_dim))
    position_ids = jnp.arange(seq_len)[None, :]
    return input_embeds, position_ids


def test_make_causal_mask_without_segments(base_inputs):
    input_embeds, position_ids = base_inputs
    mask = make_causal_mask(
        mask_impl="sdpa",
        input_embeds=input_embeds,
        attention_mask=None,
        segment_ids=None,
    )

    expected = jnp.tril(jnp.ones((input_embeds.shape[1], input_embeds.shape[1]), dtype=bool))
    expected = expected[None, :, None, :]

    assert jnp.array_equal(mask, expected)


def test_make_causal_mask_with_segment_ids(base_inputs):
    input_embeds, position_ids = base_inputs
    segment_ids = jnp.array([[0, 0, 1, 1]])

    mask = make_causal_mask(
        mask_impl="sdpa",
        input_embeds=input_embeds,
        position_ids=position_ids,
        attention_mask=None,
        segment_ids=segment_ids,
    )

    causal = jnp.tril(jnp.ones((input_embeds.shape[1], input_embeds.shape[1]), dtype=bool))
    causal = causal[None, :, None, :]
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    same_segment = same_segment[:, :, None, :]
    expected = causal & same_segment

    assert jnp.array_equal(mask, expected)
