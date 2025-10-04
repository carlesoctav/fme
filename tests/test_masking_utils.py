import jax.numpy as jnp
import pytest
from src._masking_utils import make_causal_mask, make_full_mask


class TestCausalMask:
    """Test causal mask generation with different scenarios."""

    def test_causal_mask_with_padding_mask(self):
        """Test causal mask with simple padding mask (B, T)."""
        B, T, H = 2, 4, 8

        input_embeds = jnp.ones((B, T, H), dtype=jnp.float32)

        attention_mask = jnp.array(
            [
                [1, 1, 1, 0],  
                [1, 1, 0, 0],  
            ],
            dtype=jnp.bool_,
        )

        mask = make_causal_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            segment_ids=None,
        )

        assert mask.shape == (B, T, T), f"Expected shape {(B, T, T)}, got {mask.shape}"

        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    if kv > q:
                        assert not mask[b, q, kv], (
                            f"Future position should be masked at b={b}, q={q}, kv={kv}"
                        )
                    elif kv <= q and attention_mask[b, kv]:
                        assert mask[b, q, kv], (
                            f"Valid past position should be visible at b={b}, q={q}, kv={kv}"
                        )
                    elif not attention_mask[b, kv]:
                        assert not mask[b, q, kv], (
                            f"Padding should be masked at b={b}, q={q}, kv={kv}"
                        )

    def test_causal_mask_with_segment_ids(self):
        """Test causal mask with segment IDs for document boundaries (B, T)."""
        B, T, H = 2, 6, 8

        input_embeds = jnp.ones((B, T, H), dtype=jnp.float32)

        segment_ids = jnp.array(
            [
                [0, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 2, 2],
            ],
            dtype=jnp.int32,
        )

        mask = make_causal_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=None,
            segment_ids=segment_ids,
        )

        assert mask.shape == (B, T, T), f"Expected shape {(B, T, T)}, got {mask.shape}"

        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    same_segment = segment_ids[b, q] == segment_ids[b, kv]
                    is_causal = kv <= q

                    if same_segment and is_causal:
                        assert mask[b, q, kv], (
                            f"Same segment and causal should be visible at b={b}, q={q}, kv={kv}"
                        )
                    else:
                        assert not mask[b, q, kv], (
                            f"Different segment or future should be masked at b={b}, q={q}, kv={kv}"
                        )

    def test_causal_mask_no_masking(self):
        """Test causal mask with no padding or segment IDs."""
        B, T, H = 2, 4, 8

        input_embeds = jnp.ones((B, T, H), dtype=jnp.float32)

        mask = make_causal_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=None,
            segment_ids=None,
        )

        assert mask.shape == (B, T, T), f"Expected shape {(B, T, T)}, got {mask.shape}"

        # Check pure causal masking
        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    if kv <= q:
                        assert mask[b, q, kv], (
                            f"Past/current should be visible at b={b}, q={q}, kv={kv}"
                        )
                    else:
                        assert not mask[b, q, kv], (
                            f"Future should be masked at b={b}, q={q}, kv={kv}"
                        )


class TestFullMask:
    """Test full (bidirectional) mask generation with different scenarios."""

    def test_full_mask_with_padding_mask(self):
        """Test full mask with simple padding mask (B, T)."""
        B, T, H = 2, 4, 8

        input_embeds = jnp.ones((B, T, H), dtype=jnp.float32)

        attention_mask = jnp.array(
            [
                [1, 1, 1, 0],  
                [1, 1, 0, 0], 
            ],
            dtype=jnp.bool_,
        )

        mask = make_full_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            segment_ids=None,
        )

        assert mask.shape == (B, T, T), f"Expected shape {(B, T, T)}, got {mask.shape}"

        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    if attention_mask[b, kv]:
                        assert mask[b, q, kv], (
                            f"Valid position should be visible at b={b}, q={q}, kv={kv}"
                        )
                    else:
                        assert not mask[b, q, kv], (
                            f"Padding should be masked at b={b}, q={q}, kv={kv}"
                        )

    def test_full_mask_with_segment_ids(self):
        """Test full mask with segment IDs for document boundaries (B, T)."""
        B, T, H = 2, 6, 8

        input_embeds = jnp.ones((B, T, H), dtype=jnp.float32)

        segment_ids = jnp.array(
            [
                [0, 0, 0, 1, 1, 1],
                [0, 0, 1, 1, 2, 2],
            ],
            dtype=jnp.int32,
        )

        mask = make_full_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=None,
            segment_ids=segment_ids,
        )

        assert mask.shape == (B, T, T), f"Expected shape {(B, T, T)}, got {mask.shape}"

        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    same_segment = segment_ids[b, q] == segment_ids[b, kv]

                    if same_segment:
                        assert mask[b, q, kv], (
                            f"Same segment should be visible at b={b}, q={q}, kv={kv}"
                        )
                    else:
                        assert not mask[b, q, kv], (
                            f"Different segment should be masked at b={b}, q={q}, kv={kv}"
                        )

    def test_full_mask_no_masking(self):
        """Test full mask with no padding or segment IDs."""
        B, T, H = 2, 4, 8

        input_embeds = jnp.ones((B, T, H), dtype=jnp.float32)

        mask = make_full_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=None,
            segment_ids=None,
        )

        assert mask.shape == (B, T, T), f"Expected shape {(B, T, T)}, got {mask.shape}"

        # Check that all positions can attend to all positions (full attention)
        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    assert mask[b, q, kv], (
                        f"All positions should be visible at b={b}, q={q}, kv={kv}"
                    )
