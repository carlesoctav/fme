import jax.numpy as jnp
import pytest
from src._masking_utils import make_causal_mask, make_full_mask


class TestCausalMask:
    """Test causal mask generation with different scenarios."""

    def test_causal_mask_with_padding_mask(self):
        """Test causal mask with simple padding mask (B, T)."""
        B, T, N, H = 2, 4, 1, 8

        input_embeds = jnp.ones((B, T, N, H), dtype=jnp.float32)

        attention_mask = jnp.array(
            [
                [1, 1, 1, 0],  # first sequence: 3 tokens, 1 padding
                [1, 1, 0, 0],  # second sequence: 2 tokens, 2 padding
            ],
            dtype=jnp.bool_,
        )

        mask = make_causal_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            segment_ids=None,
        )

        assert mask.shape == (B, T, N, T), (
            f"Expected shape {(B, T, N, T)}, got {mask.shape}"
        )

        # Check causal property: position i can only attend to positions <= i
        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    if kv > q:
                        # Future positions should be masked out
                        assert not mask[b, q, 0, kv], (
                            f"Future position should be masked at b={b}, q={q}, kv={kv}"
                        )
                    elif kv <= q and attention_mask[b, kv]:
                        # Past/current valid positions should be visible
                        assert mask[b, q, 0, kv], (
                            f"Valid past position should be visible at b={b}, q={q}, kv={kv}"
                        )
                    elif not attention_mask[b, kv]:
                        # Padding positions should be masked
                        assert not mask[b, q, 0, kv], (
                            f"Padding should be masked at b={b}, q={q}, kv={kv}"
                        )

    def test_causal_mask_with_segment_ids(self):
        """Test causal mask with segment IDs for document boundaries (B, T)."""
        B, T, N, H = 2, 6, 1, 8

        input_embeds = jnp.ones((B, T, N, H), dtype=jnp.float32)

        # First batch: two documents [0,0,0] and [1,1,1]
        # Second batch: three documents [0,0], [1,1], [2,2]
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

        assert mask.shape == (B, T, N, T), (
            f"Expected shape {(B, T, N, T)}, got {mask.shape}"
        )

        # Check that tokens only attend within their document and causally
        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    same_segment = segment_ids[b, q] == segment_ids[b, kv]
                    is_causal = kv <= q

                    if same_segment and is_causal:
                        assert mask[b, q, 0, kv], (
                            f"Same segment and causal should be visible at b={b}, q={q}, kv={kv}"
                        )
                    else:
                        assert not mask[b, q, 0, kv], (
                            f"Different segment or future should be masked at b={b}, q={q}, kv={kv}"
                        )

    def test_causal_mask_no_masking(self):
        """Test causal mask with no padding or segment IDs."""
        B, T, N, H = 2, 4, 2, 8

        input_embeds = jnp.ones((B, T, N, H), dtype=jnp.float32)

        mask = make_causal_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=None,
            segment_ids=None,
        )

        assert mask.shape == (B, T, N, T), (
            f"Expected shape {(B, T, N, T)}, got {mask.shape}"
        )

        # Check pure causal masking
        for b in range(B):
            for n in range(N):
                for q in range(T):
                    for kv in range(T):
                        if kv <= q:
                            assert mask[b, q, n, kv], (
                                f"Past/current should be visible at b={b}, n={n}, q={q}, kv={kv}"
                            )
                        else:
                            assert not mask[b, q, n, kv], (
                                f"Future should be masked at b={b}, n={n}, q={q}, kv={kv}"
                            )


class TestFullMask:
    """Test full (bidirectional) mask generation with different scenarios."""

    def test_full_mask_with_padding_mask(self):
        """Test full mask with simple padding mask (B, T)."""
        B, T, N, H = 2, 4, 1, 8

        input_embeds = jnp.ones((B, T, N, H), dtype=jnp.float32)

        attention_mask = jnp.array(
            [
                [1, 1, 1, 0],  # first sequence: 3 tokens, 1 padding
                [1, 1, 0, 0],  # second sequence: 2 tokens, 2 padding
            ],
            dtype=jnp.bool_,
        )

        mask = make_full_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            segment_ids=None,
        )

        assert mask.shape == (B, T, N, T), (
            f"Expected shape {(B, T, N, T)}, got {mask.shape}"
        )

        # Check bidirectional property: all valid positions can attend to each other
        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    if attention_mask[b, kv]:
                        # Valid positions should be visible (bidirectional)
                        assert mask[b, q, 0, kv], (
                            f"Valid position should be visible at b={b}, q={q}, kv={kv}"
                        )
                    else:
                        # Padding positions should be masked
                        assert not mask[b, q, 0, kv], (
                            f"Padding should be masked at b={b}, q={q}, kv={kv}"
                        )

    def test_full_mask_with_segment_ids(self):
        """Test full mask with segment IDs for document boundaries (B, T)."""
        B, T, N, H = 2, 6, 1, 8

        input_embeds = jnp.ones((B, T, N, H), dtype=jnp.float32)

        # First batch: two documents [0,0,0] and [1,1,1]
        # Second batch: three documents [0,0], [1,1], [2,2]
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

        assert mask.shape == (B, T, N, T), (
            f"Expected shape {(B, T, N, T)}, got {mask.shape}"
        )

        # Check that tokens can attend to any token in their document (bidirectional)
        for b in range(B):
            for q in range(T):
                for kv in range(T):
                    same_segment = segment_ids[b, q] == segment_ids[b, kv]

                    if same_segment:
                        assert mask[b, q, 0, kv], (
                            f"Same segment should be visible at b={b}, q={q}, kv={kv}"
                        )
                    else:
                        assert not mask[b, q, 0, kv], (
                            f"Different segment should be masked at b={b}, q={q}, kv={kv}"
                        )

    def test_full_mask_no_masking(self):
        """Test full mask with no padding or segment IDs."""
        B, T, N, H = 2, 4, 2, 8

        input_embeds = jnp.ones((B, T, N, H), dtype=jnp.float32)

        mask = make_full_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=None,
            segment_ids=None,
        )

        assert mask.shape == (B, T, N, T), (
            f"Expected shape {(B, T, N, T)}, got {mask.shape}"
        )

        # Check that all positions can attend to all positions (full attention)
        for b in range(B):
            for n in range(N):
                for q in range(T):
                    for kv in range(T):
                        assert mask[b, q, n, kv], (
                            f"All positions should be visible at b={b}, n={n}, q={q}, kv={kv}"
                        )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_token_sequence(self):
        """Test with single token sequences."""
        B, T, N, H = 1, 1, 1, 8

        input_embeds = jnp.ones((B, T, N, H), dtype=jnp.float32)
        attention_mask = jnp.array([[1]], dtype=jnp.bool_)

        causal_mask = make_causal_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            segment_ids=None,
        )

        assert causal_mask.shape == (B, T, N, T)
        assert causal_mask[0, 0, 0, 0], "Single token should attend to itself"

        full_mask = make_full_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            segment_ids=None,
        )

        assert full_mask.shape == (B, T, N, T)
        assert full_mask[0, 0, 0, 0], "Single token should attend to itself"

    def test_multiple_heads(self):
        """Test with multiple attention heads."""
        B, T, N, H = 2, 4, 4, 8

        input_embeds = jnp.ones((B, T, N, H), dtype=jnp.float32)
        attention_mask = jnp.array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 0],
            ],
            dtype=jnp.bool_,
        )

        mask = make_causal_mask(
            mask_impl="eager",
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            segment_ids=None,
        )

        assert mask.shape == (B, T, N, T)

        # All heads should have the same mask
        for b in range(B):
            for n in range(1, N):
                assert jnp.array_equal(mask[b, :, n, :], mask[b, :, 0, :]), (
                    f"All heads should have same mask for batch {b}"
                )
