"""Tests for orthocache.alltoallv.

Tests the non-collective helper functions that can run on CPU:
1. stream_compact — active-first index ordering and count.
2. compute_offsets — exclusive prefix sum for write offsets.
3. pack_active_blocks — gather active blocks into static buffer.
4. unpack_received_blocks — extract valid tokens from concatenated shards.

The pmap-dependent functions (count_sync, alltoallv_exchange,
alltoallv_kv_exchange) require multi-device TPU and are TPU-only.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from orthocache.alltoallv import (
    stream_compact,
    compute_offsets,
    pack_active_blocks,
    unpack_received_blocks,
    BLOCK_SIZE,
)

# ── Constants ────────────────────────────────────────────────────────────────

NUM_HEADS = 2
HEAD_DIM = 64
RNG = jax.random.PRNGKey(42)


# ── stream_compact ──────────────────────────────────────────────────────────


class TestStreamCompact:
    """Tests for sort-based stream compaction."""

    def test_basic(self):
        """Active blocks should be sorted to the front."""
        mask = jnp.array([True, False, True, False])
        indices, n_active = stream_compact(mask)
        assert int(n_active) == 2
        active_set = {int(indices[0]), int(indices[1])}
        assert active_set == {0, 2}

    def test_all_active(self):
        """All-True mask → all blocks active."""
        mask = jnp.ones(8, dtype=jnp.bool_)
        _, n_active = stream_compact(mask)
        assert int(n_active) == 8

    def test_all_inactive(self):
        """All-False mask → 0 active."""
        mask = jnp.zeros(8, dtype=jnp.bool_)
        _, n_active = stream_compact(mask)
        assert int(n_active) == 0

    def test_preserves_relative_order(self):
        """Active indices should appear in their original relative order."""
        mask = jnp.array([False, True, False, True, True, False])
        indices, n_active = stream_compact(mask)
        n = int(n_active)
        assert n == 3
        active_list = [int(indices[i]) for i in range(n)]
        assert active_list == [1, 3, 4]


# ── compute_offsets ─────────────────────────────────────────────────────────


class TestComputeOffsets:
    """Tests for exclusive prefix sum offset computation."""

    def test_basic(self):
        """Offsets should be exclusive prefix sums of counts."""
        counts = jnp.array([3, 2, 5, 1], dtype=jnp.int32)
        offsets, total = compute_offsets(counts)
        np.testing.assert_array_equal(offsets, [0, 3, 5, 10])
        assert int(total) == 11

    def test_uniform_counts(self):
        """Uniform counts should produce evenly-spaced offsets."""
        counts = jnp.array([4, 4, 4, 4], dtype=jnp.int32)
        offsets, total = compute_offsets(counts)
        np.testing.assert_array_equal(offsets, [0, 4, 8, 12])
        assert int(total) == 16

    def test_zero_counts(self):
        """Zero counts should produce all-zero offsets with total 0."""
        counts = jnp.zeros(4, dtype=jnp.int32)
        offsets, total = compute_offsets(counts)
        np.testing.assert_array_equal(offsets, [0, 0, 0, 0])
        assert int(total) == 0

    def test_single_device(self):
        """Single device: offset should be 0, total = count."""
        counts = jnp.array([7], dtype=jnp.int32)
        offsets, total = compute_offsets(counts)
        np.testing.assert_array_equal(offsets, [0])
        assert int(total) == 7


# ── pack_active_blocks ──────────────────────────────────────────────────────


class TestPackActiveBlocks:
    """Tests for packing active KV blocks into a static send buffer."""

    def _make_kv(self, num_blocks, rng=RNG):
        """Create a KV shard with distinguishable blocks."""
        total_tokens = num_blocks * BLOCK_SIZE
        k1 = jax.random.split(rng)[0]
        kv = jax.random.normal(
            k1, (total_tokens, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16
        )
        return kv

    def test_all_active_packs_all(self):
        """With all blocks active, packed buffer should contain all data."""
        num_blocks = 4
        kv = self._make_kv(num_blocks)
        indices = jnp.arange(num_blocks, dtype=jnp.int32)
        num_active = jnp.int32(num_blocks)

        packed = pack_active_blocks(kv, indices, num_active, num_blocks)
        assert packed.shape == kv.shape

        # First num_active blocks should have nonzero data
        for i in range(num_blocks):
            block = packed[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            energy = float(jnp.sum(block ** 2))
            assert energy > 0, f"Block {i} is unexpectedly zero"

    def test_partial_active_pads_tail(self):
        """Inactive slots should be zero-padded."""
        num_blocks = 4
        kv = self._make_kv(num_blocks)
        # Only first 2 blocks active
        mask = jnp.array([True, True, False, False])
        indices, num_active = stream_compact(mask)

        packed = pack_active_blocks(kv, indices, num_active, num_blocks)
        assert packed.shape == kv.shape

        na = int(num_active)
        # Inactive blocks (tail) should be zero
        for i in range(na, num_blocks):
            block = packed[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]
            energy = float(jnp.sum(block ** 2))
            assert energy == 0.0, f"Padding block {i} is nonzero"

    def test_output_shape_static(self):
        """Output shape should always be (max_blocks * BLOCK_SIZE, H, D)."""
        num_blocks = 8
        kv = self._make_kv(num_blocks)
        indices = jnp.arange(num_blocks, dtype=jnp.int32)
        num_active = jnp.int32(3)

        packed = pack_active_blocks(kv, indices, num_active, num_blocks)
        expected_shape = (num_blocks * BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        assert packed.shape == expected_shape


# ── unpack_received_blocks ──────────────────────────────────────────────────


class TestUnpackReceivedBlocks:
    """Tests for extracting valid tokens from all-to-all output."""

    def test_uniform_counts(self):
        """With uniform counts, all tokens should be valid."""
        num_devices = 4
        max_blocks = 2
        total_tokens = num_devices * max_blocks * BLOCK_SIZE
        rng = jax.random.PRNGKey(77)
        kv = jax.random.normal(
            rng, (total_tokens, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16
        )
        # All blocks active on every device
        all_counts = jnp.array([2, 2, 2, 2], dtype=jnp.int32)

        active_kv, offsets = unpack_received_blocks(
            kv, all_counts, num_devices, max_blocks
        )

        assert active_kv.shape == kv.shape
        # All tokens should be valid (no padding)
        total_valid = int(jnp.sum(all_counts)) * BLOCK_SIZE
        assert total_valid == total_tokens
        # Offsets should be exclusive prefix sum × BLOCK_SIZE
        np.testing.assert_array_equal(offsets, [0, 2 * BLOCK_SIZE, 4 * BLOCK_SIZE, 6 * BLOCK_SIZE])

    def test_mixed_counts_offsets(self):
        """With varied counts, offsets should reflect the prefix sum."""
        num_devices = 3
        max_blocks = 4
        total_tokens = num_devices * max_blocks * BLOCK_SIZE
        rng = jax.random.PRNGKey(88)
        kv = jax.random.normal(
            rng, (total_tokens, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16
        )
        all_counts = jnp.array([1, 3, 2], dtype=jnp.int32)

        _, offsets = unpack_received_blocks(
            kv, all_counts, num_devices, max_blocks
        )
        np.testing.assert_array_equal(
            offsets,
            [0, 1 * BLOCK_SIZE, 4 * BLOCK_SIZE],
        )
