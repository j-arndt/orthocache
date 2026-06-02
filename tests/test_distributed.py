"""Tests for orthocache.distributed_attention.

Tests the CPU-testable core functions individually:
1. stream_compact — active-first ordering and count.
2. _pack_active_blocks — gather active blocks into a static buffer.
3. _online_softmax_indirect_attention — numerically stable attention.
4. ici_data_volume — ICI transfer volume accounting.

distributed_orthocache_attention requires pmap and is TPU-only.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from orthocache.distributed_attention import (
    stream_compact,
    _pack_active_blocks,
    _online_softmax_indirect_attention,
    ici_data_volume,
    BLOCK_SIZE,
)

# ── Constants ────────────────────────────────────────────────────────────────

NUM_HEADS = 2
HEAD_DIM = 64
RNG = jax.random.PRNGKey(55)


def _make_kv(num_blocks, rng=RNG):
    """Create a KV shard of shape (num_blocks * BLOCK_SIZE, NUM_HEADS, HEAD_DIM)."""
    total = num_blocks * BLOCK_SIZE
    k1 = jax.random.split(rng)[0]
    return jax.random.normal(k1, (total, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16)


def _dense_attention(q, keys, values):
    """Reference dense multi-head attention in float32 → bf16."""
    q32 = q.astype(jnp.float32)
    k32 = keys.astype(jnp.float32)
    v32 = values.astype(jnp.float32)
    scale = jnp.sqrt(jnp.float32(HEAD_DIM))
    logits = jnp.einsum('qhd,khd->qkh', q32, k32) / scale
    weights = jax.nn.softmax(logits, axis=1)
    return jnp.einsum('qkh,khd->qhd', weights, v32).astype(jnp.bfloat16)


# ── stream_compact ──────────────────────────────────────────────────────────


class TestStreamCompact:
    """Tests for distributed_attention.stream_compact."""

    def test_basic(self):
        """Active blocks should appear first in the index array."""
        mask = jnp.array([True, False, True, True, False])
        indices, n_active = stream_compact(mask)
        n = int(n_active)
        assert n == 3
        active_list = [int(indices[i]) for i in range(n)]
        assert active_list == [0, 2, 3]

    def test_empty_mask(self):
        """All-False mask produces n_active=0."""
        mask = jnp.zeros(4, dtype=jnp.bool_)
        _, n_active = stream_compact(mask)
        assert int(n_active) == 0


# ── _pack_active_blocks ─────────────────────────────────────────────────────


class TestPackActiveBlocks:
    """Tests for packing active KV blocks into a static buffer."""

    def test_preserves_active_data(self):
        """Packed buffer should contain the correct blocks for active indices."""
        num_blocks = 4
        kv = _make_kv(num_blocks)
        # Blocks 0, 2 active
        mask = jnp.array([True, False, True, False])
        indices, n_active = stream_compact(mask)

        packed = _pack_active_blocks(kv, indices, n_active, num_blocks)

        # Block 0 in packed should equal original block at indices[0]
        na = int(n_active)
        kv_blocked = kv.reshape(num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        for i in range(na):
            orig_idx = int(indices[i])
            np.testing.assert_array_equal(
                np.array(packed[i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE]),
                np.array(kv_blocked[orig_idx]),
            )

    def test_output_shape_static(self):
        """Output shape must always be (max_blocks * BLOCK_SIZE, H, D)."""
        num_blocks = 6
        kv = _make_kv(num_blocks)
        indices = jnp.arange(num_blocks, dtype=jnp.int32)
        n_active = jnp.int32(2)

        packed = _pack_active_blocks(kv, indices, n_active, num_blocks)
        assert packed.shape == (num_blocks * BLOCK_SIZE, NUM_HEADS, HEAD_DIM)


# ── _online_softmax_indirect_attention ──────────────────────────────────────


class TestOnlineSoftmaxIndirectAttention:
    """Tests for the online softmax attention over indexed blocks."""

    def test_matches_dense_all_active(self):
        """With all blocks active (identity index), should match dense attention."""
        num_blocks = 4
        seq_q = 1
        rng = jax.random.PRNGKey(123)
        k1, k2, k3 = jax.random.split(rng, 3)
        q = jax.random.normal(k1, (seq_q, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (num_blocks * BLOCK_SIZE, NUM_HEADS, HEAD_DIM),
                              dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (num_blocks * BLOCK_SIZE, NUM_HEADS, HEAD_DIM),
                              dtype=jnp.bfloat16)

        indices = jnp.arange(num_blocks, dtype=jnp.int32)
        result = _online_softmax_indirect_attention(q, k, v, indices, num_blocks)
        expected = _dense_attention(q, k, v)

        np.testing.assert_allclose(
            np.array(result), np.array(expected),
            atol=0.05, rtol=0.05,
            err_msg="Online softmax attention doesn't match dense reference",
        )

    def test_output_shape_and_dtype(self):
        """Output should be (seq_q, num_heads, head_dim) in bf16."""
        num_blocks = 2
        seq_q = 1
        rng = jax.random.PRNGKey(456)
        k1, k2, k3 = jax.random.split(rng, 3)
        q = jax.random.normal(k1, (seq_q, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (num_blocks * BLOCK_SIZE, NUM_HEADS, HEAD_DIM),
                              dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (num_blocks * BLOCK_SIZE, NUM_HEADS, HEAD_DIM),
                              dtype=jnp.bfloat16)

        indices = jnp.arange(num_blocks, dtype=jnp.int32)
        result = _online_softmax_indirect_attention(q, k, v, indices, num_blocks)
        assert result.shape == (seq_q, NUM_HEADS, HEAD_DIM)
        assert result.dtype == jnp.bfloat16

    def test_subset_blocks(self):
        """Attending over a subset of blocks should produce valid (non-NaN) output."""
        num_blocks = 8
        seq_q = 1
        rng = jax.random.PRNGKey(789)
        k1, k2, k3 = jax.random.split(rng, 3)
        q = jax.random.normal(k1, (seq_q, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (num_blocks * BLOCK_SIZE, NUM_HEADS, HEAD_DIM),
                              dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (num_blocks * BLOCK_SIZE, NUM_HEADS, HEAD_DIM),
                              dtype=jnp.bfloat16)

        # Only attend to blocks 0, 3, 5
        indices = jnp.array([0, 3, 5, 0, 0, 0, 0, 0], dtype=jnp.int32)
        result = _online_softmax_indirect_attention(q, k, v, indices, 3)
        assert not np.any(np.isnan(np.array(result)))


# ── ici_data_volume ─────────────────────────────────────────────────────────


class TestICIDataVolume:
    """Tests for the ICI data transfer volume accounting function."""

    def test_zero_sparsity(self):
        """With no eviction, dense == sparse and savings == 0."""
        result = ici_data_volume(
            num_blocks_per_device=8,
            num_devices=4,
            active_counts=[8, 8, 8, 8],
            num_heads=2,
            head_dim=64,
        )
        assert result['dense_bytes'] == result['sparse_bytes']
        assert result['savings_bytes'] == 0
        assert result['savings_pct'] == 0.0

    def test_half_eviction(self):
        """50% eviction should save 50% of bytes."""
        result = ici_data_volume(
            num_blocks_per_device=8,
            num_devices=4,
            active_counts=[4, 4, 4, 4],
            num_heads=2,
            head_dim=64,
        )
        assert result['savings_pct'] == pytest.approx(50.0)
        assert result['sparse_bytes'] == result['dense_bytes'] // 2

    def test_savings_proportional(self):
        """Savings should scale linearly with eviction fraction."""
        result = ici_data_volume(
            num_blocks_per_device=10,
            num_devices=2,
            active_counts=[3, 7],
            num_heads=4,
            head_dim=128,
        )
        dense = result['dense_bytes']
        sparse = result['sparse_bytes']
        expected_sparse = (3 + 7) * BLOCK_SIZE * 4 * 128 * 2  # sum(active) × bytes_per_block
        assert sparse == expected_sparse
        assert result['savings_bytes'] == dense - sparse

    @pytest.mark.skipif(
        jax.devices()[0].platform != 'tpu',
        reason='distributed_orthocache_attention requires TPU pmap',
    )
    def test_distributed_orthocache_attention_placeholder(self):
        """Placeholder for distributed attention test on TPU."""
        pass
