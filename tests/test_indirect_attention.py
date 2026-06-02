"""Tests for orthocache.indirect_attention.

Validates:
1. _next_bucket bucketing helper (round-up to power-of-2 bucket).
2. indirect_attention top-level API output shape and metadata.
3. Zero-eviction case (all blocks active) matches dense attention.
4. Full-eviction case returns zeros.
5. Partial eviction produces correct metadata fields.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from orthocache.indirect_attention import (
    _next_bucket,
    indirect_attention,
    BUCKETS,
)

# ── Constants ────────────────────────────────────────────────────────────────

BLOCK_SIZE = 512
NUM_HEADS = 2
HEAD_DIM = 64
RNG = jax.random.PRNGKey(7)


def _make_tensors(num_blocks, seq_len_q=4, rng=RNG):
    """Create q, keys, values for testing."""
    seq_len_k = num_blocks * BLOCK_SIZE
    k1, k2, k3 = jax.random.split(rng, 3)
    q = jax.random.normal(k1, (seq_len_q, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    keys = jax.random.normal(k2, (seq_len_k, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    values = jax.random.normal(k3, (seq_len_k, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    return q, keys, values


def _dense_attention(q, keys, values):
    """Reference dense attention for comparison."""
    scale = jnp.sqrt(jnp.float32(HEAD_DIM))
    logits = jnp.einsum('qhd,khd->qkh', q, keys) / scale
    weights = jax.nn.softmax(logits, axis=1)
    return jnp.einsum('qkh,khd->qhd', weights, values)


# ── _next_bucket tests ──────────────────────────────────────────────────────


class TestNextBucket:
    """Tests for the _next_bucket bucketing helper."""

    def test_exact_bucket_values(self):
        """Exact bucket sizes should map to themselves."""
        for b in BUCKETS:
            assert _next_bucket(b) == b

    def test_rounds_up(self):
        """Values between buckets should round up to the next bucket."""
        assert _next_bucket(3) == 4
        assert _next_bucket(5) == 8
        assert _next_bucket(9) == 16
        assert _next_bucket(17) == 32

    def test_one(self):
        """n=1 should return bucket 1."""
        assert _next_bucket(1) == 1

    def test_beyond_max_bucket(self):
        """n larger than the biggest bucket returns n itself."""
        big = BUCKETS[-1] + 100
        assert _next_bucket(big) == big


# ── indirect_attention top-level API ────────────────────────────────────────


class TestIndirectAttention:
    """Tests for indirect_attention (the Pallas-dispatching wrapper).

    NOTE: compile_indirect_attention uses pallas_call which requires TPU.
    indirect_attention calls it internally, so these tests are TPU-only.
    On CPU we test the metadata-only paths (full eviction) and the helper.
    """

    def test_full_eviction_returns_zeros(self):
        """When all blocks are evicted, output should be zeros."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        # All-False mask → all evicted
        block_mask = jnp.zeros((num_blocks, NUM_HEADS), dtype=jnp.bool_)

        output, meta = indirect_attention(q, keys, values, block_mask, BLOCK_SIZE)

        np.testing.assert_allclose(np.array(output), 0.0, atol=1e-9)
        assert meta['num_active'] == 0
        assert meta['eviction_rate'] == 1.0

    def test_full_eviction_shape(self):
        """Output shape should match q even when all blocks are evicted."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks, seq_len_q=8)
        block_mask = jnp.zeros((num_blocks, NUM_HEADS), dtype=jnp.bool_)

        output, meta = indirect_attention(q, keys, values, block_mask, BLOCK_SIZE)
        assert output.shape == (8, NUM_HEADS, HEAD_DIM)

    def test_metadata_keys_present(self):
        """Metadata should contain expected keys even in the zero-eviction path."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        block_mask = jnp.zeros((num_blocks, NUM_HEADS), dtype=jnp.bool_)

        _, meta = indirect_attention(q, keys, values, block_mask, BLOCK_SIZE)
        assert 'num_active' in meta
        assert 'bucket' in meta
        assert 'eviction_rate' in meta

    @pytest.mark.skipif(
        jax.devices()[0].platform != 'tpu',
        reason='Pallas indirect kernel requires TPU',
    )
    def test_zero_eviction_matches_dense(self):
        """With all blocks active, output should match dense attention."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        block_mask = jnp.ones((num_blocks, NUM_HEADS), dtype=jnp.bool_)

        output, meta = indirect_attention(q, keys, values, block_mask, BLOCK_SIZE)
        dense_out = _dense_attention(q, keys, values)

        np.testing.assert_allclose(
            np.array(output), np.array(dense_out),
            atol=1e-2, rtol=1e-2,
            err_msg="Indirect attention at 0% eviction differs from dense",
        )
        assert meta['eviction_rate'] == 0.0

    @pytest.mark.skipif(
        jax.devices()[0].platform != 'tpu',
        reason='Pallas indirect kernel requires TPU',
    )
    def test_partial_eviction_metadata(self):
        """Partial eviction should report correct active counts and bucket."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        # Keep every other block → 4 active
        mask_1d = jnp.array([True, False] * 4)
        block_mask = jnp.broadcast_to(mask_1d[:, None], (num_blocks, NUM_HEADS))

        output, meta = indirect_attention(q, keys, values, block_mask, BLOCK_SIZE)

        assert meta['num_active'] == 4
        assert meta['num_blocks'] == 8
        assert meta['bucket'] == _next_bucket(4)
        assert 0.0 < meta['eviction_rate'] < 1.0
