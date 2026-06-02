"""Tests for orthocache.pipeline (orthocache_forward).

Validates:
1. _dense_attention produces correct standard attention output.
2. orthocache_forward 'dense' mode returns correct shape and metadata.
3. Invalid mode raises ValueError.
4. 'sparse' and 'compact' modes return valid output and metadata.
5. Metadata contains expected timing and statistics fields.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from orthocache.pipeline import orthocache_forward, _dense_attention

# ── Constants ────────────────────────────────────────────────────────────────

BLOCK_SIZE = 512
NUM_HEADS = 4
HEAD_DIM = 64
RNG = jax.random.PRNGKey(42)


def _make_tensors(num_blocks, seq_len_q=4, rng=RNG):
    """Create q, keys, values for testing."""
    seq_len_k = num_blocks * BLOCK_SIZE
    k1, k2, k3 = jax.random.split(rng, 3)
    q = jax.random.normal(k1, (seq_len_q, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    keys = jax.random.normal(k2, (seq_len_k, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    values = jax.random.normal(k3, (seq_len_k, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    return q, keys, values


# ── _dense_attention ────────────────────────────────────────────────────────


class TestDenseAttention:
    """Tests for the internal _dense_attention reference function."""

    def test_matches_manual_einsum(self):
        """_dense_attention should match a manual einsum-based computation."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        result = _dense_attention(q, keys, values, HEAD_DIM)

        # Manual reference
        scale = jnp.sqrt(jnp.float32(HEAD_DIM))
        logits = jnp.einsum('qhd,khd->qkh', q, keys) / scale
        weights = jax.nn.softmax(logits, axis=1)
        expected = jnp.einsum('qkh,khd->qhd', weights, values)

        np.testing.assert_allclose(
            np.array(result), np.array(expected),
            atol=1e-5, rtol=1e-5,
        )

    def test_output_shape(self):
        """Output should be (seq_len_q, num_heads, head_dim)."""
        num_blocks = 2
        q, keys, values = _make_tensors(num_blocks, seq_len_q=8)
        result = _dense_attention(q, keys, values, HEAD_DIM)
        assert result.shape == (8, NUM_HEADS, HEAD_DIM)

    def test_single_token_query(self):
        """Single-token query should work correctly."""
        num_blocks = 2
        q, keys, values = _make_tensors(num_blocks, seq_len_q=1)
        result = _dense_attention(q, keys, values, HEAD_DIM)
        assert result.shape == (1, NUM_HEADS, HEAD_DIM)
        assert not np.any(np.isnan(np.array(result)))


# ── orthocache_forward: dense mode ──────────────────────────────────────────


class TestForwardDenseMode:
    """Tests for orthocache_forward with mode='dense'."""

    def test_output_shape(self):
        """Dense mode output should match query shape."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        output, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE, mode='dense',
        )

        assert output.shape == (q.shape[0], NUM_HEADS, HEAD_DIM)

    def test_metadata_fields(self):
        """Dense mode metadata should have expected keys."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        _, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE, mode='dense',
        )

        assert meta['mode'] == 'dense'
        assert meta['eviction_rate'] == 0.0
        assert 'latency_ms' in meta
        assert meta['num_blocks'] == num_blocks
        assert meta['num_heads'] == NUM_HEADS
        assert meta['head_dim'] == HEAD_DIM

    def test_matches_dense_attention(self):
        """Dense mode should produce identical output to _dense_attention."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        output, _ = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE, mode='dense',
        )
        expected = _dense_attention(q, keys, values, HEAD_DIM)

        np.testing.assert_allclose(
            np.array(output), np.array(expected),
            atol=1e-5, rtol=1e-5,
        )


# ── orthocache_forward: invalid mode ────────────────────────────────────────


class TestForwardInvalidMode:
    """Tests for error handling on bad mode parameter."""

    def test_invalid_mode_raises_valueerror(self):
        """An unrecognized mode string should raise ValueError."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        with pytest.raises(ValueError, match="Unknown mode"):
            orthocache_forward(
                q, keys, values, block_size=BLOCK_SIZE, mode='invalid',
            )


# ── orthocache_forward: sparse mode ─────────────────────────────────────────


class TestForwardSparseMode:
    """Tests for orthocache_forward with mode='sparse'."""

    def test_output_shape(self):
        """Sparse mode output should match query shape."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        output, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            zeta_max=100.0,  # Very high → minimal eviction
            mode='sparse',
        )

        assert output.shape == (q.shape[0], NUM_HEADS, HEAD_DIM)
        assert meta['mode'] == 'sparse'

    def test_spectral_metadata_present(self):
        """Sparse mode should include spectral analysis metadata."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        _, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            zeta_max=100.0,
            mode='sparse',
        )

        assert 'zeta_mean' in meta
        assert 'zeta_std' in meta
        assert 'eviction_rate' in meta
        assert 'spectral_ms' in meta
        assert 'tau' in meta


# ── orthocache_forward: compact mode ────────────────────────────────────────


class TestForwardCompactMode:
    """Tests for orthocache_forward with mode='compact'."""

    def test_output_shape(self):
        """Compact mode output should match query shape."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        output, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            zeta_max=100.0,  # High threshold = low eviction
            mode='compact',
        )

        assert output.shape == (q.shape[0], NUM_HEADS, HEAD_DIM)
        assert meta['mode'] == 'compact'
        assert 'compact_num_active' in meta

    def test_timing_metadata(self):
        """Compact mode should include timing fields."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)

        _, meta = orthocache_forward(
            q, keys, values, block_size=BLOCK_SIZE,
            zeta_max=100.0,
            mode='compact',
        )

        assert 'total_ms' in meta
        assert 'attention_ms' in meta
        assert meta['total_ms'] >= 0
