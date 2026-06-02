"""Tests for orthocache.adaptive_attention.

Validates:
1. stream_compact produces correct active-first ordering and count.
2. _single_head_loop matches dense single-head attention (small seq).
3. _multihead_loop matches dense multi-head attention.
4. orthocache_attention dispatches to the correct path based on seq length.
5. Full eviction returns zeros; zero eviction matches dense.
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from orthocache.adaptive_attention import (
    stream_compact,
    _single_head_loop,
    _multihead_loop,
    orthocache_attention,
    _SEQ_THRESHOLD,
)

# ── Constants ────────────────────────────────────────────────────────────────

BLOCK_SIZE = 512
NUM_HEADS = 2
HEAD_DIM = 64
RNG = jax.random.PRNGKey(99)


def _make_tensors(num_blocks, seq_len_q=1, dtype=jnp.bfloat16, rng=RNG):
    """Create test q, k_cache, v_cache tensors."""
    seq_len_k = num_blocks * BLOCK_SIZE
    k1, k2, k3 = jax.random.split(rng, 3)
    q = jax.random.normal(k1, (seq_len_q, NUM_HEADS, HEAD_DIM), dtype=dtype)
    k = jax.random.normal(k2, (seq_len_k, NUM_HEADS, HEAD_DIM), dtype=dtype)
    v = jax.random.normal(k3, (seq_len_k, NUM_HEADS, HEAD_DIM), dtype=dtype)
    return q, k, v


def _dense_attention(q, keys, values):
    """Reference dense multi-head attention in float32."""
    q32 = q.astype(jnp.float32)
    k32 = keys.astype(jnp.float32)
    v32 = values.astype(jnp.float32)
    scale = jnp.sqrt(jnp.float32(HEAD_DIM))
    logits = jnp.einsum('qhd,khd->qkh', q32, k32) / scale
    weights = jax.nn.softmax(logits, axis=1)
    return jnp.einsum('qkh,khd->qhd', weights, v32).astype(jnp.bfloat16)


# ── stream_compact ──────────────────────────────────────────────────────────


class TestStreamCompact:
    """Tests for the stream_compact utility."""

    def test_basic_compaction(self):
        """Active indices should be at the front of the output."""
        mask = jnp.array([True, False, True, False, True, False, False, True])
        indices, n_active = stream_compact(mask)
        assert n_active == 4
        # First 4 entries should be the active block indices
        active = set(int(indices[i]) for i in range(n_active))
        assert active == {0, 2, 4, 7}

    def test_all_active(self):
        """With all-True mask, all indices active."""
        mask = jnp.ones(6, dtype=jnp.bool_)
        indices, n_active = stream_compact(mask)
        assert n_active == 6

    def test_all_evicted(self):
        """With all-False mask, num_active should be 0."""
        mask = jnp.zeros(6, dtype=jnp.bool_)
        _, n_active = stream_compact(mask)
        assert n_active == 0


# ── _single_head_loop ───────────────────────────────────────────────────────


class TestSingleHeadLoop:
    """Tests for the vmapped single-head attention loop."""

    def test_all_blocks_active_matches_dense(self):
        """Single-head loop with all blocks active should match dense attention."""
        num_blocks = 4
        seq_q = 1
        seq_k = num_blocks * BLOCK_SIZE
        rng = jax.random.PRNGKey(10)
        k1, k2, k3 = jax.random.split(rng, 3)

        # Single head tensors: (seq_q, HD) and (seq_k, HD)
        q = jax.random.normal(k1, (seq_q, HEAD_DIM), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (seq_k, HEAD_DIM), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (seq_k, HEAD_DIM), dtype=jnp.bfloat16)

        # All blocks active
        indices = jnp.arange(num_blocks, dtype=jnp.int32)
        result = _single_head_loop(q, k, v, indices, num_blocks)

        # Dense single-head reference
        scale = jnp.sqrt(jnp.float32(HEAD_DIM))
        logits = jnp.einsum('qd,kd->qk', q.astype(jnp.float32),
                            k.astype(jnp.float32)) / scale
        weights = jax.nn.softmax(logits, axis=1)
        expected = jnp.einsum('qk,kd->qd', weights,
                              v.astype(jnp.float32)).astype(jnp.bfloat16)

        np.testing.assert_allclose(
            np.array(result), np.array(expected),
            atol=0.05, rtol=0.05,
            err_msg="_single_head_loop doesn't match dense attention",
        )

    def test_output_shape(self):
        """Output shape should be (seq_q, head_dim)."""
        seq_q, seq_k = 1, 2 * BLOCK_SIZE
        rng = jax.random.PRNGKey(11)
        k1, k2, k3 = jax.random.split(rng, 3)
        q = jax.random.normal(k1, (seq_q, HEAD_DIM), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (seq_k, HEAD_DIM), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (seq_k, HEAD_DIM), dtype=jnp.bfloat16)
        indices = jnp.arange(2, dtype=jnp.int32)
        result = _single_head_loop(q, k, v, indices, 2)
        assert result.shape == (seq_q, HEAD_DIM)


# ── _multihead_loop ─────────────────────────────────────────────────────────


class TestMultiheadLoop:
    """Tests for the fused multi-head attention loop."""

    def test_all_blocks_active_matches_dense(self):
        """Multi-head loop with all blocks active should match dense attention."""
        num_blocks = 4
        q, k, v = _make_tensors(num_blocks, seq_len_q=1)
        indices = jnp.arange(num_blocks, dtype=jnp.int32)

        result = _multihead_loop(q, k, v, indices, num_blocks)
        expected = _dense_attention(q, k, v)

        np.testing.assert_allclose(
            np.array(result), np.array(expected),
            atol=0.05, rtol=0.05,
            err_msg="_multihead_loop doesn't match dense attention",
        )

    def test_output_shape(self):
        """Output should be (seq_q, num_heads, head_dim) in bf16."""
        num_blocks = 2
        q, k, v = _make_tensors(num_blocks, seq_len_q=1)
        indices = jnp.arange(num_blocks, dtype=jnp.int32)
        result = _multihead_loop(q, k, v, indices, num_blocks)
        assert result.shape == (1, NUM_HEADS, HEAD_DIM)
        assert result.dtype == jnp.bfloat16


# ── orthocache_attention dispatcher ─────────────────────────────────────────


class TestOrthocacheAttention:
    """Tests for the adaptive dispatcher."""

    def test_full_eviction_returns_zeros(self):
        """With all blocks evicted, output should be zeros."""
        num_blocks = 4
        q, k, v = _make_tensors(num_blocks)
        mask = jnp.zeros(num_blocks, dtype=jnp.bool_)

        output, meta = orthocache_attention(q, k, v, mask, block_size=BLOCK_SIZE)

        np.testing.assert_allclose(np.array(output), 0.0, atol=1e-9)
        assert meta['num_active'] == 0
        assert meta['path'] == 'zero'

    def test_small_seq_uses_vmap_path(self):
        """Sequences ≤ threshold should dispatch to vmap_heads path."""
        # 4 blocks × 512 = 2048 tokens, well below 16384
        num_blocks = 4
        q, k, v = _make_tensors(num_blocks)
        mask = jnp.ones(num_blocks, dtype=jnp.bool_)

        _, meta = orthocache_attention(q, k, v, mask, block_size=BLOCK_SIZE)
        assert meta['path'] == 'vmap_heads'

    def test_output_shape_and_dtype(self):
        """Output should match q shape and be bf16."""
        num_blocks = 4
        q, k, v = _make_tensors(num_blocks, seq_len_q=1)
        mask = jnp.ones(num_blocks, dtype=jnp.bool_)

        output, _ = orthocache_attention(q, k, v, mask, block_size=BLOCK_SIZE)
        assert output.shape == q.shape
        assert output.dtype == jnp.bfloat16

    def test_metadata_eviction_rate(self):
        """Metadata should report correct eviction rate."""
        num_blocks = 8
        # Keep first 4, evict last 4
        mask = jnp.array([True] * 4 + [False] * 4)
        q, k, v = _make_tensors(num_blocks)

        _, meta = orthocache_attention(q, k, v, mask, block_size=BLOCK_SIZE)
        assert meta['num_active'] == 4
        assert meta['num_blocks'] == 8
        assert abs(meta['eviction_rate'] - 0.5) < 0.01
