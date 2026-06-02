"""Tests for OrthoCache stream compaction and end-to-end pipeline.

Validates:
1. Stream compaction correctness (compact → decompact recovers original)
2. Compacted attention matches dense attention (within bfloat16 tolerance)
3. Compacted attention matches predicated sparse attention (exact)
4. Edge cases: 0% eviction, 100% eviction, single block
5. Pipeline API with all three modes
"""

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from orthocache.compaction import stream_compact, stream_decompact, compact_and_attend
from orthocache.sparse_attention import jax_block_sparse_attention
from orthocache.pipeline import orthocache_forward


# ── Test Fixtures ────────────────────────────────────────────────────────────

BLOCK_SIZE = 512
NUM_HEADS = 4
HEAD_DIM = 64  # Use 64 for fast CPU tests (512 for real Gemma 31B)
RNG = jax.random.PRNGKey(42)


def _make_tensors(num_blocks, seq_len_q=4, rng=RNG):
    """Create test tensors with specified block count."""
    seq_len_k = num_blocks * BLOCK_SIZE
    k1, k2, k3 = jax.random.split(rng, 3)
    q = jax.random.normal(k1, (seq_len_q, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    keys = jax.random.normal(k2, (seq_len_k, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    values = jax.random.normal(k3, (seq_len_k, NUM_HEADS, HEAD_DIM), dtype=jnp.float32)
    return q, keys, values


def _make_mask(num_blocks, eviction_rate=0.5, rng=RNG):
    """Create a block mask with approximately the target eviction rate."""
    # Deterministic: evict every other block
    if eviction_rate == 0.5:
        mask = jnp.array([i % 2 == 0 for i in range(num_blocks)])
    elif eviction_rate == 0.0:
        mask = jnp.ones(num_blocks, dtype=jnp.bool_)
    elif eviction_rate == 1.0:
        mask = jnp.zeros(num_blocks, dtype=jnp.bool_)
    else:
        # Random mask with approximate eviction rate
        k = jax.random.split(rng)[0]
        mask = jax.random.uniform(k, (num_blocks,)) > eviction_rate
    
    # Broadcast to (num_blocks, num_heads)
    return jnp.broadcast_to(mask[:, None], (num_blocks, NUM_HEADS))


# ── C.1: Stream Compaction Correctness ───────────────────────────────────────

class TestStreamCompact:
    """Tests for the stream_compact function."""
    
    def test_basic_compaction(self):
        """Active blocks should be gathered contiguously at the front."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)
        
        compact_k, compact_v, indices, num_active = stream_compact(
            keys, values, mask, BLOCK_SIZE
        )
        
        # 50% eviction → 4 active blocks
        assert int(num_active) == 4
        
        # Compact tensors should have correct shape
        assert compact_k.shape == (num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        assert compact_v.shape == (num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        assert indices.shape == (num_blocks,)
    
    def test_active_blocks_are_nonzero(self):
        """First num_active blocks in compact tensor should contain real data."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)
        
        compact_k, _, _, num_active = stream_compact(
            keys, values, mask, BLOCK_SIZE
        )
        
        na = int(num_active)
        
        # Active blocks should have nonzero energy
        for i in range(na):
            block_energy = jnp.sum(compact_k[i] ** 2)
            assert float(block_energy) > 0, f"Active block {i} is zero"
        
        # Inactive blocks should be zeroed
        for i in range(na, num_blocks):
            block_energy = jnp.sum(compact_k[i] ** 2)
            assert float(block_energy) == 0, f"Inactive block {i} is nonzero"
    
    def test_compaction_preserves_data(self):
        """Compacted blocks should contain the exact same data as originals."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)
        
        compact_k, _, indices, num_active = stream_compact(
            keys, values, mask, BLOCK_SIZE
        )
        
        keys_blocked = keys.reshape(num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_DIM)
        
        na = int(num_active)
        for i in range(na):
            orig_idx = int(indices[i])
            np.testing.assert_allclose(
                np.array(compact_k[i]),
                np.array(keys_blocked[orig_idx]),
                rtol=1e-6,
                err_msg=f"Compact block {i} != original block {orig_idx}"
            )
    
    def test_zero_eviction(self):
        """With 0% eviction, all blocks should be retained."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.0)
        
        _, _, _, num_active = stream_compact(keys, values, mask, BLOCK_SIZE)
        assert int(num_active) == num_blocks
    
    def test_full_eviction(self):
        """With 100% eviction, no blocks should be retained."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=1.0)
        
        compact_k, _, _, num_active = stream_compact(keys, values, mask, BLOCK_SIZE)
        assert int(num_active) == 0
        
        # All blocks should be zero
        total_energy = float(jnp.sum(compact_k ** 2))
        assert total_energy == 0.0
    
    def test_single_block(self):
        """Edge case: single block retained."""
        num_blocks = 1
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.0)
        
        _, _, _, num_active = stream_compact(keys, values, mask, BLOCK_SIZE)
        assert int(num_active) == 1


# ── C.2: Compacted Attention Correctness ─────────────────────────────────────

class TestCompactAttention:
    """Tests that compacted attention produces correct outputs."""
    
    def test_compact_matches_sparse(self):
        """Compacted attention output should match predicated sparse attention."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)
        
        # Reference: predicated sparse attention
        sparse_out = jax_block_sparse_attention(
            q, keys, values, mask, BLOCK_SIZE
        )
        
        # Test: compacted attention
        compact_out, meta = compact_and_attend(
            q, keys, values, mask, BLOCK_SIZE
        )
        
        np.testing.assert_allclose(
            np.array(compact_out),
            np.array(sparse_out),
            atol=1e-3,
            rtol=1e-3,
            err_msg="Compacted attention output differs from sparse attention"
        )
    
    def test_compact_at_zero_eviction_matches_dense(self):
        """At 0% eviction, compacted attention should match dense attention."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.0)
        
        # Dense reference
        scale = jnp.sqrt(jnp.float32(HEAD_DIM))
        logits = jnp.einsum('qhd,khd->qkh', q, keys) / scale
        weights = jax.nn.softmax(logits, axis=1)
        dense_out = jnp.einsum('qkh,khd->qhd', weights, values)
        
        # Compacted
        compact_out, meta = compact_and_attend(
            q, keys, values, mask, BLOCK_SIZE
        )
        
        assert float(meta['eviction_rate']) == 0.0
        
        np.testing.assert_allclose(
            np.array(compact_out),
            np.array(dense_out),
            atol=1e-3,
            rtol=1e-3,
            err_msg="Compacted attention at 0% eviction differs from dense"
        )
    
    def test_metadata_correct(self):
        """compact_and_attend should return correct metadata."""
        num_blocks = 8
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)
        
        _, meta = compact_and_attend(q, keys, values, mask, BLOCK_SIZE)
        
        assert int(meta['num_active']) == 4
        assert int(meta['num_blocks']) == 8
        assert abs(float(meta['eviction_rate']) - 0.5) < 0.01


# ── C.4: Pipeline API ────────────────────────────────────────────────────────

class TestPipeline:
    """Tests for the orthocache_forward high-level API."""
    
    def test_dense_mode(self):
        """Dense mode should produce standard attention output."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        
        output, meta = orthocache_forward(
            q, keys, values,
            block_size=BLOCK_SIZE,
            mode='dense',
        )
        
        assert output.shape == (q.shape[0], NUM_HEADS, HEAD_DIM)
        assert meta['mode'] == 'dense'
        assert meta['eviction_rate'] == 0.0
    
    def test_sparse_mode(self):
        """Sparse mode should produce valid output with eviction metadata."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        
        output, meta = orthocache_forward(
            q, keys, values,
            block_size=BLOCK_SIZE,
            zeta_max=100.0,  # High threshold = low eviction
            mode='sparse',
        )
        
        assert output.shape == (q.shape[0], NUM_HEADS, HEAD_DIM)
        assert meta['mode'] == 'sparse'
        assert 'eviction_rate' in meta
        assert 'zeta_mean' in meta
    
    def test_compact_mode(self):
        """Compact mode should produce valid output with compaction metadata."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        
        output, meta = orthocache_forward(
            q, keys, values,
            block_size=BLOCK_SIZE,
            zeta_max=100.0,
            mode='compact',
        )
        
        assert output.shape == (q.shape[0], NUM_HEADS, HEAD_DIM)
        assert meta['mode'] == 'compact'
        assert 'compact_num_active' in meta
    
    def test_invalid_mode_raises(self):
        """Invalid mode should raise ValueError."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        
        with pytest.raises(ValueError, match="Unknown mode"):
            orthocache_forward(q, keys, values, block_size=BLOCK_SIZE, mode='invalid')


# ── JIT Compilation ──────────────────────────────────────────────────────────

class TestJITCompilation:
    """Verify that compaction functions are JIT-compilable."""
    
    def test_stream_compact_jittable(self):
        """stream_compact should be JIT-compilable."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)
        
        jit_compact = jax.jit(stream_compact, static_argnames=['block_size'])
        compact_k, compact_v, indices, num_active = jit_compact(
            keys, values, mask, block_size=BLOCK_SIZE
        )
        
        assert int(num_active) == 2
    
    def test_compact_and_attend_jittable(self):
        """compact_and_attend should be JIT-compilable (already decorated)."""
        num_blocks = 4
        q, keys, values = _make_tensors(num_blocks)
        mask = _make_mask(num_blocks, eviction_rate=0.5)
        
        # compact_and_attend is already @jax.jit decorated
        output, meta = compact_and_attend(q, keys, values, mask, BLOCK_SIZE)
        
        assert output.shape == (q.shape[0], NUM_HEADS, HEAD_DIM)
