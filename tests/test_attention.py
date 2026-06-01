import pytest
import numpy as np
import jax
import jax.numpy as jnp
from orthocache.sparse_attention import jax_block_sparse_attention, compile_pallas_sparse_attention
from orthocache.reference import compute_tv_distance

def test_sparse_attention_correctness():
    np.random.seed(42)
    seq_len_q = 8
    seq_len_k = 1024
    num_heads = 2
    head_dim = 64
    block_size = 512
    
    q = np.random.randn(seq_len_q, num_heads, head_dim).astype(np.float32)
    keys = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)
    values = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)
    
    # Generate a block mask (2 blocks of size 512)
    # Let block 0 be retained and block 1 evicted for head 0, and vice versa for head 1
    block_mask = jnp.array([[True, False], [False, True]], dtype=bool)
    
    # Compute using JAX block sparse attention
    output_jax = jax_block_sparse_attention(
        jnp.array(q), jnp.array(keys), jnp.array(values), block_mask, block_size
    )
    
    # Compute using pallas compile wrapper (which runs fallback on CPU)
    output_pallas = compile_pallas_sparse_attention(
        jnp.array(q), jnp.array(keys), jnp.array(values), block_mask, block_size
    )
    
    # Check shape and equivalence of both routes
    assert output_jax.shape == (seq_len_q, num_heads, head_dim)
    np.testing.assert_allclose(output_jax, output_pallas, rtol=1e-5, atol=1e-5)
    
    # Let's verify that TV distance works as expected
    # Dense/full attention can be computed with all True block_mask
    full_mask = jnp.ones_like(block_mask, dtype=bool)
    dense_out = jax_block_sparse_attention(
        jnp.array(q), jnp.array(keys), jnp.array(values), full_mask, block_size
    )
    
    # Simple sanity check: sparse and dense attention outputs differ slightly due to eviction
    diff = np.max(np.abs(dense_out - output_jax))
    assert diff > 0.0
