import pytest
import numpy as np
import jax
import jax.numpy as jnp
from orthocache.spectral_energy import (
    compute_block_energy_jax,
    generate_threshold_mask,
    compute_query_aware_bounds,
    compute_query_aware_mask,
)
from orthocache.reference import (
    compute_block_energy_reference,
    compute_query_aware_bounds_reference,
)

def test_spectral_energy_correctness():
    np.random.seed(42)
    seq_len = 1024
    num_heads = 4
    head_dim = 64
    block_size = 512
    
    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    
    # Compute reference spectral energies
    ref_energies = compute_block_energy_reference(keys, block_size)
    
    # Compute JAX spectral energies
    jax_energies = compute_block_energy_jax(jnp.array(keys), block_size)
    
    # Compare energy results
    np.testing.assert_allclose(jax_energies, ref_energies, rtol=1e-5, atol=1e-5)

def test_query_aware_bounds_correctness():
    np.random.seed(42)
    seq_len_k = 1024
    seq_len_q = 8
    num_heads = 4
    head_dim = 64
    block_size = 512
    
    q = np.random.randn(seq_len_q, num_heads, head_dim).astype(np.float32)
    keys = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)
    
    # Compute reference query-aware bounds
    ref_bounds = compute_query_aware_bounds_reference(q, keys, block_size)
    
    # Compute JAX query-aware bounds
    jax_bounds = compute_query_aware_bounds(jnp.array(q), jnp.array(keys), block_size)
    
    # Compare bound results
    np.testing.assert_allclose(jax_bounds, ref_bounds, rtol=1e-5, atol=1e-5)

def test_query_aware_mask():
    np.random.seed(42)
    seq_len_k = 1024
    seq_len_q = 1
    num_heads = 4
    head_dim = 64
    block_size = 512
    
    q = np.random.randn(seq_len_q, num_heads, head_dim).astype(np.float32)
    keys = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)
    
    # Generate mask with threshold
    tau = 0.5
    mask = compute_query_aware_mask(jnp.array(q), jnp.array(keys), tau, block_size)
    
    # Check shape (num_blocks, num_heads)
    assert mask.shape == (seq_len_k // block_size, num_heads)
    assert mask.dtype == bool

def test_threshold_mask():
    energies = jnp.array([[10.5, 2.3], [1.1, 15.6]])
    epsilon = 5.0
    mask = generate_threshold_mask(energies, epsilon)
    
    expected_mask = np.array([[True, False], [False, True]])
    np.testing.assert_equal(np.array(mask), expected_mask)

