import pytest
import numpy as np
import jax
import jax.numpy as jnp
from orthocache.spectral_energy import compute_block_energy_jax, generate_threshold_mask
from orthocache.reference import compute_block_energy_reference

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

def test_threshold_mask():
    energies = jnp.array([[10.5, 2.3], [1.1, 15.6]])
    epsilon = 5.0
    mask = generate_threshold_mask(energies, epsilon)
    
    expected_mask = np.array([[True, False], [False, True]])
    np.testing.assert_equal(np.array(mask), expected_mask)
