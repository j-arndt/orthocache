import pytest
import numpy as np
import jax
import jax.numpy as jnp
from orthocache.fwht import fwht_512
from orthocache.reference import numpy_fwht

def test_fwht_correctness():
    # Set up random input keys (512 tokens, 128 dimensions)
    np.random.seed(42)
    a = np.random.randn(512, 128).astype(np.float32)
    
    # Compute NumPy reference
    ref_out = numpy_fwht(a)
    
    # Compute JAX implementation
    jax_out = fwht_512(jnp.array(a))
    
    # Assert close within appropriate numerical precision
    np.testing.assert_allclose(jax_out, ref_out, rtol=1e-5, atol=1e-5)

def test_fwht_1d():
    np.random.seed(42)
    a = np.random.randn(512).astype(np.float32)
    
    ref_out = numpy_fwht(a[:, None]).squeeze(1)
    jax_out = fwht_512(jnp.array(a))
    
    np.testing.assert_allclose(jax_out, ref_out, rtol=1e-5, atol=1e-5)
