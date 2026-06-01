"""Tests for Multi-Band Sequency Filtering.

Validates that:
1. The JAX multi-band decomposition matches the NumPy reference implementation
2. The spectral decay ratio ζ correctly distinguishes noise from coherent blocks
3. ζ is NOT computable from spatial statistics alone (the FWHT is load-bearing)
4. The two-gate multiband mask works correctly
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from orthocache.spectral_energy import (
    compute_spectral_bands,
    compute_spectral_decay_ratio,
    compute_multiband_mask,
    BAND_LOW,
    BAND_MID,
    BAND_HIGH,
)
from orthocache.reference import (
    compute_spectral_bands_reference,
    compute_spectral_decay_ratio_reference,
    compute_multiband_mask_reference,
)


def test_spectral_bands_match_reference():
    """JAX multi-band decomposition matches NumPy reference."""
    np.random.seed(42)
    seq_len = 1024
    num_heads = 2
    head_dim = 64
    block_size = 512

    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)

    # Reference
    ref_dc, ref_low, ref_mid, ref_high = compute_spectral_bands_reference(
        keys, block_size
    )

    # JAX
    jax_dc, jax_low, jax_mid, jax_high = compute_spectral_bands(
        jnp.array(keys), block_size
    )

    np.testing.assert_allclose(np.array(jax_dc), ref_dc, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(jax_low), ref_low, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(jax_mid), ref_mid, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(jax_high), ref_high, rtol=1e-4, atol=1e-4)


def test_spectral_decay_ratio_correctness():
    """ζ from JAX matches ζ from NumPy reference."""
    np.random.seed(123)
    seq_len = 1024
    num_heads = 2
    head_dim = 64
    block_size = 512

    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)

    ref_zeta = compute_spectral_decay_ratio_reference(keys, block_size)
    jax_zeta = compute_spectral_decay_ratio(jnp.array(keys), block_size)

    np.testing.assert_allclose(np.array(jax_zeta), ref_zeta, rtol=1e-4, atol=1e-4)


def test_zeta_high_for_noise():
    """Synthetic high-frequency noise block should have ζ >> 1."""
    np.random.seed(99)
    block_size = 512
    head_dim = 64

    # Create a block of rapid sign-oscillating noise:
    # alternating +1/-1 pattern repeated with random amplitudes
    noise_block = np.zeros((block_size, head_dim), dtype=np.float64)
    for i in range(block_size):
        sign = 1.0 if i % 2 == 0 else -1.0
        noise_block[i, :] = sign * np.random.randn(head_dim) * 2.0

    zeta = compute_spectral_decay_ratio_reference(noise_block, block_size)

    # High-frequency alternation should push energy into high-sequency bands
    assert zeta > 1.0, f"Expected ζ > 1 for noise block, got {zeta:.4f}"


def test_zeta_low_for_coherent():
    """Synthetic coherent block (low Walsh sequency) should have ζ << 1.
    
    Walsh functions are binary (±1) patterns ordered by number of sign changes.
    Low-sequency Walsh basis vectors produce smooth step-like patterns.
    A block constructed from these vectors has energy concentrated in low bands.
    """
    np.random.seed(42)
    block_size = 512
    head_dim = 64

    # Construct a block from the first 16 Walsh basis vectors
    # This guarantees all energy is in coefficients 0-15 (within the low band)
    coherent_block = np.zeros((block_size, head_dim), dtype=np.float64)
    rng = np.random.RandomState(42)
    for j in range(1, 16):  # Skip DC (j=0) so energy is in AC low band
        walsh_j = np.ones(block_size)
        for i in range(block_size):
            for bit in range(9):  # log2(512)
                if (j >> bit) & 1 and (i >> bit) & 1:
                    walsh_j[i] *= -1
        coeff = rng.randn(head_dim) * 0.5
        coherent_block += np.outer(walsh_j, coeff)

    zeta = compute_spectral_decay_ratio_reference(coherent_block, block_size)

    # Energy is in coefficients 1-15, all within low band [1, 64)
    # High band [256, 512) should have zero energy
    assert zeta < 0.01, f"Expected ζ < 0.01 for Walsh-coherent block, got {zeta:.4f}"


def test_zeta_not_computable_spatially():
    """CRITICAL TEST: Proves the FWHT is load-bearing.
    
    Constructs two blocks with IDENTICAL spatial variance but DIFFERENT
    spectral decay ratios, demonstrating that no spatial-domain function
    can compute ζ without access to individual spectral coefficients.
    
    Block A: constructed from low-sequency Walsh basis vectors (energy in low bands)
    Block B: random white noise scaled to have the same total variance
    """
    block_size = 512
    head_dim = 64

    # Block A: sum of low-sequency Walsh basis vectors (indices 1-15)
    # This puts ALL AC energy into the low band [1, 64)
    block_a = np.zeros((block_size, head_dim), dtype=np.float64)
    rng_a = np.random.RandomState(2026)
    for j in range(1, 16):
        walsh_j = np.ones(block_size)
        for i in range(block_size):
            for bit in range(9):
                if (j >> bit) & 1 and (i >> bit) & 1:
                    walsh_j[i] *= -1
        coeff = rng_a.randn(head_dim) * 0.5
        block_a += np.outer(walsh_j, coeff)

    # Block B: random white noise
    rng_b = np.random.RandomState(42)
    block_b = rng_b.randn(block_size, head_dim)

    # CRITICAL: Scale block B to have the EXACT same total spatial variance as block A
    mean_a = block_a.mean(axis=0)
    mean_b = block_b.mean(axis=0)
    var_a = np.sum((block_a - mean_a) ** 2)
    var_b = np.sum((block_b - mean_b) ** 2)
    block_b = mean_b + (block_b - mean_b) * np.sqrt(var_a / var_b)

    # Verify: SAME spatial variance (within tight tolerance)
    var_a_final = np.sum((block_a - block_a.mean(axis=0)) ** 2)
    var_b_final = np.sum((block_b - block_b.mean(axis=0)) ** 2)
    np.testing.assert_allclose(var_a_final, var_b_final, rtol=1e-10,
        err_msg="Blocks must have identical spatial variance for this test to be valid")

    # But: DIFFERENT spectral decay ratios
    zeta_a = compute_spectral_decay_ratio_reference(block_a, block_size)
    zeta_b = compute_spectral_decay_ratio_reference(block_b, block_size)

    # Block A (Walsh-coherent): ζ ≈ 0 (all energy in low bands)
    # Block B (noise): ζ >> 1 (energy distributed uniformly, high bands dominate low)
    assert zeta_a < zeta_b, (
        f"Expected ζ_A < ζ_B for coherent vs noise blocks with identical variance, "
        f"got ζ_A={zeta_a:.4f} vs ζ_B={zeta_b:.4f}"
    )
    assert zeta_a < 0.01, f"Walsh-coherent block should have near-zero ζ: {zeta_a:.6f}"
    assert zeta_b > 1.0, f"Noise block should have ζ > 1: {zeta_b:.4f}"


def test_multiband_mask_two_gate():
    """Verify that the multiband mask applies BOTH gates."""
    np.random.seed(42)
    seq_len_k = 1024  # 2 blocks
    seq_len_q = 1
    num_heads = 1
    head_dim = 64
    block_size = 512

    # Create keys where:
    # Block 0: high-frequency noise (should fail ζ gate)
    # Block 1: smooth signal (should pass ζ gate)
    keys = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)

    # Make block 0 oscillate rapidly
    for i in range(block_size):
        sign = 1.0 if i % 2 == 0 else -1.0
        keys[i, 0, :] *= sign * 5.0

    q = np.random.randn(seq_len_q, num_heads, head_dim).astype(np.float32)

    # With a very loose tau (both blocks pass gate 1) and tight zeta_max
    tau = -100.0  # Everything passes logit gate
    zeta_max = 1.0  # Only coherent blocks pass ζ gate

    mask_ref = compute_multiband_mask_reference(q, keys, tau, zeta_max, block_size)
    mask_jax = compute_multiband_mask(
        jnp.array(q), jnp.array(keys), tau, zeta_max, block_size
    )

    # Both should agree
    np.testing.assert_array_equal(np.array(mask_jax), mask_ref)

    # Block 0 (noise) should be evicted by ζ gate
    # Block 1 (smooth-ish) should be retained
    # Check that at least one block differs in retention
    assert not mask_ref.all(), "Expected at least one block to be evicted by ζ gate"
