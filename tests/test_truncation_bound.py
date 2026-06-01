"""Test that the OrthoCache Truncation Bound holds empirically.

Validates that for random Q/K/V data, the measured Total Variation distance
between full and truncated attention distributions is bounded by the
theoretical OrthoCache exponential bound:

    TV(α, α̂) ≤ |S^c| · exp(β - z_max)

where β = ||q|| · sqrt(ε) / sqrt(d_k) and z_max = max retained logit.
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from orthocache.fwht import fwht_512
from orthocache.spectral_energy import compute_block_energy_jax, generate_threshold_mask
from orthocache.sparse_attention import jax_block_sparse_attention
from orthocache.reference import compute_tv_distance


def test_truncation_bound_holds():
    """Verify that the theoretical TV bound holds for random data at multiple thresholds."""
    np.random.seed(2026)
    seq_len_k = 2048  # 4 blocks of 512
    num_heads = 2
    head_dim = 64
    block_size = 512
    num_blocks = seq_len_k // block_size

    # Create random Q, K, V
    q = np.random.randn(1, num_heads, head_dim).astype(np.float32)
    keys = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)
    values = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)

    # Compute block energies
    energies = np.array(compute_block_energy_jax(jnp.array(keys), block_size))

    # Test across multiple thresholds — target different eviction rates
    for epsilon_percentile in [25, 50, 75]:
        # Set epsilon based on a percentile of the energy distribution
        epsilon = float(np.percentile(energies, epsilon_percentile))
        if epsilon <= 0:
            continue

        # Generate block mask
        block_mask = np.array(generate_threshold_mask(jnp.array(energies), epsilon))

        # Skip if all blocks are retained or all evicted
        if block_mask.all() or not block_mask.any():
            continue

        # For each head, verify the bound
        for h in range(num_heads):
            head_mask = block_mask[:, h]  # (num_blocks,)

            # Count evicted tokens
            num_evicted_blocks = int(np.sum(~head_mask))
            num_evicted_tokens = num_evicted_blocks * block_size

            if num_evicted_tokens == 0:
                continue

            # Compute full dense attention weights for this head
            q_h = q[0, h, :]  # (head_dim,)
            k_h = keys[:, h, :]  # (seq_len_k, head_dim)

            # Raw logits
            logits = k_h @ q_h / np.sqrt(head_dim)  # (seq_len_k,)

            # Full softmax
            logits_shifted = logits - np.max(logits)
            exp_logits = np.exp(logits_shifted)
            alpha_full = exp_logits / np.sum(exp_logits)

            # Build per-token mask from block mask
            token_mask = np.repeat(head_mask, block_size)  # (seq_len_k,)

            # Truncated softmax (only over retained tokens)
            exp_retained = exp_logits * token_mask
            Z_hat = np.sum(exp_retained)
            if Z_hat == 0:
                continue
            alpha_hat = np.where(token_mask, exp_retained / Z_hat, 0.0)

            # Measured TV distance
            tv_measured = compute_tv_distance(alpha_full, alpha_hat)

            # Theoretical bound: TV ≤ |S^c| * exp(beta - z_max)
            q_norm = np.linalg.norm(q_h)
            beta = q_norm * np.sqrt(epsilon) / np.sqrt(head_dim)

            # z_max = max logit among retained tokens (unshifted)
            retained_logits = logits[token_mask.astype(bool)]
            z_max = float(np.max(retained_logits))

            theoretical_bound = num_evicted_tokens * np.exp(beta - z_max)

            # The bound should hold (with numerical margin)
            assert tv_measured <= theoretical_bound + 1e-7, (
                f"TV bound violated at percentile={epsilon_percentile}, head={h}: "
                f"TV_measured={tv_measured:.6f} > bound={theoretical_bound:.6f} "
                f"(|S^c|={num_evicted_tokens}, beta={beta:.4f}, z_max={z_max:.4f})"
            )


def test_truncation_bound_tight_at_extreme():
    """Verify that with very aggressive eviction, the bound still holds."""
    np.random.seed(42)
    seq_len_k = 1024  # 2 blocks of 512
    num_heads = 1
    head_dim = 32
    block_size = 512

    # Create data where one block has very low energy (near-zero keys)
    keys = np.random.randn(seq_len_k, num_heads, head_dim).astype(np.float32)
    keys[:block_size, :, :] *= 0.001  # Make first block near-zero energy

    q = np.random.randn(1, num_heads, head_dim).astype(np.float32)

    # Compute energies
    energies = np.array(compute_block_energy_jax(jnp.array(keys), block_size))

    # Use a threshold that evicts only the low-energy block
    epsilon = float(np.mean(energies))
    block_mask = np.array(generate_threshold_mask(jnp.array(energies), epsilon))

    # Verify the first block is evicted and second retained
    assert not block_mask[0, 0], "Expected first block to be evicted"
    assert block_mask[1, 0], "Expected second block to be retained"

    # Compute TV distance
    k_h = keys[:, 0, :]
    q_h = q[0, 0, :]
    logits = k_h @ q_h / np.sqrt(head_dim)

    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    alpha_full = exp_logits / np.sum(exp_logits)

    token_mask = np.repeat(block_mask[:, 0], block_size)
    exp_retained = exp_logits * token_mask
    Z_hat = np.sum(exp_retained)
    alpha_hat = np.where(token_mask, exp_retained / Z_hat, 0.0)

    tv_measured = compute_tv_distance(alpha_full, alpha_hat)

    # Theoretical bound
    q_norm = np.linalg.norm(q_h)
    beta = q_norm * np.sqrt(epsilon) / np.sqrt(head_dim)
    retained_logits = logits[token_mask.astype(bool)]
    z_max = float(np.max(retained_logits))
    num_evicted = int(np.sum(~token_mask))

    theoretical_bound = num_evicted * np.exp(beta - z_max)

    assert tv_measured <= theoretical_bound + 1e-7, (
        f"TV bound violated: TV={tv_measured:.6f} > bound={theoretical_bound:.6f}"
    )

    # With near-zero keys in evicted block, TV should be moderate
    # (evicting 50% of tokens produces measurable but bounded distributional shift)
    assert tv_measured < 0.5, f"TV distance unexpectedly large: {tv_measured:.4f}"
