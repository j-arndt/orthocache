import jax
import jax.numpy as jnp
from orthocache.fwht import fwht_512

def compute_block_energy_jax(keys: jax.Array, block_size: int = 512) -> jax.Array:
    """Computes the spatial energy of keys per block (Parseval equivalent).
    
    Args:
        keys: JAX array of shape (seq_len, num_heads, head_dim). seq_len must be a multiple of block_size.
        block_size: Size of the blocks (default: 512).
        
    Returns:
        JAX array of shape (num_blocks, num_heads) containing the spatial energy per block.
    """
    seq_len, num_heads, head_dim = keys.shape
    num_blocks = seq_len // block_size
    blocks = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    # Spatial energy is sum of squared key norms per block
    return jnp.sum(blocks ** 2, axis=(1, 3))

def compute_query_aware_bounds(
    q: jax.Array, keys: jax.Array, block_size: int = 512
) -> jax.Array:
    """Computes query-aware attention bounds per block using Walsh-Hadamard DC/AC decomposition.
    
    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        keys: Key tensor of shape (seq_len_k, num_heads, head_dim).
        block_size: Size of the blocks (default: 512).
        
    Returns:
        JAX array of shape (seq_len_q, num_blocks, num_heads) containing the logit upper bounds.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    num_blocks = seq_len_k // block_size
    
    # Reshape keys to group by block: (num_blocks, block_size, num_heads, head_dim)
    blocks = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    
    # Transpose to put sequence dimension of block as the first axis for fwht_512:
    # (block_size, num_blocks, num_heads, head_dim)
    blocks_t = jnp.transpose(blocks, (1, 0, 2, 3))
    
    # Reshape to flatten other dimensions for batching: (block_size, num_blocks * num_heads * head_dim)
    flat_blocks = blocks_t.reshape(block_size, num_blocks * num_heads * head_dim)
    
    # Compute FWHT of keys
    spectral_flat = fwht_512(flat_blocks)
    
    # Reshape back to (block_size, num_blocks, num_heads, head_dim)
    spectral = spectral_flat.reshape(block_size, num_blocks, num_heads, head_dim)
    
    # Transpose back: (num_blocks, block_size, num_heads, head_dim)
    spectral_orig = jnp.transpose(spectral, (1, 0, 2, 3))
    
    # 0th frequency (DC) coefficient represents the block mean (scaled by sqrt(block_size))
    # DC component: (num_blocks, num_heads, head_dim)
    dc_component = spectral_orig[:, 0, :, :]
    
    # AC components: (num_blocks, block_size - 1, num_heads, head_dim)
    ac_components = spectral_orig[:, 1:, :, :]
    
    # AC energy: (num_blocks, num_heads)
    ac_energy = jnp.sum(ac_components ** 2, axis=(1, 3))
    
    # Mean vector for each block:
    # Since WHT is normalized by 1/sqrt(block_size), the DC term is sum(keys) / sqrt(block_size).
    # Thus, block mean = dc_component / sqrt(block_size).
    block_mean = dc_component / jnp.sqrt(block_size)
    
    # Compute query-mean alignment:
    # q is (seq_len_q, num_heads, head_dim)
    # block_mean is (num_blocks, num_heads, head_dim)
    # Alignment: (seq_len_q, num_blocks, num_heads)
    alignment = jnp.einsum("qhd,bhd->qbh", q, block_mean) / jnp.sqrt(head_dim)
    
    # Compute residual bound using Cauchy-Schwarz on the AC energy:
    # ||q||_2: (seq_len_q, num_heads)
    q_norm = jnp.linalg.norm(q, axis=-1)  # (seq_len_q, num_heads)
    
    # AC residual standard deviation per token in the block:
    # Since WHT is orthogonal, sum_{i} ||k_i - mean||_2^2 = sum_{s > 0} ||spectral_s||_2^2 = ac_energy.
    # Therefore, max_{i} ||k_i - mean||_2 <= sqrt(ac_energy).
    # Thus, residual logit bound = ||q||_2 * sqrt(ac_energy) / sqrt(head_dim).
    # q_norm: (seq_len_q, num_heads) -> (seq_len_q, 1, num_heads)
    # ac_energy: (num_blocks, num_heads) -> (1, num_blocks, num_heads)
    residual_bound = (q_norm[:, jnp.newaxis, :] * jnp.sqrt(ac_energy)[jnp.newaxis, :, :]) / jnp.sqrt(head_dim)
    
    # Total query-aware bound:
    bounds = alignment + residual_bound
    return bounds

def compute_query_aware_mask(
    q: jax.Array, keys: jax.Array, tau: float, block_size: int = 512
) -> jax.Array:
    """Generates a query-aware boolean mask for block eviction.
    
    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        keys: Key tensor of shape (seq_len_k, num_heads, head_dim).
        tau: The threshold value. Blocks with bound >= tau are retained (True).
        block_size: Size of the blocks (default: 512).
        
    Returns:
        A boolean JAX array of shape (num_blocks, num_heads) indicating retained blocks.
    """
    # Compute bounds: (seq_len_q, num_blocks, num_heads)
    bounds = compute_query_aware_bounds(q, keys, block_size)
    
    # Take the maximum bound over the query dimension to ensure
    # that any block that is important for ANY query token is retained.
    max_bounds = jnp.max(bounds, axis=0)  # (num_blocks, num_heads)
    
    return max_bounds >= tau

def generate_threshold_mask(energies: jax.Array, epsilon: float) -> jax.Array:
    """Generates a boolean mask indicating whether blocks are retained (backward compatibility)."""
    return energies >= epsilon

