import jax
import jax.numpy as jnp
from orthocache.fwht import fwht_512

def compute_block_energy_jax(keys: jax.Array, block_size: int = 512) -> jax.Array:
    """Computes the spectral energy of keys per block.
    
    Args:
        keys: JAX array of shape (seq_len, num_heads, head_dim). seq_len must be a multiple of block_size.
        block_size: Size of the blocks (default: 512).
        
    Returns:
        JAX array of shape (num_blocks, num_heads) containing the spectral energy per block.
    """
    seq_len, num_heads, head_dim = keys.shape
    num_blocks = seq_len // block_size
    
    # Reshape to group by block: (num_blocks, block_size, num_heads, head_dim)
    blocks = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    
    # Transpose to put sequence dimension of block as the first axis for fwht_512:
    # (block_size, num_blocks, num_heads, head_dim)
    blocks_t = jnp.transpose(blocks, (1, 0, 2, 3))
    
    # Reshape to flatten other dimensions for batching: (block_size, num_blocks * num_heads * head_dim)
    flat_blocks = blocks_t.reshape(block_size, num_blocks * num_heads * head_dim)
    
    # Compute FWHT
    spectral_flat = fwht_512(flat_blocks)
    
    # Reshape back to (block_size, num_blocks, num_heads, head_dim)
    spectral = spectral_flat.reshape(block_size, num_blocks, num_heads, head_dim)
    
    # Transpose back: (num_blocks, block_size, num_heads, head_dim)
    spectral_orig = jnp.transpose(spectral, (1, 0, 2, 3))
    
    # Compute energy by summing squared coefficients over the block_size and head_dim axes
    energies = jnp.sum(spectral_orig ** 2, axis=(1, 3))
    
    return energies

def generate_threshold_mask(energies: jax.Array, epsilon: float) -> jax.Array:
    """Generates a boolean mask indicating whether blocks are retained.
    
    Args:
        energies: JAX array of shape (num_blocks, num_heads) representing spectral energy.
        epsilon: The threshold value. Blocks with energy >= epsilon are retained (True).
        
    Returns:
        A boolean JAX array of shape (num_blocks, num_heads).
    """
    return energies >= epsilon
