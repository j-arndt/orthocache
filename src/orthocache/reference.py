import numpy as np

def numpy_fwht_1d(a: np.ndarray) -> np.ndarray:
    """Computes the 1D Fast Walsh-Hadamard Transform using the butterfly algorithm.
    
    Args:
        a: 1D array of length power-of-2 (e.g. 512).
        
    Returns:
        The transformed 1D array, normalized by 1 / sqrt(len(a)).
    """
    x = a.copy().astype(np.float64)
    h = 1
    n = len(x)
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                u = x[j]
                v = x[j + h]
                x[j] = u + v
                x[j + h] = u - v
        h *= 2
    return x / np.sqrt(n)

def numpy_fwht(tile: np.ndarray) -> np.ndarray:
    """Computes the FWHT of a 2D tile along the first (row/sequence) axis.
    
    Args:
        tile: 2D array of shape (num_tokens, head_dim) where num_tokens is a power-of-2.
        
    Returns:
        The row-wise transformed 2D array.
    """
    transformed = np.zeros_like(tile, dtype=np.float64)
    for d in range(tile.shape[1]):
        transformed[:, d] = numpy_fwht_1d(tile[:, d])
    return transformed.astype(tile.dtype)

def compute_block_energy_reference(keys: np.ndarray, block_size: int = 512) -> np.ndarray:
    """Computes the reference spectral energy per block using the numpy FWHT.
    
    Args:
        keys: 3D array of shape (seq_len, num_heads, head_dim).
        block_size: Size of block to segment sequence into (e.g. 512).
        
    Returns:
        An array of shape (num_blocks, num_heads) containing spectral energy per block.
    """
    seq_len, num_heads, head_dim = keys.shape
    num_blocks = seq_len // block_size
    energies = np.zeros((num_blocks, num_heads), dtype=np.float64)
    
    for h in range(num_heads):
        for b in range(num_blocks):
            block_keys = keys[b * block_size : (b + 1) * block_size, h, :]
            spectral = numpy_fwht(block_keys)
            energies[b, h] = np.sum(spectral ** 2)
            
    return energies

def compute_query_aware_bounds_reference(q: np.ndarray, keys: np.ndarray, block_size: int = 512) -> np.ndarray:
    """Computes the reference query-aware bounds per block using numpy FWHT.
    
    Args:
        q: Query array of shape (seq_len_q, num_heads, head_dim).
        keys: Key array of shape (seq_len_k, num_heads, head_dim).
        block_size: Size of block to segment sequence into (e.g. 512).
        
    Returns:
        An array of shape (seq_len_q, num_blocks, num_heads) containing logit bounds.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    seq_len_q = q.shape[0]
    num_blocks = seq_len_k // block_size
    bounds = np.zeros((seq_len_q, num_blocks, num_heads), dtype=np.float64)
    
    for h in range(num_heads):
        for b in range(num_blocks):
            block_keys = keys[b * block_size : (b + 1) * block_size, h, :]
            spectral = numpy_fwht(block_keys)
            
            dc = spectral[0]
            ac = spectral[1:]
            ac_energy = np.sum(ac ** 2)
            
            block_mean = dc / np.sqrt(block_size)
            
            for qi in range(seq_len_q):
                q_vec = q[qi, h, :]
                alignment = np.dot(q_vec, block_mean) / np.sqrt(head_dim)
                q_norm = np.linalg.norm(q_vec)
                residual = (q_norm * np.sqrt(ac_energy)) / np.sqrt(head_dim)
                bounds[qi, b, h] = alignment + residual
                
    return bounds


def compute_tv_distance(alpha: np.ndarray, alpha_hat: np.ndarray) -> float:
    """Computes the Total Variation (TV) distance between two attention distributions.
    
    Args:
        alpha: Full attention probability distribution (1D or 2D).
        alpha_hat: Truncated attention probability distribution (same shape as alpha).
        
    Returns:
        The Total Variation distance.
    """
    return 0.5 * np.sum(np.abs(alpha - alpha_hat))
