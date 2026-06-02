"""Bucketed compaction + Pallas attention.

Instead of a dynamic while_loop (high per-iteration overhead) or a
predicated unrolled loop (zero speedup), this module:

1. Stream-compacts the KV-cache (active blocks first)
2. Pulls num_active to host (single scalar sync)
3. Rounds up to nearest bucket size (power of 2)
4. Calls the standard Pallas kernel with bucket blocks instead of all blocks

The Pallas kernel is unrolled to `bucket` iterations, not `num_blocks`.
At 50% eviction with 64 blocks, Pallas unrolls to 32 iterations — half
the MXU work, same pipelining efficiency.

JAX auto-caches compiled kernels per bucket size, so the first call for
each bucket size incurs a compilation cost, but subsequent calls are instant.
"""

import jax
import jax.numpy as jnp
from functools import partial

from orthocache.compaction import stream_compact
from orthocache.sparse_attention import compile_pallas_sparse_attention


# Bucket sizes: powers of 2 from 1 to 512.
# Covers up to 256K context at block_size=512.
BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def _next_bucket(n: int) -> int:
    """Round up to the nearest bucket size."""
    for b in BUCKETS:
        if b >= n:
            return b
    return n  # Fallback: use exact count if > max bucket


def bucketed_attention(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    block_mask: jax.Array,
    block_size: int = 512,
) -> tuple[jax.Array, dict]:
    """Compacted attention using bucketed Pallas dispatch.
    
    This is the Phase D kernel that achieves both:
    - Pallas-level MXU efficiency (unrolled, pipelined)
    - Proportional Δτ scaling with eviction rate
    
    Args:
        q: Query (seq_len_q, num_heads, head_dim).
        keys: Keys (seq_len_k, num_heads, head_dim).
        values: Values (seq_len_k, num_heads, head_dim).
        block_mask: Boolean (num_blocks, num_heads).
        block_size: Tokens per block.
        
    Returns:
        (output, metadata) where output is (seq_len_q, num_heads, head_dim).
    """
    seq_len_k, num_heads, head_dim = keys.shape
    num_blocks = seq_len_k // block_size
    
    # Step 1: Stream compact — active blocks first, padded with zeros
    compact_keys, compact_values, active_indices, num_active = stream_compact(
        keys, values, block_mask, block_size
    )
    # compact_keys: (num_blocks, block_size, num_heads, head_dim)
    # num_active: scalar int32 on device
    
    # Step 2: Pull num_active to host (single scalar transfer)
    # This is the only sync point. ~microseconds for one int32.
    n_active = int(num_active)
    
    # Step 3: Determine bucket
    if n_active == 0:
        # All blocks evicted — return zeros
        return jnp.zeros_like(q), {
            'num_active': 0, 'bucket': 0, 'num_blocks': num_blocks,
            'eviction_rate': 1.0,
        }
    
    bucket = _next_bucket(n_active)
    
    # Step 4: Slice compact tensor to bucket size
    # compact_keys[:bucket] gives us the first `bucket` blocks
    ck = compact_keys[:bucket]  # (bucket, block_size, num_heads, head_dim)
    cv = compact_values[:bucket]
    
    # Reshape to (bucket * block_size, num_heads, head_dim)
    ck_flat = ck.reshape(bucket * block_size, num_heads, head_dim)
    cv_flat = cv.reshape(bucket * block_size, num_heads, head_dim)
    
    # All-true mask (every block in the compact tensor is active)
    bucket_mask = jnp.ones((bucket, num_heads), dtype=jnp.bool_)
    
    # Step 5: Pallas kernel with REDUCED block count
    # Pallas traces with `bucket` blocks → unrolled to `bucket` iterations
    # JAX caches the compiled kernel per bucket size
    output = compile_pallas_sparse_attention(
        q, ck_flat, cv_flat, bucket_mask, block_size
    )
    
    metadata = {
        'num_active': n_active,
        'bucket': bucket,
        'num_blocks': num_blocks,
        'eviction_rate': 1.0 - n_active / num_blocks,
    }
    
    return output, metadata
