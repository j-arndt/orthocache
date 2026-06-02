"""Lean bucketed attention: no physical compaction, just indirect gather + reduced Pallas.

The expensive part of bucketed_attention.py was stream_compact copying
512MB of KV data. This version:
1. Computes active_indices from the mask (argsort on 64 elements — microseconds)
2. Gathers ONLY the active blocks using advanced indexing  
3. Runs dense einsum attention on the reduced set (no Pallas needed)

The gather + einsum gets fused by XLA into a single pass.
"""

import jax
import jax.numpy as jnp
from functools import partial

BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

def _next_bucket(n):
    for b in BUCKETS:
        if b >= n:
            return b
    return n


def lean_bucketed_attention(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    block_mask: jax.Array,
    block_size: int = 512,
) -> tuple[jax.Array, dict]:
    """Lean bucketed attention: gather active blocks + dense attention.
    
    No stream_compact. No full-tensor copy. Just index the blocks we need.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    num_blocks = seq_len_k // block_size
    
    # Step 1: Compute active indices (CHEAP — operates on tiny mask array)
    block_active = jnp.any(block_mask, axis=-1)  # (num_blocks,)
    active_int = block_active.astype(jnp.int32)
    num_active_dev = jnp.sum(active_int)
    sort_order = jnp.argsort(-active_int, stable=True)  # (num_blocks,)
    
    # Step 2: Pull num_active to host for bucketing (single int32 sync)
    n_active = int(num_active_dev)
    
    if n_active == 0:
        return jnp.zeros_like(q), {'num_active': 0, 'bucket': 0, 'eviction_rate': 1.0}
    
    bucket = _next_bucket(n_active)
    active_idx = sort_order[:bucket]  # (bucket,) — just indices, no data copy yet
    
    # Step 3: Gather active blocks (XLA fuses this with the attention einsum)
    k_blocked = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    v_blocked = values.reshape(num_blocks, block_size, num_heads, head_dim)
    
    k_active = k_blocked[active_idx]  # (bucket, block_size, num_heads, head_dim)
    v_active = v_blocked[active_idx]
    
    # Flatten: (bucket * block_size, num_heads, head_dim)
    k_flat = k_active.reshape(bucket * block_size, num_heads, head_dim)
    v_flat = v_active.reshape(bucket * block_size, num_heads, head_dim)
    
    # Step 4: Dense attention on reduced set
    scale = jnp.sqrt(jnp.float32(head_dim))
    q_f32 = q.astype(jnp.float32)
    k_f32 = k_flat.astype(jnp.float32)
    v_f32 = v_flat.astype(jnp.float32)
    
    logits = jnp.einsum('qhd,khd->qkh', q_f32, k_f32) / scale
    weights = jax.nn.softmax(logits, axis=1)
    output = jnp.einsum('qkh,khd->qhd', weights, v_f32)
    
    return output, {
        'num_active': n_active,
        'bucket': bucket,
        'num_blocks': num_blocks,
        'eviction_rate': 1.0 - n_active / num_blocks,
    }
