"""Indirect Pallas kernel: no data copy, just an index table.

The KV-cache stays in place. Instead of physically moving blocks,
we pass an indirection table to the Pallas kernel. The kernel loops
over `bucket_size` iterations (fewer than `num_blocks`) and uses
dynamic_slice with `active_indices[i]` to jump directly to the
relevant blocks.

This combines:
- Pallas MXU efficiency (unrolled, pipelined, VMEM-resident)
- Proportional Δτ scaling (fewer iterations = less work)
- Zero data copy overhead (KV stays in HBM/VMEM as-is)
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial

BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

def _next_bucket(n):
    for b in BUCKETS:
        if b >= n:
            return b
    return n


def indirect_pallas_kernel(q_ref, k_ref, v_ref, idx_ref, out_ref, block_size, bucket_size):
    """Pallas kernel with indirect block indexing.
    
    Instead of `for b in range(num_blocks)`, iterates over `bucket_size`
    entries from the indirection table. Each entry tells us which original
    block to load via dynamic_slice.
    """
    q = q_ref[...]  # (seq_len_q, 1, head_dim)
    q = q.squeeze(1)  # (seq_len_q, head_dim)
    
    head_dim = q.shape[-1]
    scale = jnp.sqrt(jnp.float32(head_dim))
    seq_len_q = q.shape[0]
    
    # Load the full K, V for this head from VMEM
    # k_ref shape: (seq_len_k, 1, head_dim) via BlockSpec
    k_all = k_ref[:, 0, :]  # (seq_len_k, head_dim)
    v_all = v_ref[:, 0, :]  # (seq_len_k, head_dim)
    
    # Load indirection table for this head
    idx_all = idx_ref[:, 0]  # (bucket_size,) int32
    
    # Initialize online softmax accumulators
    r_max = jnp.full((seq_len_q, 1), -1e9, dtype=jnp.float32)
    r_sum = jnp.zeros((seq_len_q, 1), dtype=jnp.float32)
    r_out = jnp.zeros((seq_len_q, head_dim), dtype=jnp.float32)
    
    # Iterate over ONLY active blocks (bucket_size <= num_blocks)
    for i in range(bucket_size):
        # Look up original block index from indirection table
        orig_b = idx_all[i]
        start = orig_b * block_size
        
        # Dynamic slice into the ORIGINAL (non-compacted) K/V
        k_block = jax.lax.dynamic_slice(k_all, (start, 0), (block_size, head_dim))
        v_block = jax.lax.dynamic_slice(v_all, (start, 0), (block_size, head_dim))
        
        # Standard online softmax attention
        logits = jnp.matmul(q, k_block.T) / scale  # (seq_len_q, block_size)
        
        local_max = jnp.max(logits, axis=-1, keepdims=True)
        new_max = jnp.maximum(r_max, local_max)
        
        exp_logits = jnp.exp(logits - new_max)
        sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)
        
        scale_old = jnp.exp(r_max - new_max)
        
        r_sum = r_sum * scale_old + sum_exp
        r_out = r_out * scale_old + jnp.matmul(exp_logits, v_block)
        r_max = new_max
    
    # Normalize
    final_out = r_out / jnp.maximum(r_sum, 1e-9)
    out_ref[...] = final_out[:, jnp.newaxis, :]


def compile_indirect_attention(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    active_indices: jax.Array,
    bucket_size: int,
    block_size: int = 512,
) -> jax.Array:
    """Compile and run the indirect Pallas kernel.
    
    Args:
        q: (seq_len_q, num_heads, head_dim)
        keys: (seq_len_k, num_heads, head_dim) — ORIGINAL, not compacted
        values: same as keys
        active_indices: (bucket_size, num_heads) int32 — block indices per head
        bucket_size: number of active blocks (static int for Pallas tracing)
        block_size: tokens per block
    """
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_k = keys.shape[0]
    
    out_shape = jax.ShapeDtypeStruct((seq_len_q, num_heads, head_dim), q.dtype)
    
    out = pl.pallas_call(
        lambda q_r, k_r, v_r, i_r, o_r: indirect_pallas_kernel(
            q_r, k_r, v_r, i_r, o_r, block_size, bucket_size
        ),
        out_shape=out_shape,
        grid=(num_heads,),
        in_specs=[
            pl.BlockSpec(lambda h: (0, h, 0), (seq_len_q, 1, head_dim)),    # q
            pl.BlockSpec(lambda h: (0, h, 0), (seq_len_k, 1, head_dim)),    # keys
            pl.BlockSpec(lambda h: (0, h, 0), (seq_len_k, 1, head_dim)),    # values
            pl.BlockSpec(lambda h: (0, h), (bucket_size, 1)),               # indices
        ],
        out_specs=pl.BlockSpec(lambda h: (0, h, 0), (seq_len_q, 1, head_dim)),
    )(q, keys, values, active_indices)
    
    return out


def indirect_attention(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    block_mask: jax.Array,
    block_size: int = 512,
) -> tuple[jax.Array, dict]:
    """Top-level indirect attention: compute indices, dispatch to Pallas.
    
    Zero data copy. The KV-cache stays in place.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    num_blocks = seq_len_k // block_size
    
    # Compute active indices per head (or unified across heads)
    block_active = jnp.any(block_mask, axis=-1)  # (num_blocks,)
    active_int = block_active.astype(jnp.int32)
    num_active_dev = jnp.sum(active_int)
    sort_order = jnp.argsort(-active_int, stable=True)  # (num_blocks,)
    
    # Host sync for bucketing
    n_active = int(num_active_dev)
    
    if n_active == 0:
        return jnp.zeros_like(q), {'num_active': 0, 'bucket': 0, 'eviction_rate': 1.0}
    
    bucket = _next_bucket(n_active)
    
    # Build indirection table: (bucket, num_heads)
    # Same indices for all heads (unified mask)
    active_idx = sort_order[:bucket]  # (bucket,)
    idx_table = jnp.broadcast_to(active_idx[:, None], (bucket, num_heads))
    
    # Dispatch to Pallas with reduced bucket_size
    output = compile_indirect_attention(
        q, keys, values, idx_table, bucket_size=bucket, block_size=block_size
    )
    
    return output, {
        'num_active': n_active,
        'bucket': bucket,
        'num_blocks': num_blocks,
        'eviction_rate': 1.0 - n_active / num_blocks,
    }
