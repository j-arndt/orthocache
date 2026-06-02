"""Dynamic-bound attention kernel using jax.lax.while_loop.

This replaces the Pallas unrolled-loop kernel with a JAX-native kernel
that uses jax.lax.while_loop for truly dynamic iteration. XLA compiles
while_loop to a real While HLO instruction whose condition is checked
at runtime — meaning the loop terminates when i >= num_active.

This is the Phase D kernel. It runs on TPU without any C++ or XLA
compiler modifications.

Key difference from sparse_attention.py:
  - sparse_attention.py: `for b in range(num_blocks):` → Python unrolls
    at trace time → XLA sees N sequential blocks → predication
  - This file: `jax.lax.while_loop(cond, body, init)` → XLA sees a
    real While instruction → dynamic termination → true loop elision
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['block_size'])
def dynamic_compact_attention(
    q: jax.Array,
    compact_keys: jax.Array,
    compact_values: jax.Array,
    num_active: jax.Array,
    block_size: int = 512,
) -> jax.Array:
    """Attention over compacted KV-cache with dynamic loop bound.
    
    Uses jax.lax.while_loop so XLA emits a real While HLO instruction
    that terminates when i >= num_active, achieving true loop elision.
    
    Args:
        q: Query tensor (seq_len_q, head_dim). Single head.
        compact_keys: Compacted key blocks (num_blocks * block_size, head_dim).
            First num_active * block_size entries are real data.
        compact_values: Same shape as compact_keys.
        num_active: Scalar int32. Number of active (non-evicted) blocks.
        block_size: Tokens per block.
        
    Returns:
        Attention output (seq_len_q, head_dim).
    """
    seq_len_q, head_dim = q.shape
    scale = jnp.float32(1.0 / jnp.sqrt(jnp.float32(head_dim)))
    
    def single_query_attention(q_vec):
        """Process one query vector against all active blocks."""
        # State: (loop_var, running_max, running_sum, running_output)
        init_state = (
            jnp.int32(0),                              # i
            jnp.float32(-1e9),                          # r_max
            jnp.float32(0.0),                           # r_sum
            jnp.zeros((head_dim,), dtype=jnp.float32),  # r_out
        )
        
        def cond_fn(state):
            i, _, _, _ = state
            return i < num_active
        
        def body_fn(state):
            i, r_max, r_sum, r_out = state
            
            # Load block i using dynamic_slice (XLA-friendly dynamic indexing)
            block_start = i * block_size
            k_block = jax.lax.dynamic_slice(
                compact_keys, (block_start, 0), (block_size, head_dim)
            )  # (block_size, head_dim)
            v_block = jax.lax.dynamic_slice(
                compact_values, (block_start, 0), (block_size, head_dim)
            )  # (block_size, head_dim)
            
            # Compute logits: q_vec . k_block^T
            logits = jnp.dot(k_block, q_vec) * scale  # (block_size,)
            
            # Online softmax: find local max, update running stats
            local_max = jnp.max(logits)
            new_max = jnp.maximum(r_max, local_max)
            
            exp_logits = jnp.exp(logits - new_max)        # (block_size,)
            scale_old = jnp.exp(r_max - new_max)           # scalar
            
            new_sum = r_sum * scale_old + jnp.sum(exp_logits)
            new_out = r_out * scale_old + jnp.dot(exp_logits, v_block)  # (head_dim,)
            
            return (i + 1, new_max, new_sum, new_out)
        
        _, _, final_sum, final_out = jax.lax.while_loop(cond_fn, body_fn, init_state)
        
        # Normalize
        return final_out / jnp.maximum(final_sum, 1e-9)
    
    # vmap over query positions
    q_f32 = q.astype(jnp.float32)
    output = jax.vmap(single_query_attention)(q_f32)
    
    return output


@partial(jax.jit, static_argnames=['block_size'])
def dynamic_multihead_attention(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    block_mask: jax.Array,
    block_size: int = 512,
) -> jax.Array:
    """Full multi-head attention with dynamic compaction and while_loop.
    
    Uses a UNIFIED num_active across all heads (any-head retention) so
    we can vmap the while_loop kernel across heads for parallel execution.
    
    Args:
        q: Query (seq_len_q, num_heads, head_dim).
        keys: Keys (seq_len_k, num_heads, head_dim).
        values: Values (seq_len_k, num_heads, head_dim).
        block_mask: Boolean mask (num_blocks, num_heads).
        block_size: Tokens per block.
        
    Returns:
        Attention output (seq_len_q, num_heads, head_dim).
    """
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_k = keys.shape[0]
    num_blocks = seq_len_k // block_size
    
    # Unified mask: retain block if ANY head says retain
    block_active = jnp.any(block_mask, axis=-1)  # (num_blocks,)
    active_int = block_active.astype(jnp.int32)
    num_active = jnp.sum(active_int)
    
    # Sort order: active blocks first (stable preserves relative order)
    sort_order = jnp.argsort(-active_int, stable=True)  # (num_blocks,)
    
    # Compact ALL heads at once using the unified sort order
    # keys: (seq_len_k, num_heads, head_dim) → (num_blocks, block_size, num_heads, head_dim)
    k_blocked = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    v_blocked = values.reshape(num_blocks, block_size, num_heads, head_dim)
    
    compact_k = k_blocked[sort_order]  # (num_blocks, block_size, num_heads, head_dim)
    compact_v = v_blocked[sort_order]
    
    # Transpose to (num_heads, num_blocks * block_size, head_dim) for vmap
    compact_k_flat = compact_k.reshape(num_blocks * block_size, num_heads, head_dim)
    compact_v_flat = compact_v.reshape(num_blocks * block_size, num_heads, head_dim)
    
    # (num_heads, seq_len_k, head_dim)
    compact_k_heads = jnp.transpose(compact_k_flat, (1, 0, 2))
    compact_v_heads = jnp.transpose(compact_v_flat, (1, 0, 2))
    
    # q: (seq_len_q, num_heads, head_dim) → (num_heads, seq_len_q, head_dim)
    q_heads = jnp.transpose(q, (1, 0, 2))
    
    # vmap dynamic_compact_attention over the head dimension
    # Each head gets the same num_active (unified mask)
    vmapped_attn = jax.vmap(
        lambda q_h, k_h, v_h: dynamic_compact_attention(
            q_h, k_h, v_h, num_active, block_size=block_size
        )
    )
    
    # (num_heads, seq_len_q, head_dim)
    out_heads = vmapped_attn(q_heads, compact_k_heads, compact_v_heads)
    
    # Transpose back: (num_heads, seq_len_q, head_dim) → (seq_len_q, num_heads, head_dim)
    return jnp.transpose(out_heads, (1, 0, 2))

