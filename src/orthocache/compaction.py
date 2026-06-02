"""Stream compaction for OrthoCache block-sparse attention.

Implements the user-space stream compaction primitive that transforms a
block-masked KV-cache into a dense, compacted tensor containing only
retained blocks. This eliminates the need for predication in the attention
kernel — the loop iterates over only active blocks.

Architecture:
    1. Prefix sum on boolean block mask → cumulative index array
    2. Gather active block indices
    3. Gather active key/value blocks into compacted tensors
    4. Return compacted tensors + indirection table

Design decision: We use the "pad to max" strategy for XLA compatibility.
The output tensors are always shaped [num_blocks, block_size, heads, dim],
but only the first num_active blocks contain real data. The rest are zeros.
This avoids dynamic shapes while letting the attention kernel loop over
[0, num_active) instead of [0, num_blocks).

See docs/xla_pass_design.md §2 for the compiler-level version of this.
"""

import jax
import jax.numpy as jnp
from functools import partial


def stream_compact(
    keys: jax.Array,
    values: jax.Array,
    block_mask: jax.Array,
    block_size: int = 512,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compact KV-cache blocks, retaining only active (non-evicted) blocks.
    
    Takes the full KV-cache and a boolean block mask, returns compacted
    tensors containing only the retained blocks in a contiguous layout.
    
    Args:
        keys: Key tensor of shape (seq_len, num_heads, head_dim).
        values: Value tensor of shape (seq_len, num_heads, head_dim).
        block_mask: Boolean tensor of shape (num_blocks, num_heads).
            A block is retained if mask is True for ANY head (logical OR).
        block_size: Token count per block (default: 512).
        
    Returns:
        Tuple of (compact_keys, compact_values, active_indices, num_active):
        - compact_keys: Shape (num_blocks, block_size, num_heads, head_dim).
            First num_active blocks are real data, rest are zeros.
        - compact_values: Same shape and layout as compact_keys.
        - active_indices: Shape (num_blocks,) int32 array. 
            active_indices[i] = original block index for compacted position i.
            Valid for i < num_active; undefined for i >= num_active.
        - num_active: Scalar int32. Number of retained blocks.
    """
    seq_len, num_heads, head_dim = keys.shape
    num_blocks = seq_len // block_size
    
    # Reshape into blocks: (num_blocks, block_size, num_heads, head_dim)
    keys_blocked = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    values_blocked = values.reshape(num_blocks, block_size, num_heads, head_dim)
    
    # Reduce mask across heads: retain block if ANY head says retain
    # block_mask: (num_blocks, num_heads) → (num_blocks,)
    block_active = jnp.any(block_mask, axis=-1)  # (num_blocks,) boolean
    
    # --- Stream Compaction via Prefix Sum ---
    # 
    # Convert boolean mask to int for prefix sum:
    #   active = [1, 0, 1, 1, 0, 1, 0, 0]
    #   prefix = [1, 1, 2, 3, 3, 4, 4, 4]  (inclusive prefix sum)
    # 
    # Active indices are found by sorting: blocks with mask=True
    # get low indices, blocks with mask=False get high indices.
    
    active_int = block_active.astype(jnp.int32)  # (num_blocks,)
    num_active = jnp.sum(active_int)  # scalar
    
    # Use argsort on the negated mask to push active blocks to the front.
    # -active_int: active blocks get -1 (sorts first), inactive get 0 (sorts last).
    # jax.lax.sort is stable, so relative order within active blocks is preserved.
    sort_order = jnp.argsort(-active_int, stable=True)  # (num_blocks,)
    
    # sort_order[0:num_active] are the original indices of active blocks, in order.
    # sort_order[num_active:] are the original indices of inactive blocks.
    active_indices = sort_order  # (num_blocks,) — valid positions [0, num_active)
    
    # Gather active blocks into compacted layout
    compact_keys = keys_blocked[active_indices]    # (num_blocks, block_size, num_heads, head_dim)
    compact_values = values_blocked[active_indices]  # same
    
    # Zero out inactive positions to prevent data leakage
    # Create a mask: (num_blocks,) where position i is True if i < num_active
    position_mask = jnp.arange(num_blocks) < num_active  # (num_blocks,)
    position_mask_4d = position_mask[:, None, None, None]  # (num_blocks, 1, 1, 1)
    
    compact_keys = jnp.where(position_mask_4d, compact_keys, jnp.zeros_like(compact_keys))
    compact_values = jnp.where(position_mask_4d, compact_values, jnp.zeros_like(compact_values))
    
    return compact_keys, compact_values, active_indices, num_active


def stream_decompact(
    compact_output: jax.Array,
    active_indices: jax.Array,
    num_active: jax.Array,
    num_blocks_original: int,
    block_size: int = 512,
) -> jax.Array:
    """Reverse the compaction: scatter compacted blocks back to original positions.
    
    Useful for verification (compact → decompact should recover the original
    masked tensor).
    
    Args:
        compact_output: Shape (num_blocks, block_size, num_heads, head_dim).
        active_indices: Shape (num_blocks,) from stream_compact.
        num_active: Scalar from stream_compact.
        num_blocks_original: Original number of blocks before compaction.
        block_size: Token count per block.
        
    Returns:
        Scattered tensor of shape (num_blocks_original, block_size, num_heads, head_dim)
        with active blocks placed at their original positions, inactive blocks zeroed.
    """
    _, bs, nh, hd = compact_output.shape
    
    # Initialize output with zeros
    output = jnp.zeros((num_blocks_original, bs, nh, hd), dtype=compact_output.dtype)
    
    # Scatter: for each compacted position i < num_active,
    # place compact_output[i] at position active_indices[i]
    # Use dynamic_update_slice or scatter
    position_mask = jnp.arange(num_blocks_original) < num_active
    
    # active_indices tells us WHERE each compacted block came from
    # We need to scatter back: output[active_indices[i]] = compact_output[i]
    output = output.at[active_indices].set(compact_output)
    
    # Zero out positions beyond num_active (they got scattered to wrong places)
    # Actually, we need to be more careful. active_indices has valid entries
    # only for [0, num_active). For [num_active, num_blocks), the indices
    # are for inactive blocks — we need to zero those back out.
    inactive_indices = active_indices[num_active:]
    # But since we can't use dynamic slicing easily, we just re-zero
    # using the original block_active mask pattern:
    # Simpler approach: set output[active_indices[i]] for i >= num_active to zero
    # The cleanest way: just mask the scattered output
    is_active = jnp.zeros(num_blocks_original, dtype=jnp.bool_)
    is_active = is_active.at[active_indices].set(position_mask)
    
    output = jnp.where(is_active[:, None, None, None], output, jnp.zeros_like(output))
    
    return output


@partial(jax.jit, static_argnames=['block_size'])
def compact_and_attend(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    block_mask: jax.Array,
    block_size: int = 512,
) -> tuple[jax.Array, dict]:
    """Full compaction + attention pipeline (JIT-compiled).
    
    This is the user-space compaction path (Phase C). It:
    1. Compacts the KV-cache using stream_compact
    2. Runs attention over only the active blocks
    3. Returns the output + metadata
    
    Args:
        q: Query tensor (seq_len_q, num_heads, head_dim).
        keys: Key tensor (seq_len_k, num_heads, head_dim).
        values: Value tensor (seq_len_k, num_heads, head_dim).
        block_mask: Boolean mask (num_blocks, num_heads).
        block_size: Tokens per block.
        
    Returns:
        Tuple of (output, metadata):
        - output: Attention output (seq_len_q, num_heads, head_dim).
        - metadata: Dict with num_active, num_blocks, eviction_rate.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    num_blocks = seq_len_k // block_size
    
    # Step 1: Compact
    compact_keys, compact_values, active_indices, num_active = stream_compact(
        keys, values, block_mask, block_size
    )
    
    # Step 2: Flatten compact tensors back to (seq_len, heads, dim) for attention
    # The compact tensors are (num_blocks, block_size, heads, dim)
    # We need (num_blocks * block_size, heads, dim) but only first
    # num_active * block_size tokens are real
    compact_keys_flat = compact_keys.reshape(num_blocks * block_size, num_heads, head_dim)
    compact_values_flat = compact_values.reshape(num_blocks * block_size, num_heads, head_dim)
    
    # Step 3: Create an all-True mask for the compact tensor
    # (the compaction already filtered out inactive blocks)
    compact_mask = jnp.arange(num_blocks) < num_active  # (num_blocks,)
    compact_mask_heads = jnp.broadcast_to(
        compact_mask[:, None], (num_blocks, num_heads)
    )  # (num_blocks, num_heads)
    
    # Step 4: Run standard sparse attention on the compact tensor
    # Import here to avoid circular imports
    from orthocache.sparse_attention import compile_pallas_sparse_attention
    
    output = compile_pallas_sparse_attention(
        q, compact_keys_flat, compact_values_flat,
        compact_mask_heads, block_size
    )
    
    # Metadata
    eviction_rate = 1.0 - (num_active / num_blocks)
    
    return output, {
        'num_active': num_active,
        'num_blocks': num_blocks,
        'eviction_rate': eviction_rate,
    }
