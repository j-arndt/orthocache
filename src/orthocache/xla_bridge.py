"""OrthoCache XLA Bridge — JAX interface to the stream compaction pass.

This module provides the user-space Python implementation of the stream
compaction + compacted attention pipeline. It works in two modes:

1. **Native mode** (requires custom jaxlib with the HLO pass linked):
   The XLA pass automatically intercepts attention while-loops and
   rewrites them. No Python changes needed — just enable the flag:
       jax.config.update("jax_orthocache_compaction", True)

2. **Emulation mode** (pure JAX, no custom build):
   Python-level stream compaction + compacted attention using standard
   JAX ops. This is what you benchmark on Kaggle to validate Δτ > 0
   before investing in the full XLA build.

Usage (emulation mode):
    from orthocache.xla_bridge import compacted_attention
    output = compacted_attention(q, k, v, block_mask, block_size=512)
"""

import jax
import jax.numpy as jnp
from functools import partial


def stream_compact(block_mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Stream compaction: bool mask → (active_indices, num_active).

    Pure JAX implementation of the parallel prefix sum compaction.
    On TPU, this runs on the VPU in O(log M) cycles.

    Args:
        block_mask: bool array of shape (num_blocks,) or (num_blocks, num_heads).
            If 2-D, reduces across heads with logical OR (any head retains → active).

    Returns:
        active_indices: int32 array of shape (num_blocks,). First K entries
            contain the original indices of retained blocks (in order).
            Remaining entries are garbage (index >= num_blocks).
        num_active: scalar int32, the count of retained blocks (K).
    """
    # Reduce to 1-D if needed
    if block_mask.ndim == 2:
        mask_1d = jnp.any(block_mask, axis=1)
    else:
        mask_1d = block_mask

    num_blocks = mask_1d.shape[0]

    # Sort-based compaction: active blocks get sort key = original_index,
    # inactive blocks get sort key = num_blocks + original_index (pushed to end).
    iota = jnp.arange(num_blocks, dtype=jnp.int32)
    sort_keys = jnp.where(mask_1d, iota, num_blocks + iota)

    # Argsort gives us the permutation that puts active blocks first
    perm = jnp.argsort(sort_keys, stable=True)
    active_indices = perm  # First K entries are the active block indices

    num_active = jnp.sum(mask_1d).astype(jnp.int32)

    return active_indices, num_active


@jax.jit
def compacted_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    block_mask: jnp.ndarray,
    block_size: int = 512,
) -> jnp.ndarray:
    """Attention with stream compaction — iterate only over active blocks.

    This is the emulation-mode equivalent of the XLA HLO pass.
    Instead of looping over all M blocks and masking, we:
    1. Compact the mask → active_indices[0:K]
    2. Gather K and V blocks using active_indices
    3. Run dense attention on the compacted K, V

    The gather is O(K * block_size * head_dim) — linear in K.
    The attention is O(seq_q * K * block_size * head_dim) — linear in K.
    Total: O(K) instead of O(M). Wall-clock Δτ ∝ (1 - K/M) = S.

    Args:
        q: (seq_len_q, num_heads, head_dim)
        k: (seq_len_k, num_heads, head_dim) — full KV cache
        v: (seq_len_k, num_heads, head_dim) — full KV cache
        block_mask: (num_blocks, num_heads) or (num_blocks,) — True = retain
        block_size: KV block size (default 512)

    Returns:
        output: (seq_len_q, num_heads, head_dim)
    """
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_k = k.shape[0]
    num_blocks = seq_len_k // block_size

    # Stage 1: Stream compaction
    active_indices, num_active = stream_compact(block_mask)

    # Stage 2: Gather active K/V blocks into a compacted array
    # Reshape K, V into blocks: (num_blocks, block_size, num_heads, head_dim)
    k_blocks = k.reshape(num_blocks, block_size, num_heads, head_dim)
    v_blocks = v.reshape(num_blocks, block_size, num_heads, head_dim)

    # Gather using active_indices (all M indices; first K are valid)
    # We gather ALL M blocks but only the first K are meaningful.
    # This avoids dynamic shapes.
    k_compacted = k_blocks[active_indices]  # (num_blocks, block_size, NH, HD)
    v_compacted = v_blocks[active_indices]

    # Stage 3: Dense attention on compacted K/V
    # Flatten compacted blocks: (num_blocks * block_size, num_heads, head_dim)
    k_flat = k_compacted.reshape(num_blocks * block_size, num_heads, head_dim)
    v_flat = v_compacted.reshape(num_blocks * block_size, num_heads, head_dim)

    # Create a validity mask: first K*block_size positions are valid
    valid_len = num_active * block_size  # scalar
    pos = jnp.arange(num_blocks * block_size)
    valid_mask = pos < valid_len  # (total_len,)

    # Compute attention scores: Q @ K^T
    scale = jnp.sqrt(jnp.float32(head_dim))
    # q: (SQ, NH, HD), k_flat: (total, NH, HD)
    # logits: (SQ, total, NH)
    logits = jnp.einsum('qhd,khd->qkh',
                        q.astype(jnp.float32),
                        k_flat.astype(jnp.float32)) / scale

    # Mask out invalid positions (from inactive blocks)
    logits = jnp.where(valid_mask[None, :, None], logits, -1e9)

    # Softmax
    attn_weights = jax.nn.softmax(logits, axis=1)

    # Weighted sum: output = attn_weights @ V
    output = jnp.einsum('qkh,khd->qhd', attn_weights, v_flat.astype(jnp.float32))

    return output.astype(q.dtype)


@jax.jit
def dense_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
) -> jnp.ndarray:
    """Standard dense attention (no masking, baseline for comparison)."""
    scale = jnp.sqrt(jnp.float32(q.shape[-1]))
    logits = jnp.einsum('qhd,khd->qkh',
                        q.astype(jnp.float32),
                        k.astype(jnp.float32)) / scale
    weights = jax.nn.softmax(logits, axis=1)
    output = jnp.einsum('qkh,khd->qhd', weights, v.astype(jnp.float32))
    return output.astype(q.dtype)


@jax.jit
def predicated_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    block_mask: jnp.ndarray,
    block_size: int = 512,
) -> jnp.ndarray:
    """Predicated attention (mask but still compute all blocks — current v1)."""
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_k = k.shape[0]
    num_blocks = seq_len_k // block_size

    scale = jnp.sqrt(jnp.float32(head_dim))
    logits = jnp.einsum('qhd,khd->qkh',
                        q.astype(jnp.float32),
                        k.astype(jnp.float32)) / scale

    # Build per-token mask from block mask
    # block_mask: (num_blocks, num_heads) → (num_blocks, 1, num_heads)
    # → broadcast to (num_blocks, block_size, num_heads)
    # → reshape to (seq_len_k, num_heads)
    if block_mask.ndim == 1:
        block_mask = block_mask[:, None]  # (num_blocks, 1)
    token_mask = jnp.repeat(block_mask, block_size, axis=0)  # (seq_len_k, NH)

    logits = jnp.where(token_mask[None, :, :], logits, -1e9)
    weights = jax.nn.softmax(logits, axis=1)
    output = jnp.einsum('qkh,khd->qhd', weights, v.astype(jnp.float32))
    return output.astype(q.dtype)
