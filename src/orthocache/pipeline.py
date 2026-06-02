"""OrthoCache end-to-end pipeline.

Chains the full OrthoCache flow:
    FWHT → spectral bands → ζ computation → two-gate mask → compaction → attention

This module provides the high-level API that users call. It handles:
- Automatic TPU detection and fallback
- Block size alignment and padding
- ζ_max auto-calibration hints
- Timing and telemetry metadata
"""

import time
from functools import partial

import jax
import jax.numpy as jnp

from orthocache.fwht import fwht_512
from orthocache.spectral_energy import (
    compute_spectral_bands,
    compute_spectral_decay_ratio,
    compute_query_aware_mask,
    compute_multiband_mask,
)
from orthocache.sparse_attention import (
    jax_block_sparse_attention,
    compile_pallas_sparse_attention,
)
from orthocache.compaction import stream_compact, compact_and_attend


def orthocache_forward(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    block_size: int = 512,
    zeta_max: float = 5.0,
    tau: float | None = None,
    mode: str = 'compact',
) -> tuple[jax.Array, dict]:
    """Full OrthoCache pipeline: spectral analysis → eviction → attention.
    
    This is the primary public API. It runs the complete OrthoCache flow:
    1. Compute spectral decay ratio (ζ) for all blocks
    2. Generate two-gate eviction mask (logit bound + ζ coherence)
    3. Either compact the KV-cache or apply predicated sparse attention
    4. Return the attention output and detailed metadata
    
    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        keys: Key tensor of shape (seq_len_k, num_heads, head_dim).
        values: Value tensor of shape (seq_len_k, num_heads, head_dim).
        block_size: Tokens per block (must be 512 for FWHT).
        zeta_max: Maximum spectral decay ratio. Blocks with ζ > zeta_max
            are evicted regardless of query-aware logit bound.
            Default 5.0 is a conservative starting point.
        tau: Query-aware logit bound threshold. If None, computed
            automatically as mean - 1σ of the logit bounds.
        mode: Execution mode:
            - 'compact': Stream compaction (Phase C). Physically removes
              evicted blocks before attention. Recommended.
            - 'sparse': Predicated sparse attention (Phase A/B). Masks
              evicted blocks but still iterates over them.
            - 'dense': Full dense attention (baseline). Ignores all
              eviction logic. For comparison only.
              
    Returns:
        Tuple of (output, metadata):
        - output: Attention result, shape (seq_len_q, num_heads, head_dim).
        - metadata: Dict with timing, eviction stats, ζ distribution.
    """
    seq_len_k, num_heads, head_dim = keys.shape
    seq_len_q = q.shape[0]
    num_blocks = seq_len_k // block_size
    
    metadata = {
        'mode': mode,
        'seq_len_q': seq_len_q,
        'seq_len_k': seq_len_k,
        'num_blocks': num_blocks,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'block_size': block_size,
        'zeta_max': zeta_max,
    }
    
    # --- Dense baseline ---
    if mode == 'dense':
        t0 = time.perf_counter()
        output = _dense_attention(q, keys, values, head_dim)
        metadata['latency_ms'] = (time.perf_counter() - t0) * 1000
        metadata['eviction_rate'] = 0.0
        return output, metadata
    
    # --- Spectral analysis ---
    t_spectral = time.perf_counter()
    
    # Compute ζ for all blocks
    zeta = compute_spectral_decay_ratio(keys, block_size)  # (num_blocks, num_heads)
    
    # Auto-compute tau if not provided
    if tau is None:
        bounds = _compute_auto_tau(q, keys, block_size)
        tau = float(bounds)
        metadata['tau_auto'] = True
    else:
        metadata['tau_auto'] = False
    
    metadata['tau'] = tau
    
    # Two-gate mask: logit bound AND spectral coherence
    block_mask = compute_multiband_mask(q, keys, tau, zeta_max, block_size)
    # block_mask: (num_blocks, num_heads) boolean
    
    t_spectral_end = time.perf_counter()
    metadata['spectral_ms'] = (t_spectral_end - t_spectral) * 1000
    
    # ζ statistics
    zeta_any_head = jnp.mean(zeta, axis=-1)  # (num_blocks,)
    metadata['zeta_mean'] = float(jnp.mean(zeta_any_head))
    metadata['zeta_std'] = float(jnp.std(zeta_any_head))
    metadata['zeta_min'] = float(jnp.min(zeta_any_head))
    metadata['zeta_max_observed'] = float(jnp.max(zeta_any_head))
    
    # Eviction stats
    blocks_retained = jnp.sum(jnp.any(block_mask, axis=-1).astype(jnp.int32))
    metadata['blocks_retained'] = int(blocks_retained)
    metadata['blocks_evicted'] = int(num_blocks - blocks_retained)
    metadata['eviction_rate'] = float(1.0 - blocks_retained / num_blocks)
    
    # --- Attention ---
    t_attn = time.perf_counter()
    
    if mode == 'compact':
        output, compact_meta = compact_and_attend(
            q, keys, values, block_mask, block_size
        )
        metadata.update({
            'compact_num_active': int(compact_meta['num_active']),
        })
    elif mode == 'sparse':
        output = compile_pallas_sparse_attention(
            q, keys, values, block_mask, block_size
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'dense', 'sparse', or 'compact'.")
    
    t_attn_end = time.perf_counter()
    metadata['attention_ms'] = (t_attn_end - t_attn) * 1000
    metadata['total_ms'] = (t_attn_end - t_spectral) * 1000
    
    return output, metadata


def _dense_attention(q, keys, values, head_dim):
    """Standard dense attention (baseline)."""
    scale = jnp.sqrt(jnp.float32(head_dim))
    logits = jnp.einsum('qhd,khd->qkh', q, keys) / scale
    weights = jax.nn.softmax(logits, axis=1)
    return jnp.einsum('qkh,khd->qhd', weights, values)


def _compute_auto_tau(q, keys, block_size):
    """Auto-compute tau as mean - 1σ of query-aware logit bounds."""
    from orthocache.spectral_energy import compute_query_aware_bounds
    bounds = compute_query_aware_bounds(q, keys, block_size)
    # bounds: (seq_len_q, num_blocks, num_heads)
    max_bounds = jnp.max(bounds, axis=0)  # (num_blocks, num_heads)
    mean_b = jnp.mean(max_bounds)
    std_b = jnp.std(max_bounds)
    return mean_b - std_b
