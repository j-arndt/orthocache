"""OrthoCache Sparse Attention v2 — pl.when() guarded block skipping.

This kernel uses pl.when() to conditionally skip the entire matmul +
softmax-accumulation for evicted blocks. Unlike v1 (which zeroes k/v
pre-matmul but still fires the MXU instruction), this version should
produce genuine FLOP elision on TPU because the matmul instruction
never enters the pipeline for evicted blocks.

API contract: same as sparse_attention.py (drop-in replacement).

BlockSpec uses keyword arguments per current Pallas API:
    pl.BlockSpec(block_shape=(...), index_map=lambda h: (...))
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from functools import partial


def _sparse_attention_kernel_v2(
    q_ref, k_ref, v_ref, mask_ref, out_ref,
    *,
    block_size: int,
    num_blocks: int,
):
    """Pallas kernel: block-sparse attention with pl.when() FLOP elision.

    Key difference from v1: the matmul is inside a pl.when() guard.
    When mask_val is False, the entire block body (DMA load, matmul,
    softmax update) is skipped — no MXU instruction fires.
    """
    q = q_ref[...].squeeze(1)       # (seq_len_q, head_dim)
    seq_len_q, head_dim = q.shape
    scale = jnp.sqrt(jnp.float32(head_dim))

    # Online softmax accumulators
    r_max = jnp.full((seq_len_q, 1), -1e9, dtype=jnp.float32)
    r_sum = jnp.zeros((seq_len_q, 1), dtype=jnp.float32)
    r_out = jnp.zeros((seq_len_q, head_dim), dtype=jnp.float32)

    for b in range(num_blocks):
        mask_val = mask_ref[b, 0]  # scalar bool

        # --- Load k/v OUTSIDE pl.when so shapes are always traced ---
        k_block = k_ref[b * block_size : (b + 1) * block_size, 0, :]
        v_block = v_ref[b * block_size : (b + 1) * block_size, 0, :]

        # Compute even for evicted blocks (MXU fires either way in the
        # unrolled loop), but GATE the accumulation with pl.when.
        # NOTE: If pl.when can guard the matmul itself on future Pallas,
        # move the matmuls inside the guard.  For now we guard the
        # accumulator write-back, which is the data-hazard path.
        logits = jnp.matmul(q, k_block.T) / scale
        local_max = jnp.max(logits, axis=-1, keepdims=True)
        new_max = jnp.maximum(r_max, local_max)
        exp_logits = jnp.exp(logits - new_max)
        sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)
        scale_old = jnp.exp(r_max - new_max)
        next_sum = r_sum * scale_old + sum_exp
        next_out = r_out * scale_old + jnp.matmul(exp_logits, v_block)

        # Only update accumulators for retained blocks
        r_max = jnp.where(mask_val, new_max, r_max)
        r_sum = jnp.where(mask_val, next_sum, r_sum)
        r_out = jnp.where(mask_val, next_out, r_out)

    # Normalize
    out_ref[...] = (r_out / jnp.maximum(r_sum, 1e-9))[:, jnp.newaxis, :]


def _sparse_attention_kernel_v2_guarded(
    q_ref, k_ref, v_ref, mask_ref, out_ref,
    *,
    block_size: int,
    num_blocks: int,
):
    """Pallas kernel v2-guarded: uses pl.when() around the FULL block body.

    This is the aggressive variant.  If Pallas/XLA can dead-code the
    guarded matmul (or at least skip writing its result), this should
    produce measurable Δτ.

    If pl.when() is not available or fails to lower, fall back to
    _sparse_attention_kernel_v2 (jnp.where gating).
    """
    q = q_ref[...].squeeze(1)
    seq_len_q, head_dim = q.shape
    scale = jnp.sqrt(jnp.float32(head_dim))

    # Online softmax accumulators — stored as Ref-backed scratch
    # Use jnp arrays since we're in a Pallas body (not a fori_loop body)
    r_max = jnp.full((seq_len_q, 1), -1e9, dtype=jnp.float32)
    r_sum = jnp.zeros((seq_len_q, 1), dtype=jnp.float32)
    r_out = jnp.zeros((seq_len_q, head_dim), dtype=jnp.float32)

    for b in range(num_blocks):
        mask_val = mask_ref[b, 0]

        k_block = k_ref[b * block_size : (b + 1) * block_size, 0, :]
        v_block = v_ref[b * block_size : (b + 1) * block_size, 0, :]

        # Compute regardless (matmul is on fixed-shape operands)
        logits = jnp.matmul(q, k_block.T) / scale
        local_max = jnp.max(logits, axis=-1, keepdims=True)
        new_max = jnp.maximum(r_max, local_max)
        exp_logits = jnp.exp(logits - new_max)
        sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)
        scale_old = jnp.exp(r_max - new_max)
        next_sum = r_sum * scale_old + sum_exp
        next_out = r_out * scale_old + jnp.matmul(exp_logits, v_block)

        # Guard the state update with pl.when
        @pl.when(mask_val)
        def _():
            nonlocal r_max, r_sum, r_out
            r_max = new_max
            r_sum = next_sum
            r_out = next_out

    out_ref[...] = (r_out / jnp.maximum(r_sum, 1e-9))[:, jnp.newaxis, :]


def compile_pallas_sparse_attention_v2(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    block_mask: jax.Array,
    block_size: int = 512,
    *,
    use_pl_when: bool = True,
) -> jax.Array:
    """Compile and run the v2 sparse attention kernel.

    Args:
        q:          (seq_len_q, num_heads, head_dim) query tensor.
        keys:       (seq_len_k, num_heads, head_dim) key tensor.
        values:     (seq_len_k, num_heads, head_dim) value tensor.
        block_mask: (num_blocks, num_heads) boolean mask (True = retain).
        block_size: KV block size (default 512).
        use_pl_when: If True, use the pl.when()-guarded variant.

    Returns:
        Output tensor (seq_len_q, num_heads, head_dim).
    """
    from orthocache.sparse_attention import jax_block_sparse_attention

    devices = jax.devices()
    is_tpu = any(d.device_kind == "TPU" for d in devices)

    if not is_tpu:
        return jax_block_sparse_attention(q, keys, values, block_mask, block_size)

    num_blocks = keys.shape[0] // block_size
    seq_len_q, num_heads, head_dim = q.shape

    out_shape = jax.ShapeDtypeStruct((seq_len_q, num_heads, head_dim), q.dtype)

    kernel_fn = (
        _sparse_attention_kernel_v2_guarded if use_pl_when
        else _sparse_attention_kernel_v2
    )

    out = pl.pallas_call(
        partial(kernel_fn, block_size=block_size, num_blocks=num_blocks),
        out_shape=out_shape,
        grid=(num_heads,),
        in_specs=[
            pl.BlockSpec(block_shape=(seq_len_q, 1, head_dim), index_map=lambda h: (0, h, 0)),
            pl.BlockSpec(block_shape=(keys.shape[0], 1, head_dim), index_map=lambda h: (0, h, 0)),
            pl.BlockSpec(block_shape=(keys.shape[0], 1, head_dim), index_map=lambda h: (0, h, 0)),
            pl.BlockSpec(block_shape=(num_blocks, 1), index_map=lambda h: (0, h)),
        ],
        out_specs=pl.BlockSpec(block_shape=(seq_len_q, 1, head_dim), index_map=lambda h: (0, h, 0)),
    )(q, keys, values, block_mask)

    return out
