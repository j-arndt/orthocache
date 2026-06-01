import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def jax_block_sparse_attention(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    block_mask: jax.Array,
    block_size: int = 512
) -> jax.Array:
    """Computes block-sparse attention using a precomputed boolean block mask.
    
    This function implements the mathematical behavior of the Pallas sparse attention
    kernel using native JAX operations.
    
    Args:
        q: Query tensor of shape (seq_len_q, num_heads, head_dim).
        keys: Key tensor of shape (seq_len_k, num_heads, head_dim).
        values: Value tensor of shape (seq_len_k, num_heads, head_dim).
        block_mask: Boolean tensor of shape (num_blocks_k, num_heads) indicating
                    which key-value blocks are retained.
        block_size: The size of each KV block (default: 512).
        
    Returns:
        The output tensor of shape (seq_len_q, num_heads, head_dim).
    """
    seq_len_q, num_heads, head_dim = q.shape
    seq_len_k = keys.shape[0]
    num_blocks = seq_len_k // block_size
    
    # Reshape keys and values into blocks: (num_blocks, block_size, num_heads, head_dim)
    keys_blocked = keys.reshape(num_blocks, block_size, num_heads, head_dim)
    values_blocked = values.reshape(num_blocks, block_size, num_heads, head_dim)
    
    # Compute attention scores for each block
    # q: (seq_len_q, num_heads, head_dim) -> (seq_len_q, 1, num_heads, head_dim)
    q_expanded = q[:, jnp.newaxis, :, :]
    
    # (num_blocks, seq_len_q, block_size, num_heads)
    # Dot product of query with keys in each block:
    logits = jnp.einsum("qihd,bkhd->bqkh", q_expanded, keys_blocked) / jnp.sqrt(head_dim)
    
    # Apply block mask to zero/evict the logits of block_mask == False
    # block_mask: (num_blocks, num_heads) -> (num_blocks, 1, 1, num_heads)
    mask_expanded = block_mask[:, jnp.newaxis, jnp.newaxis, :]
    
    # Mask out evicted blocks with a large negative value
    logits_masked = jnp.where(mask_expanded, logits, -1e9)
    
    # Softmax over all keys (flattening blocks and block_size):
    # Reshape logits to (seq_len_q, num_blocks * block_size, num_heads)
    # which is (seq_len_q, seq_len_k, num_heads)
    logits_flat = jnp.transpose(logits_masked, (1, 0, 2, 3)).reshape(seq_len_q, seq_len_k, num_heads)
    
    # Softmax over sequence dimension
    attn_weights = jax.nn.softmax(logits_flat, axis=1)
    
    # Compute weighted sum of values:
    # attn_weights: (seq_len_q, seq_len_k, num_heads)
    # values: (seq_len_k, num_heads, head_dim)
    output = jnp.einsum("qkh,khd->qhd", attn_weights, values)
    
    return output

# Pallas Kernel helper functions for TPU compilation
def pallas_sparse_attention_kernel(q_ref, k_ref, v_ref, mask_ref, out_ref, block_size):
    """Pallas kernel function that executes block-sparse attention with online softmax."""
    q = q_ref[...]  # Shape: (seq_len_q, 1, head_dim)
    q = q.squeeze(1)  # (seq_len_q, head_dim)
    
    seq_len_k = k_ref.shape[0]
    num_blocks = seq_len_k // block_size
    head_dim = q.shape[-1]
    scale = jnp.sqrt(jnp.float32(head_dim))
    
    seq_len_q = q.shape[0]
    
    # Initialize online softmax accumulators
    r_max = jnp.full((seq_len_q, 1), -1e9, dtype=jnp.float32)
    r_sum = jnp.zeros((seq_len_q, 1), dtype=jnp.float32)
    r_out = jnp.zeros((seq_len_q, head_dim), dtype=jnp.float32)
    
    # Loop over blocks
    for b in range(num_blocks):
        # Load mask value for block b, head 0 (since head dimension is blocked to size 1)
        mask_val = mask_ref[b, 0]  # boolean scalar
        
        # Load key and value blocks. Note that head index is 0 in local block reference
        k_block = k_ref[b * block_size : (b + 1) * block_size, 0, :]  # (block_size, head_dim)
        v_block = v_ref[b * block_size : (b + 1) * block_size, 0, :]  # (block_size, head_dim)
        
        # Compute dot product attention
        logits = jnp.matmul(q, k_block.T) / scale  # (seq_len_q, block_size)
        
        # Determine local max for this block
        local_max = jnp.max(logits, axis=-1, keepdims=True)
        new_max = jnp.maximum(r_max, local_max)
        
        # Softmax scaling terms
        exp_logits = jnp.exp(logits - new_max)
        sum_exp = jnp.sum(exp_logits, axis=-1, keepdims=True)
        
        scale_old = jnp.exp(r_max - new_max)
        
        # Next accumulators
        next_sum = r_sum * scale_old + sum_exp
        next_out = r_out * scale_old + jnp.matmul(exp_logits, v_block)
        
        # Update dynamically based on mask
        r_max = jnp.where(mask_val, new_max, r_max)
        r_sum = jnp.where(mask_val, next_sum, r_sum)
        r_out = jnp.where(mask_val, next_out, r_out)
        
    # Normalize final output
    final_out = r_out / jnp.maximum(r_sum, 1e-9)
    
    # Write to output (add head dimension back)
    out_ref[...] = final_out[:, jnp.newaxis, :]

def compile_pallas_sparse_attention(
    q: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    block_mask: jax.Array,
    block_size: int = 512
) -> jax.Array:
    """Wrapper function to invoke the Pallas kernel on TPU.
    
    Falls back to jax_block_sparse_attention on non-TPU platforms.
    """
    # Fallback to JAX block sparse attention on non-TPU backends (like CPU or GPU emulation)
    devices = jax.devices()
    is_tpu = any(d.device_kind == "TPU" for d in devices)
    
    if not is_tpu:
        return jax_block_sparse_attention(q, keys, values, block_mask, block_size)
    
    # TPU-specific compilation using pallas
    num_blocks = keys.shape[0] // block_size
    seq_len_q, num_heads, head_dim = q.shape
    
    out_shape = jax.ShapeDtypeStruct((seq_len_q, num_heads, head_dim), q.dtype)
    
    # Grid specification and Pallas call
    out = pl.pallas_call(
        lambda q_r, k_r, v_r, m_r, o_r: pallas_sparse_attention_kernel(
            q_r, k_r, v_r, m_r, o_r, block_size
        ),
        out_shape=out_shape,
        grid=(num_heads,),
        in_specs=[
            pl.BlockSpec(lambda h: (0, h, 0), (seq_len_q, 1, head_dim)),
            pl.BlockSpec(lambda h: (0, h, 0), (keys.shape[0], 1, head_dim)),
            pl.BlockSpec(lambda h: (0, h, 0), (keys.shape[0], 1, head_dim)),
            pl.BlockSpec(lambda h: (0, h), (num_blocks, 1)),
        ],
        out_specs=pl.BlockSpec(lambda h: (0, h, 0), (seq_len_q, 1, head_dim)),
    )(q, keys, values, block_mask)
    
    return out

