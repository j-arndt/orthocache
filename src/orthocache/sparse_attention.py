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
    """Pallas kernel function that executes block-sparse attention.
    
    This kernel is designed to be compiled via jax.experimental.pallas on TPU.
    
    Uses jnp.where() for conditional execution instead of Python `if` branches,
    which cannot be traced by JAX and would fail XLA compilation. The mask scalar
    gates the output: inactive blocks produce zero contribution without requiring
    Python-level control flow over traced values.
    """
    # Load query, keys, values, and mask
    q = q_ref[...]
    k = k_ref[...]
    v = v_ref[...]
    mask = mask_ref[...]
    
    # Compute dot product attention
    # scale factor uses float cast to ensure bfloat16 compatibility
    scale = jnp.sqrt(jnp.float32(q.shape[-1]))
    logits = jnp.matmul(q, k.T) / scale
    weights = jax.nn.softmax(logits, axis=-1)
    attn_out = jnp.matmul(weights, v)
    
    # Gate the output by the block mask: evicted blocks produce zeros.
    # jnp.where is traceable and compiles cleanly through Mosaic LLO.
    # The mask is a scalar boolean loaded from SMEM via PrefetchScalarGridSpec.
    mask_scalar = mask.reshape(())  # ensure scalar shape for broadcasting
    out_ref[...] = jnp.where(mask_scalar, attn_out, jnp.zeros_like(attn_out))

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
        grid=(num_blocks, num_heads),
        in_specs=[
            pl.BlockSpec(lambda b, h: (0, h, 0), (seq_len_q, 1, head_dim)),
            pl.BlockSpec(lambda b, h: (b * block_size, h, 0), (block_size, 1, head_dim)),
            pl.BlockSpec(lambda b, h: (b * block_size, h, 0), (block_size, 1, head_dim)),
            pl.BlockSpec(lambda b, h: (b, h), (1, 1)),
        ],
        out_specs=pl.BlockSpec(lambda b, h: (0, h, 0), (seq_len_q, 1, head_dim)),
    )(q, keys, values, block_mask)
    
    return out
