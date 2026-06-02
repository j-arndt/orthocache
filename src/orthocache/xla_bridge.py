import jax
import jax.numpy as jnp
from jax.lib import xla_client
from jax.interpreters import mlir
from jax.interpreters import xla

# Note: In a full deployment, you would compile a dynamic library (.so)
# containing the C++ XLA CustomCall target and load it here using:
# xla_client.register_custom_call_target(
#     "orthocache_stream_compact", 
#     custom_call_c_pointer,
#     platform="tpu"
# )

def orthocache_compact_call(block_mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Python bridge to the XLA CustomCall that performs stream compaction.
    
    Args:
        block_mask: A boolean array of shape (num_blocks,) indicating which
            blocks are retained.
            
    Returns:
        active_indices: Int32 array of shape (num_blocks,) containing the
            original indices of the retained blocks.
        num_active: Scalar Int32 indicating how many blocks were retained.
    """
    num_blocks = block_mask.shape[0]
    
    # We define the expected output shapes from the CustomCall
    out_shapes = [
        jax.ShapeDtypeStruct((num_blocks,), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.int32)
    ]
    
    # Currently emits a standard JAX fallback since the TPU CustomCall
    # is not linked in this pure-Python environment.
    # In production, this emits: jax.interpreters.mlir.custom_call(...)
    
    # Python fallback behavior (used when CustomCall is not registered):
    active_indices = jnp.where(block_mask, jnp.arange(num_blocks), num_blocks)
    active_indices = jnp.sort(active_indices)
    num_active = jnp.sum(block_mask).astype(jnp.int32)
    
    return active_indices, num_active

# Register the primitive for tracing (standard JAX extension boilerplate)
# orthocache_compact_p = jax.core.Primitive("orthocache_compact")
# orthocache_compact_p.multiple_results = True
