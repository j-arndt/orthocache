import jax
from jax.sharding import NamedSharding, PartitionSpec as P
from jax.experimental.custom_partitioning import custom_partitioning
from orthocache.compaction import compact_and_attend

def _supported_sharding(sharding, shape):
    """Preserve sharding along the first dimension (e.g. batch or heads)."""
    if not hasattr(sharding, "spec") or not hasattr(sharding, "mesh"):
        return sharding
    
    rank = len(shape.shape)
    if rank == 0:
        return sharding
        
    max_shared_dims = min(len(sharding.spec), rank)
    names = tuple(sharding.spec[:max_shared_dims]) + tuple(None for _ in range(rank - max_shared_dims))
    return NamedSharding(sharding.mesh, P(*names))

def _partition(mesh, arg_shapes, result_shape, block_size=512):
    q_shape = arg_shapes[0]
    out_sharding = _supported_sharding(q_shape.sharding, q_shape)
    
    def lower_fn(q, k, v, mask):
        out, _ = compact_and_attend(q, k, v, mask, block_size=block_size)
        return out
        
    return mesh, lower_fn, out_sharding, (out_sharding,)

def _infer_sharding_from_operands(mesh, arg_shapes, result_shape, block_size=512):
    q_shape = arg_shapes[0]
    return _supported_sharding(q_shape.sharding, q_shape)

def _orthocache_attention_impl(q, keys, values, block_mask, block_size):
    out, _ = compact_and_attend(q, keys, values, block_mask, block_size=block_size)
    return out

# Create the custom_partitioning instance with static_argnums=(4,) for block_size
orthocache_attention_partitioned = custom_partitioning(_orthocache_attention_impl, static_argnums=(4,))

orthocache_attention_partitioned.def_partition(
    infer_sharding_from_operands=_infer_sharding_from_operands,
    partition=_partition,
    sharding_rule='q h d, k h d, k h d, b h -> q h d'
)
