"""OrthoCache: Hardware-Native Spectral Energy Thresholding Governor for TPU KV-Cache Optimization."""

__version__ = "0.3.0"

from orthocache.fwht import fwht_512
from orthocache.spectral_energy import (
    compute_block_energy_jax,
    generate_threshold_mask,
    compute_query_aware_bounds,
    compute_query_aware_mask,
    compute_spectral_bands,
    compute_spectral_decay_ratio,
    compute_multiband_mask,
)
from orthocache.sparse_attention import jax_block_sparse_attention, compile_pallas_sparse_attention
from orthocache.compaction import stream_compact, stream_decompact, compact_and_attend
from orthocache.partitioning import orthocache_attention_partitioned
from orthocache.pipeline import orthocache_forward
from orthocache.indirect_attention import indirect_attention, indirect_attention_fori
from orthocache.adaptive_attention import adaptive_sparse_attention
from orthocache.alltoallv import alltoallv_kv_exchange
from orthocache.distributed_attention import distributed_orthocache_attention
from orthocache.ici_bandwidth_model import ici_bytes_per_step, ici_bandwidth_table, model_configs

