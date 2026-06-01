"""OrthoCache: Hardware-Native Spectral Energy Thresholding Governor for TPU KV-Cache Optimization."""

__version__ = "0.1.0"

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


