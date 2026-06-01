# XLA Pass Design: Proposed Future Compiler Integration

> **Status: Design Proposal** — This document describes a future integration path for OrthoCache's spectral energy thresholding into the XLA compiler as a native HLO optimization pass. The current implementation uses JAX/Pallas user-space kernels. Compiler integration would enable automatic, zero-configuration KV-cache optimization.

---

## 1. Motivation

OrthoCache currently operates as a **user-space JAX library**: the application code explicitly calls `compute_block_energy_jax()` to threshold KV-cache blocks and `jax_block_sparse_attention()` to skip evicted blocks. This works but requires manual integration into each model's attention implementation.

A native XLA pass would:
1. **Automatically intercept** AllToAll/AllGather collectives that transport KV-cache tensors across TPU chips
2. **Insert spectral energy computation** before the collective fires
3. **Skip DMA transfers** for low-energy blocks — reducing ICI bandwidth consumption
4. **Require zero model code changes** — the optimization is transparent to the user

---

## 2. Target: The AllToAll Bottleneck

In multi-host TPU inference with tensor parallelism, KV-cache tensors are redistributed across chips via `AllToAll` collectives at every decoder layer. For a 128K-token sequence with 128 heads × 128 head_dim in bfloat16:

```
Per-layer KV transfer = 2 × 128K × 128 × 128 × 2 bytes = 8.6 GB
Per-decode-step (80 layers) = 688 GB of ICI traffic
```

This is the dominant latency bottleneck in long-context inference. OrthoCache's spectral thresholding can identify which blocks carry negligible information and skip their transfer entirely.

---

## 3. HLO Pass Architecture

### 3.1 Pass Location in XLA Pipeline

The pass would execute **after algebraic simplification but before buffer assignment**, in the `HloModulePass` framework:

```
┌─────────────────────────────────────────────────────┐
│  XLA Compilation Pipeline                           │
│                                                     │
│  1. HLO construction (from JAX trace)               │
│  2. Algebraic simplification                        │
│  3. Layout assignment                               │
│  ┌───────────────────────────────────────┐          │
│  │ 4. OrthoCache Pass (proposed)         │          │
│  │    - Pattern-match AllToAll operands   │          │
│  │    - Insert FWHT energy computation    │          │
│  │    - Replace AllToAll with conditional │          │
│  └───────────────────────────────────────┘          │
│  5. Buffer assignment                               │
│  6. Scheduling                                      │
│  7. Code generation (Mosaic LLO for TPU)            │
└─────────────────────────────────────────────────────┘
```

### 3.2 HLO Pattern Matching

The pass identifies AllToAll operations whose operands have the shape signature of KV-cache tensors:

```
Pattern: HloAllToAll(operand)
  where operand.shape() matches:
    - rank >= 3
    - one dimension is a multiple of block_size (512)
    - dtype is bf16 or f32
    - operand is produced by a DotGeneral (attention QK^T pattern)
      OR operand feeds into a DotGeneral (attention AV pattern)
```

### 3.3 Transformation

For each matched AllToAll, the pass inserts a **spectral energy gate**:

```
BEFORE:
  %kv_cache = ...
  %redistributed = all-to-all(%kv_cache, ...)

AFTER:
  %kv_cache = ...
  %blocked = reshape(%kv_cache, [num_blocks, block_size, heads, head_dim])
  %energy = reduce-sum(multiply(%blocked, %blocked), dims=[1, 3])  // per-block energy
  %mask = compare-ge(%energy, %threshold)                          // boolean block mask
  %masked = select(%mask_broadcast, %blocked, zeros)               // zero out evicted blocks
  %filtered = reshape(%masked, original_shape)
  %redistributed = all-to-all(%filtered, ...)
```

The key insight: zeroed blocks compress to near-zero under ICI's internal encoding, effectively skipping their transfer even without explicit conditional DMA. For a more aggressive optimization, the pass could rewrite the AllToAll to use **irregular AllToAll** (variable-size shards per chip).

### 3.4 FWHT as HLO Custom Call

The FWHT computation itself would be registered as an XLA custom call target:

```cpp
// Registration (executed once at XLA initialization)
XLA_REGISTER_CUSTOM_CALL_TARGET("orthocache_fwht_512", OrthoCache_FWHT512);

// The custom call implements the 9-stage butterfly
// Input: [num_blocks, 512, head_dim] bf16
// Output: [num_blocks, 512, head_dim] bf16
void OrthoCache_FWHT512(void* out, void** ins, XlaCustomCallStatus* status) {
  // 9 statically-unrolled butterfly stages
  // Each stage: reshape → add/subtract pairs → reshape
  // Maps directly to TPU VPU vector add/subtract instructions
}
```

Alternatively, the FWHT can be expressed entirely in HLO using the same reshape-based butterfly pattern used in the Pallas kernel, avoiding the need for a custom call entirely.

---

## 4. Configuration Interface

The pass would be controlled via XLA flags:

```bash
# Enable the pass
--xla_orthocache_enable=true

# Energy threshold (blocks below this are evicted)
--xla_orthocache_threshold=0.01

# Block size for spectral analysis
--xla_orthocache_block_size=512

# Maximum eviction fraction (safety limit)
--xla_orthocache_max_eviction=0.5

# Layers to apply (empty = all decoder layers)
--xla_orthocache_target_layers=
```

From JAX user code, these would be exposed via `jax.config`:

```python
jax.config.update("jax_orthocache_enable", True)
jax.config.update("jax_orthocache_threshold", 0.01)
```

---

## 5. Safety Guarantees

### 5.1 Formal Error Bound

The pass enforces the **OrthoCache Truncation Bound** (Lean 4 verified):

$$\text{TV}(\alpha, \hat{\alpha}) \leq |S^c| \cdot \exp\!\left(\frac{\|q\|\sqrt{\epsilon}}{\sqrt{d_k}} - z_{\max}\right)$$

The `--xla_orthocache_threshold` parameter directly controls $\epsilon$, and the bound guarantees exponential decay in approximation error.

### 5.2 Fallback Behavior

- If the pass cannot pattern-match a valid KV-cache AllToAll, it **does nothing** (no-op fallback)
- If block energy computation would exceed a configurable FLOP budget (e.g., > 1% of the attention FLOPs), the pass skips that layer
- A runtime counter tracks actual eviction rates per layer; if any layer exceeds `--xla_orthocache_max_eviction`, the pass disables itself for that layer

### 5.3 Numerical Determinism

The FWHT is a **deterministic, associative** transform — no floating-point non-determinism from parallel reduction ordering. The block mask is computed from a simple threshold comparison. The overall pass preserves XLA's determinism guarantees.

---

## 6. Performance Model

Based on empirical measurements from Kaggle TPU v5e-8 (Gemma 4 E2B, 4096 tokens):

| Metric | Measured Value | Notes |
|---|---|---|
| FWHT kernel latency | 180.7 ms (512-block) | Prototype Pallas; XLA custom call would be ~10-100× faster |
| Block energy computation | 846.6 ms | Dominated by Python dispatch overhead |
| Dense attention | 1.20 ms median | Baseline |
| AllToAll bandwidth (estimated) | ~300 GB/s per link | TPU v5e ICI bandwidth |

**Expected compiler-level speedup:** With the energy computation fused into the HLO graph (no Python dispatch), the overhead drops to ~1-5 μs per layer — negligible compared to the 10-100 ms saved by skipping block DMA transfers in long-context scenarios.

### Target Operating Point

For a 128K-token sequence with 50% block eviction:
- **ICI bandwidth saved:** ~50% of KV-cache transfers = ~344 GB per decode step
- **FWHT overhead (compiler-fused):** < 0.1 ms per layer
- **Net time-to-first-token improvement:** 15-40% (bottleneck-dependent)

---

## 7. Interaction with Existing Optimizations

### 7.1 Compatibility with FlashAttention/SplashAttention

OrthoCache operates **before** FlashAttention — it determines which blocks to skip, while FlashAttention handles the efficient computation of attention over the retained blocks. They are complementary:

```
OrthoCache (which blocks?) → FlashAttention (compute attention efficiently)
```

### 7.2 Compatibility with KV-Cache Quantization (TurboQuant/TurboAngle)

OrthoCache performs **active eviction** while quantization performs **passive compression**. They compose naturally:

```
OrthoCache evicts low-energy blocks → TurboAngle compresses surviving blocks
```

The combination reduces both **bandwidth** (fewer blocks transferred) and **memory footprint** (remaining blocks quantized).

### 7.3 Compatibility with Sliding-Window Attention

As observed empirically with Gemma 4 E2B, sliding-window layers have single-block caches (512 tokens) that are never candidates for eviction. The pass would naturally skip these layers since they produce only 1 block, making energy thresholding meaningless.

---

## 8. Implementation Roadmap

| Phase | Work | Estimated Effort |
|---|---|---|
| **Phase A** | User-space JAX library (current) | ✅ Complete |
| **Phase B** | JAX custom_partitioning integration | 2-4 weeks |
| **Phase C** | XLA HLO pass prototype (single-host) | 4-8 weeks |
| **Phase D** | Multi-host AllToAll integration | 4-8 weeks |
| **Phase E** | Production hardening + benchmarking | 4-8 weeks |

**Phase B** is the natural next step: using `jax.custom_partitioning` to intercept the sharding of KV-cache tensors and insert the energy computation at the JAX level, without modifying XLA itself. This provides most of the benefit of a compiler pass while remaining in user space.

---

## 9. References

1. XLA HLO Pass Interface: [`xla/service/hlo_module_pass.h`](https://github.com/openxla/xla/blob/main/xla/service/hlo_module_pass.h)
2. AllToAll HLO Operation: [`xla/service/all_to_all_decomposer.h`](https://github.com/openxla/xla/blob/main/xla/service/all_to_all_decomposer.h)
3. Pallas TPU Kernel API: [JAX Pallas Documentation](https://jax.readthedocs.io/en/latest/pallas/index.html)
4. Custom Call Targets: [`xla/service/custom_call_target_registry.h`](https://github.com/openxla/xla/blob/main/xla/service/custom_call_target_registry.h)
5. Block-Sparse Attention on TPU: [HuggingFace Blog](https://huggingface.co/blog/rishiraj/block-sparse-attention-with-jaxpallas)
6. OrthoCache Truncation Bound (Lean 4 verified): [`proofs/OrthoCacheMath/TruncationBound.lean`](../proofs/OrthoCacheMath/TruncationBound.lean)
