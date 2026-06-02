# XLA HLO Loop-Reindexing Pass: From Predication to Compaction

> **Status: Engineering Design Specification**
> This document specifies the XLA compiler pass that transforms OrthoCache's memory savings into wall-clock compute savings. It bridges the gap between the current prototype (latency parity, 1.00× speedup) and the target deployment (15–25% cluster throughput gain).

---

## 1. The Predication Problem

On TPU v5e-8, our Gate 5 benchmarks on Gemma 4 31B measured **latency parity** (1.00× speedup) at 50% block eviction across all sequence lengths from 4K to 64K tokens. The sparse kernel achieves full memory savings — 50% fewer KV cache entries stored — but zero compute savings.

**Root cause: predication.** Standard XLA evaluates the block mask, but instead of skipping the masked work, it forces the Matrix Multiplication Unit (MXU) to execute the tile matmul cycle anyway and suppresses the write-back to memory. The hardware still spends the clock cycles; it is just multiplying by zero.

| Metric | Dense | Sparse (50% eviction) | Speedup |
|:-------|:-----:|:---------------------:|:-------:|
| 4K tokens | 2.381 ms ✓ | 2.393 ms ✓ | 0.99× |
| 8K tokens | 2.435 ms ✓ | 2.310 ms ✓ | 1.05× (noise) |
| 16K tokens | 2.999 ms ✓ | 3.000 ms ✓ | 1.00× |
| 32K tokens | 5.704 ms ✓ | 5.706 ms ✓ | 1.00× |
| 64K tokens | 11.005 ms ✓ | 11.017 ms ✓ | 1.00× |

**The fix:** shift the execution strategy from **predication** to **compaction**. An HLO module pass that intercepts the compiler's intermediate representation and rewrites the loop to iterate only over active blocks.

---

## 2. Three-Stage Pass Architecture

The pass overrides `HloModulePass::Run(HloModule* module)` and executes three transformations in sequence.

### 2.1 Stage 1: Stream Compaction Pre-Pass (Parallel Prefix Sum)

Immediately after the kernel evaluates ζ and generates the boolean mask vector, the pass injects a hardware-level **parallel prefix sum (stream compaction)** operator.

```
Input (Boolean Mask):        [ 1,  0,  0,  1,  1,  0,  1,  0 ]  (Size M = 8)

Parallel Prefix Sum:         [ 1,  1,  1,  2,  3,  3,  4,  4 ]

Output (Indirection Table):  [ 0,  3,  4,  6 ]                  (Active indices, Size K = 4)
```

The TPU's Vector Processing Unit (VPU) computes the prefix sum across 512-tile masks in $O(\log b)$ cycles — a handful of clock ticks. The output is a compressed **indirection lookup table** of length $K$, where $K = (1 - S) \cdot M$ is the dynamic count of active (non-evicted) blocks.

**Why prefix sum, not scatter:** A prefix sum is a deterministic, associative, work-efficient parallel primitive. It produces a monotonically increasing index map with no gaps, which is exactly the input format that the DMA controller expects for indirect addressing. No synchronization barriers, no atomic operations, no warp divergence.

### 2.2 Stage 2: Dynamic Coordinate Inversion (Loop Reindexing)

Normally, XLA lowers the self-attention `DotGeneral` operator into a static, sequential loop nest that steps linearly from `0` to `M-1` (the total number of sequence blocks). Every block is fetched, multiplied, and accumulated regardless of its mask value.

The pass intercepts this loop and performs **dynamic coordinate inversion**: it rewrites the loop boundary condition so the induction variable `i` no longer iterates over raw sequence blocks. It iterates strictly from `0` to `K-1` (the dynamic number of active blocks).

```cpp
// Architectural realization of HLO Loop-Reindexing
StatusOr<bool> XlaLoopReindexPass::Run(HloModule* module) {
  for (auto* computation : module->computations()) {
    for (auto* hlo : computation->instructions()) {
      // Pattern-match the Pallas attention custom call
      if (hlo->opcode() != HloOpcode::kCustomCall ||
          hlo->custom_call_target() != "pallas_attention") continue;

      // 1. Intercept the block mask operand
      HloInstruction* block_mask = hlo->mutable_operand(/* mask_operand_index */);

      // 2. Inject stream compaction node → generates indirection table
      HloInstruction* active_indices = computation->AddInstruction(
          HloInstruction::CreateCustomCall(
              ShapeUtil::MakeShape(S32, {MaxBlocks}),
              {block_mask},
              "tpu_prefix_sum"));

      // 3. Inject active count extraction
      HloInstruction* active_count = computation->AddInstruction(
          HloInstruction::CreateCustomCall(
              ShapeUtil::MakeShape(S32, {}),
              {block_mask},
              "tpu_popcount"));

      // 4. Mutate the attention kernel: replace direct indexing
      //    with indirect index lookups through the compacted table
      //    actual_block_idx = active_indices[i]
      XlaGraphRewriter::ApplyIndirectAddressing(hlo, active_indices, active_count);
    }
  }
  return true;
}
```

### 2.3 Stage 3: Indirect DMA Fetch Execution

Inside the reindexed loop, the kernel performs an **indirect Direct Memory Access (DMA) fetch**. Instead of loading the key block at physical sequence coordinate `i`, the compiled kernel executes:

$$\text{Target Block Coordinate} = \text{Indirection Table}[i]$$

$$K_{\text{active}} = \text{HBM}[\ \text{Indirection Table}[i] \times b \ : \ (\text{Indirection Table}[i] + 1) \times b\ ]$$

Because the loop terminates early at $K - 1$:
- The DMA engine never issues fetch requests for evicted blocks
- The MXU systolic array never receives evicted tile data
- The online softmax accumulator processes fewer iterations

**Wall-clock savings scale with eviction fraction:**

$$T_{\text{sparse}} \approx (1 - S) \cdot T_{\text{dense}} + C_{\text{prefix\_sum}}$$

where $C_{\text{prefix\_sum}} \ll T_{\text{dense}}$ (VPU prefix sum is $O(\log M)$ cycles vs. MXU matmul at $O(b \cdot d_k)$ cycles per block).

---

## 3. Multi-Host Collective Unlock: Irregular AllToAll

On a single chip, loop reindexing speeds up local compute. But when running a 70B+ parameter model tensor-sharded across multiple hosts via the Inter-Chip Interconnect (ICI) fabric, this pass enables the critical optimization: **irregular AllToAll collectives**.

### 3.1 Current Dense AllToAll

In tensor-parallel inference with sharding factor $P$, each attention step requires an `AllToAll` collective transferring:

$$\text{ICI}_{\text{dense}} = O\!\left(\frac{N \cdot d_k}{P}\right) \text{ bytes per chip, per attention step}$$

Every chip transmits its entire, flat, uncompressed KV cache allocation to every other chip over physical optical cables — regardless of how many blocks are actually attended to.

### 3.2 Compacted Irregular AllToAll (AllToAllv)

With the indirection table available on each chip, the collective is replaced with `AllToAllv` (variable-size shards). Each chip transmits only its $(1 - S)$ fraction of active blocks:

$$\text{ICI}_{\text{compacted}} = O\!\left(\frac{(1 - S) \cdot N \cdot d_k}{P}\right) \text{ bytes per chip}$$

**Quantitative example.** For a 70B model with $P = 8$ tensor-parallel shards, $N = 128\text{K}$ tokens, $d_k = 128$, and 80 attention layers:

$$\text{ICI}_{\text{dense}} = 80 \cdot \frac{131{,}072 \cdot 128}{8} \cdot 2 \text{ B} = 335.5 \text{ MB / decode step}$$

At $S = 0.50$:

$$\Delta\text{ICI} = 167.8 \text{ MB / step} \quad \Rightarrow \quad 167.8 \text{ GB over 1{,}000 decode steps}$$

This is not zeroing values post-computation — you are **physically preventing gigabytes of evicted formatting data from saturating the cluster's bisection bandwidth**.

### 3.3 Metadata Synchronization

The indirection table must be communicated alongside the data. This is a small overhead:

$$\text{Table size} = K \cdot 4 \text{ bytes (int32)} \leq M \cdot 4 \text{ bytes}$$

For $M = 256$ blocks (128K tokens / 512 block size), the table is 1 KB — negligible compared to the GB of KV cache traffic saved.

---

## 4. Interaction with the XLA Pipeline

### 4.1 Pass Location

```
┌─────────────────────────────────────────────────────────────┐
│  XLA Compilation Pipeline                                    │
│                                                              │
│  1. HLO construction (from JAX trace)                        │
│  2. Algebraic simplification                                 │
│  3. Layout assignment                                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ 4. OrthoCache Loop-Reindexing Pass                     │ │
│  │    a. Pattern-match attention DotGeneral/CustomCall     │ │
│  │    b. Inject prefix sum → indirection table             │ │
│  │    c. Rewrite loop bounds [0,M) → [0,K)                │ │
│  │    d. Replace direct DMA → indirect DMA                 │ │
│  │    e. Rewrite AllToAll → AllToAllv (multi-host)         │ │
│  └─────────────────────────────────────────────────────────┘ │
│  5. Buffer assignment                                        │
│  6. Scheduling                                               │
│  7. Code generation (Mosaic LLO for TPU)                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Compatibility Matrix

| Existing Optimization | Interaction | Notes |
|:---------------------|:-----------|:------|
| FlashAttention / SplashAttention | **Complementary** | OrthoCache determines *which* blocks; Flash determines *how* to compute attention over retained blocks |
| KV-Cache Quantization (TurboQuant) | **Composable** | Eviction reduces block *count*; quantization reduces per-block *size*. Multiplicative savings. |
| Sliding-Window Attention | **No conflict** | Sliding layers have window-bounded caches (1024 tokens ≤ 2 blocks). The pass naturally skips these — per our 31B measurements, all 50 sliding layers are unaffected. |
| MQA / GQA head sharing | **Compatible** | The indirection table is per-head or per-head-group. GQA with 4 global KV heads (as in Gemma 4 31B) means 4 independent tables. |

---

## 5. Updated Performance Model (with Real 31B Data)

### 5.1 Measured Inputs (✓)

From the Gemma 4 31B TPU v5e-8 run (2026-06-02):

| Parameter | Value | Source |
|:----------|:------|:-------|
| Block sparsity at 10% eviction target | 9.4% actual ✓ | Gate 4 |
| Block sparsity at 30% eviction target | 28.1% actual ✓ | Gate 4 |
| Block sparsity at 50% eviction target | 50.0% actual ✓ | Gate 4 |
| Block sparsity at 70% eviction target | 68.8% actual ✓ | Gate 4 |
| TV distance at 50% eviction | 0.500 ✓ | Gate 4 |
| Reconstruction error at 50% eviction | 1.84% ✓ | Gate 4 |
| Latency overhead of masking approach | 0% (parity) ✓ | Gate 5 |
| ζ separability ratio | 4.04 × 10¹² ✓ | Gate 6 |

### 5.2 Projected Compute Savings (⊘)

With stream compaction eliminating all evicted block computation:

| Eviction Rate | Compute Reduction | Projected Latency Factor |
|:------------:|:-----------------:|:------------------------:|
| 9.4% | ~9% fewer MXU cycles ⊘ | 0.91× ⊘ |
| 28.1% | ~28% fewer MXU cycles ⊘ | 0.72× ⊘ |
| 50.0% | ~50% fewer MXU cycles ⊘ | 0.50× ⊘ |
| 68.8% | ~69% fewer MXU cycles ⊘ | 0.31× ⊘ |

*Note: These are upper-bound projections. Actual speedup depends on the fraction of total inference time spent in KV-cache-bound attention (varies by model size and context length).*

---

## 6. Fleet Economics with Loop Reindexing

Using the measured sparsity values from Gate 4 and projecting throughput gains under the compaction pass:

### 6.1 Parameterized Deployment Profiles

| Profile | Measured $S$ ✓ | Projected $\Delta\tau$ ⊘ | Annual OpEx Savings ⊘ | Annual CapEx Deferral ⊘ | **Total Fleet Value** ⊘ |
|:--------|:--------------:|:------------------------:|:---------------------:|:-----------------------:|:-----------------------:|
| **B (Standard Context)** | 28.1% ✓ | 15% ⊘ | $3,127,419 | $60,000,000 | **$63,127,419** |
| **C (Target Ceiling)** | 50.0% ✓ | 20% ⊘ | $5,564,790 | $80,000,000 | **$85,564,790** |
| **D (Aggressive Limit)** | 68.8% ✓ | 25% ⊘ | $7,657,151 | $100,000,000 | **$107,657,151** |

### 6.2 Derivation: Profile B ($S = 0.281$, $\Delta\tau = 0.15$)

**OpEx:**

$$\Delta\text{OpEx} = (200{,}000 \times 0.40) \times [0.281 \times 0.35 \times 0.550\text{ kW} \times 1.10 \times 8760\text{ hrs} \times \$0.075/\text{kWh}]$$
$$= 80{,}000 \times \$39.09/\text{chip-year} = \$3{,}127{,}419/\text{year}$$

**CapEx:**

$$\Delta\text{CapEx} = \$1{,}000{,}000{,}000 \times 0.40 \times 0.15 = \$60{,}000{,}000/\text{year}$$

### 6.3 Key Difference from Pre-Reindexing Model

The previous cost model (§4 of `cost_benefit_analysis.md`) used arbitrary sparsity targets ($S = 0.25, 0.50, 0.70$). The updated profiles use **empirically measured eviction rates** from the 31B benchmark:

- **Profile B** uses the actual 28.1% eviction rate measured at the 30% target
- **Profile C** uses the actual 50.0% eviction rate measured at the 50% target
- **Profile D** uses the actual 68.8% eviction rate measured at the 70% target

The $\Delta\tau$ values remain projected (⊘) — they require the stream compaction pass to be implemented and benchmarked on multi-host tensor-parallel workloads.

---

## 7. Safety Guarantees

### 7.1 Formal Error Bound (Unchanged)

The OrthoCache Truncation Bound (Lean 4 verified, zero `sorry` stubs) applies identically to the compacted execution:

$$\text{TV}(\alpha, \hat{\alpha}) \leq |S^c| \cdot \exp(\tau - z_{\max})$$

The compaction pass does not change *which* blocks are evicted — only *how* the eviction is executed. The mathematical guarantee transfers directly.

### 7.2 Reconstruction Error (Measured)

At the target Profile C operating point ($S = 0.50$):
- **TV distance:** 0.500 ✓
- **Reconstruction error:** 1.84% ✓
- **Bound violations:** 0 ✓

### 7.3 Fallback Behavior

- If pattern matching fails → no-op fallback (dense execution, no regression)
- If prefix sum overhead exceeds 1% of attention FLOPs → skip that layer
- Runtime eviction rate counter; if any layer exceeds safety limit → disable for that layer

---

## 8. Configuration Interface

### XLA Flags

```bash
--xla_orthocache_enable=true
--xla_orthocache_zeta_max=6.0          # Per-layer-type ζ threshold
--xla_orthocache_block_size=512
--xla_orthocache_max_eviction=0.7      # Safety ceiling
--xla_orthocache_compaction=true       # Enable stream compaction (vs. predication fallback)
--xla_orthocache_irregular_alltoall=true  # Enable AllToAllv for multi-host
```

### JAX User API

```python
jax.config.update("jax_orthocache_enable", True)
jax.config.update("jax_orthocache_zeta_max", 6.0)
jax.config.update("jax_orthocache_compaction", True)
```

---

## 9. Implementation Roadmap

| Phase | Work | Effort | Status |
|:------|:-----|:------:|:------:|
| **A** | User-space JAX/Pallas kernels | — | ✅ Complete |
| **B** | Benchmark validation (Gemma 4 31B, TPU v5e-8) | — | ✅ Complete |
| **C** | `jax.custom_partitioning` integration (user-space compaction) | 2–4 weeks | ⊘ Next |
| **D** | XLA HLO pass prototype (single-host, stream compaction) | 4–8 weeks | ⊘ |
| **E** | Multi-host AllToAllv integration | 4–8 weeks | ⊘ |
| **F** | Production hardening + large-scale benchmarking (70B+, 128K+) | 4–8 weeks | ⊘ |

**Phase C** is the natural next step: using `jax.custom_partitioning` to intercept the sharding of KV-cache tensors and implement the stream compaction in user space, validating the compaction strategy before committing to a compiler-level pass.

---

## 10. References

1. XLA HLO Pass Interface: [`xla/service/hlo_module_pass.h`](https://github.com/openxla/xla/blob/main/xla/service/hlo_module_pass.h)
2. AllToAll HLO Operation: [`xla/service/all_to_all_decomposer.h`](https://github.com/openxla/xla/blob/main/xla/service/all_to_all_decomposer.h)
3. Pallas TPU Kernel API: [JAX Pallas Documentation](https://jax.readthedocs.io/en/latest/pallas/index.html)
4. Custom Call Targets: [`xla/service/custom_call_target_registry.h`](https://github.com/openxla/xla/blob/main/xla/service/custom_call_target_registry.h)
5. OrthoCache Truncation Bound (Lean 4 verified): [`proofs/OrthoCacheMath/TruncationBound.lean`](../proofs/OrthoCacheMath/TruncationBound.lean)
6. OrthoCache Benchmark Results (Gemma 4 31B): [`benchmarks/results/orthocache_v4_31b_results.json`](../benchmarks/results/orthocache_v4_31b_results.json)
