# Cost-Benefit Analysis: Macro-Economic Infrastructure Model for OrthoCache

This document establishes the formal macroeconomic infrastructure cost-benefit model for **OrthoCache**. It reverse-engineers Google's global accelerator fleet allocation using public Alphabet capital expenditure (CapEx) data, documented TPU thermal design parameters, and standard data center utility coefficients.

By expressing total savings as a function of the measured empirical sequence sparsity ($S$) and reclaimed compute throughput ($\Delta \tau$), this model provides a deterministic framework for validating the economic return of OrthoCache's compiler-level KV-cache truncation.

> **Epistemic Status.** Every numerical value in this document is marked with either ✓ (empirically measured on Gemma 4 E2B / TPU v5e-8) or ⊘ (parameterized projection requiring production validation). Fleet-scale constants ($N_{\text{chips}}$, $\phi_{\text{inf}}$) are engineering estimates derived from public Alphabet filings — not measured values.

---

## 1. Fleet Footprint & CapEx Baseline Estimation

To construct a defensible baseline for Google's top-tier global AI inference footprint, we reverse-engineer the fleet size via public Alphabet capital expenditure runway trajectories.

### 1.1 Capital Expenditure Input Variables

* **Alphabet Annual CapEx Run Rate:** Alphabet's quarterly CapEx runway is approximately **$12 Billion to $13 Billion**, yielding an annualized infrastructure expenditure of:
  $$\text{CapEx}_{\text{Alphabet}} \approx \$50,000,000,000 \text{ / year}$$
* **AI Compute Allocation:** Approximately **60%** of this capital flow is directed strictly toward AI data center construction, cooling infrastructure, optical switches, and custom silicon (TPU) fabrication:
  $$\text{CapEx}_{\text{AI}} \approx \$30,000,000,000 \text{ / year}$$
* **Amortized Accelerator Lifecycle Cost:** The fully burdened capital cost of a top-tier accelerator node (including custom silicon tape-outs, high-density HBM3e procurement, host system VM components, and optical Inter-Chip Interconnect routing gear) is modeled at:
  $$\text{Cost}_{\text{accelerator}} \approx \$15,000 \text{ per chip lot (3-year lifespan)}$$

### 1.2 Fleet Vector Calculations

Dividing the annualized compute CapEx allocation by the fully loaded node amortization indicates a total cumulative deployment capacity of roughly 500,000 active TPU slots across global data centers.

Accounting for training pod allocations, the specialized inference serving fleet footprint ($N_{\text{chips}}$) dedicated to operational text, code, and search pipelines is modeled as:

$$N_{\text{chips}} = 200,000 \text{ active inference accelerators}$$

We define the operational constraint variable:
$$\phi_{\text{inf}} = 0.40$$
as the conservative fraction of this global fleet actively processing distributed, long-context requests ($>128\text{K}$ tokens) where the memory wall and network-bound `AllToAll` collectives actively dominate the chip execution cycle.

---

## 2. Thermodynamic & OpEx Model

When a TPU v5p or TPU v6 (Trillium) cluster stalls waiting for cross-node network packets during long-context attention steps, the accelerators idle but continue to draw heavy static power (HBM3e interfaces, clock distribution networks, and optical transceivers).

### 2.1 Hardware Power Envelope Constants

* **TPU Active Thermal Design Power ($P_{\text{chip}}$):** Model a baseline of **550W** per accelerator slot under load.
* **Power Usage Effectiveness (PUE):** Google maintains an exceptional data center efficiency profile, anchored at an average of **1.10**.
* **Blended Industrial Energy Rate ($\text{Rate}_{\text{kWh}}$):** Modeled at a stable data center industrial cost of **$0.075 \text{ per kWh}$**.
* **Thermodynamic Attenuation Factor ($\gamma_{\text{net}}$):** Model the fraction of total chip power directly consumed by HBM3e interface clocking and physical ICI networking cards that is bypassed during block eviction:
  $$\gamma_{\text{net}} = 0.35$$

### 2.2 Reclaimed Power Equation

By executing our unrolled Fast Walsh-Hadamard Transform inside local registers and writing a block-sparse mask to SMEM, OrthoCache blocks high-power direct memory access (DMA) loops and optical transceiver activations for a measured block sparsity fraction ($S$).

The annual operational expenditure savings ($\Delta \text{OpEx}$) is modeled as:

$$\Delta \text{OpEx} = (N_{\text{chips}} \cdot \phi_{\text{inf}}) \times \left[ S \cdot \gamma_{\text{net}} \cdot P_{\text{chip}} \cdot \text{PUE} \times 8760 \text{ hrs} \times \text{Rate}_{\text{kWh}} \right]$$

**Unit verification.** $P_{\text{chip}} = 550\text{W} = 0.55\text{ kW}$. The inner bracket has units:

$$[\text{dimensionless}] \cdot [\text{dimensionless}] \cdot [\text{kW}] \cdot [\text{dimensionless}] \cdot [\text{hrs/yr}] \cdot [\$/\text{kWh}] = [\$/\text{chip-year}]$$

At $S = 1.0$ (theoretical maximum), this yields $\gamma_{\text{net}} \cdot 0.55 \cdot 1.10 \cdot 8760 \cdot 0.075 = \$139.12\text{/chip-year}$, representing the maximum reclaimable power cost per chip.

---

## 3. CapEx Avoidance & Throughput Acceleration Framework

### 3.1 Asymptotic Crossover Analysis

The economic case for OrthoCache depends on a crossover: the point at which the quadratic savings from block eviction exceed the linear overhead of the FWHT pre-pass. We formalize this as follows.

**Dense attention compute cost** per attention head, per decoding step:

$$C_{\text{dense}} = O(N^2 d_k)$$

where $N$ is the cached sequence length and $d_k$ is the head dimension.

**OrthoCache attention cost** decomposes into two terms:

$$C_{\text{ortho}} = \underbrace{O(N d_k)}_{\text{FWHT pre-pass}} + \underbrace{O\!\left((1-S)^2 N^2 d_k\right)}_{\text{attention over retained blocks}}$$

The first term is the FWHT energy computation and thresholding, which is linear in sequence length (one pass over $N$ keys, each of dimension $d_k$). The second term is the standard quadratic attention, but computed over only $(1-S)N$ retained tokens — yielding a $(1-S)^2$ factor on the quadratic cost.

**Break-even condition.** OrthoCache is profitable when $C_{\text{dense}} > C_{\text{ortho}}$:

$$N^2 d_k > N d_k + (1-S)^2 N^2 d_k + C_{\text{overhead}}$$

where $C_{\text{overhead}}$ captures fixed dispatch costs (Python/JAX function call overhead, mask materialization, buffer allocation). Rearranging:

$$\left[1 - (1-S)^2\right] N^2 d_k > N d_k + C_{\text{overhead}}$$

The quadratic savings factor expands as:

$$1 - (1-S)^2 = 1 - (1 - 2S + S^2) = S(2 - S)$$

Therefore:

$$S(2-S) \cdot N^2 d_k > N d_k + C_{\text{overhead}}$$

Dividing both sides by $d_k$:

$$S(2-S) \cdot N^2 > N + \frac{C_{\text{overhead}}}{d_k}$$

For large $N$, the $N$ term on the right is dominated by $N^2$ on the left, so:

$$N > \frac{1 + C_{\text{overhead}} / (N \cdot d_k)}{S(2-S)} \approx \frac{d_k + C_{\text{overhead}}/d_k}{S(2-S) \cdot d_k}$$

More precisely, solving the quadratic $S(2-S) N^2 - N - C_{\text{overhead}}/d_k > 0$:

$$N > \frac{1 + \sqrt{1 + 4 S(2-S) \cdot C_{\text{overhead}}/d_k}}{2 S(2-S)}$$

**At $S = 0.5$:** $S(2-S) = 0.5 \cdot 1.5 = 0.75$, giving:

$$N > \frac{1 + \sqrt{1 + 3 C_{\text{overhead}}/d_k}}{1.5}$$

**Empirical measurement (✓).** At $N = 4096$ with $S = 0.5$ on Gemma 4 E2B (TPU v5e-8), our Gate 5 profiling measured a **16× slowdown** (dense: 1.2 ms ✓, OrthoCache: 19.8 ms ✓). This is expected: at this scale, the dense MXU kernel completes in 1.2 ms — the FWHT pre-pass overhead and Python dispatch cost dominate. The fixed overhead $C_{\text{overhead}}$ is empirically $\approx 18.6\text{ ms}$ ✓ (measured), which swamps the quadratic savings at $N = 4096$.

**Crossover projection (⊘).** The crossover point occurs when $N$ is large enough that the quadratic savings $S(2-S) N^2 d_k$ exceed the linear pre-pass cost $N d_k + C_{\text{overhead}}$, **AND** when the model is tensor-parallel across chips (creating ICI traffic that eviction eliminates). With fused XLA kernels (eliminating Python dispatch, reducing $C_{\text{overhead}}$ by $\geq 100\times$), the crossover is projected to occur at $N \approx 32\text{K}$–$128\text{K}$ tokens ⊘. This requires implementation of the XLA pass described in `docs/xla_pass_design.md`.

### 3.2 ICI Bandwidth Reduction Model

At tensor-parallel sharding factor $P$ (the number of chips across which a single model's attention heads are distributed), each attention step requires an **AllToAll** collective communication transferring:

$$\text{ICI}_{\text{dense}} = O\!\left(\frac{N \cdot d_k}{P}\right) \text{ bytes per chip, per attention step}$$

This communication is required to reassemble the full attention output from sharded KV-cache fragments. With OrthoCache evicting a fraction $S$ of blocks before the attention computation, the evicted blocks are never loaded from HBM and never communicated over ICI:

$$\text{ICI}_{\text{ortho}} = O\!\left(\frac{(1-S) \cdot N \cdot d_k}{P}\right) \text{ bytes per chip}$$

The ICI bandwidth savings scale **linearly** with both $S$ and $N$:

$$\Delta \text{ICI} = S \cdot \frac{N \cdot d_k}{P} \text{ bytes per chip, per step}$$

**This is the primary economic lever.** On Gemma 4 E2B (5.1B parameters), the entire model fits on a single chip — **no ICI traffic exists**. The 16× slowdown measured in Gate 5 reflects a regime where the ICI savings channel is structurally absent. On production-scale models (70B+ parameters) with 8-way or 16-way tensor parallelism, ICI AllToAll is the dominant bottleneck — consuming up to 40% of wall-clock time at 128K+ token sequence lengths. OrthoCache's eviction directly reduces this traffic by a factor of $(1-S)$, and this reduction compounds across all attention layers and all decoding steps.

**Quantitative projection (⊘).** For a 70B-parameter model with $P = 8$ tensor-parallel shards, $N = 128\text{K}$ tokens, $d_k = 128$, and 80 attention layers:

$$\text{ICI}_{\text{dense}} = 80 \cdot \frac{131072 \cdot 128}{8} \cdot 2 \text{ bytes} = 80 \cdot 4.19\text{ MB} = 335.5\text{ MB per decoding step}$$

At $S = 0.50$: $\Delta\text{ICI} = 167.8\text{ MB per step}$ ⊘. Over 1000 decoding steps, this is $167.8\text{ GB}$ of ICI traffic eliminated per request — a substantial fraction of the TPU v5p ICI bisection bandwidth.

### 3.3 CapEx Deferral Model

The primary value proposition of OrthoCache to infrastructure leadership is **Capital Expenditure Deferral**. If OrthoCache short-circuits network stalls and reclaims an operational throughput speedup factor ($\Delta \tau$), Google can serve an increased volume of concurrent long-context requests on their *existing* physical chip allocation, eliminating the immediate capital requirement to purchase, cool, and install additional hardware lots.

Let the annualized CapEx cost of the active inference fleet be:

$$\text{CapEx}_{\text{annual}} = \frac{N_{\text{chips}} \times \$15,000}{3 \text{ years}} = \$1,000,000,000 \text{ / year}$$

The deferred capital infrastructure dividend ($\Delta \text{CapEx}$) is:

$$\Delta \text{CapEx} = \text{CapEx}_{\text{annual}} \cdot \phi_{\text{inf}} \cdot \Delta \tau$$

**Critical parameterization note.** $\Delta\tau$ is an open parameter to be validated on production-scale tensor-parallel workloads. Our current measurements on Gemma 4 E2B at 4K tokens yield $\Delta\tau < 0$ ✓ (the prototype imposes a 16× overhead, i.e., $\Delta\tau \approx -0.94$). The projected crossover to $\Delta\tau > 0$ requires:

1. **Sequence lengths $\geq 128\text{K}$ tokens** — where the quadratic attention cost dominates over the linear FWHT pre-pass (§3.1).
2. **Tensor-parallel model sharding** — where ICI bandwidth reduction (§3.2) provides the dominant savings channel.
3. **Fused XLA kernel integration** — eliminating Python dispatch overhead and intermediate buffer materialization (see `docs/xla_pass_design.md`).

Until all three conditions are met, $\Delta\tau$ remains a projected parameter, not a measured one.

---

## 4. OrthoCache Fleet Efficiency Matrix

### 4.1 Measured Empirical Results

**Table 1.** Accuracy measurements on Gemma 4 E2B, TPU v5e-8, $N = 4096$, global attention layer 4 (8 blocks, 1 KV head, 256-dim). All values empirically measured (✓) in Gate 4.

| Target Eviction | Actual Eviction (✓) | TV Distance (✓) | Recon Error (✓) | Bound Violations (✓) |
| :---: | :---: | :---: | :---: | :---: |
| 10% | 12.5% (1/8 blocks) | 0.1251 | 0.23% | **0** |
| 30% | 37.5% (3/8 blocks) | 0.3753 | 0.95% | **0** |
| 50% | 50.0% (4/8 blocks) | 0.5003 | 1.57% | **0** |
| 70% | 62.5% (5/8 blocks) | 0.6248 | 1.45% | **0** |

**Note on actual eviction granularity.** Eviction operates at block granularity ($b = 512$ tokens). With 8 blocks, the achievable eviction fractions are discrete multiples of 12.5%. Target rates are mapped to the nearest block count.

**Note on TV distance values.** The measured TV distances track the actual eviction fractions closely (e.g., TV $\approx 0.125$ at 12.5% eviction). This follows directly from the proof: $\text{TV}(\alpha, \hat{\alpha}) = \sum_{i \in S^c} \alpha_i = \delta$ (Appendix A of the technical report). When blocks have near-uniform spectral energy (std $\approx 0.03$ on a mean of $\approx 1105$, as observed in Gate 3 telemetry ✓), the evicted softmax mass is approximately proportional to the fraction of evicted tokens.

### 4.2 Projected Fleet Economics

**Table 2.** Projected annual fleet savings under the parameterized model. All values are projections (⊘) based on the OpEx and CapEx equations from §2.2 and §3.3. $\Delta\tau$ values are targets requiring empirical validation on tensor-parallel deployments at 128K+ token sequence lengths.

| Scenario | Block Sparsity ($S$) | Throughput Gain ($\Delta\tau$) (⊘) | Annual OpEx Savings (⊘) | Annual CapEx Deferral (⊘) | Total Annual Value (⊘) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Conservative** | 0.25 | 5% | $2,782,395 | $20,000,000 | **$22,782,395** |
| **Moderate** | 0.50 | 15% | $5,564,790 | $60,000,000 | **$65,564,790** |
| **Aggressive** | 0.70 | 25% | $7,790,706 | $100,000,000 | **$107,790,706** |

**Derivation trace for the Conservative row ($S = 0.25$, $\Delta\tau = 0.05$):**

$$\Delta\text{OpEx} = (200{,}000 \cdot 0.40) \cdot [0.25 \cdot 0.35 \cdot 0.550\text{ kW} \cdot 1.10 \cdot 8760\text{ hrs} \cdot \$0.075/\text{kWh}]$$
$$= 80{,}000 \cdot [0.25 \cdot 0.35 \cdot 0.550 \cdot 1.10 \cdot 8760 \cdot 0.075]$$
$$= 80{,}000 \cdot \$34.78/\text{chip-year} = \$2{,}782{,}395/\text{year}$$

$$\Delta\text{CapEx} = \$1{,}000{,}000{,}000 \cdot 0.40 \cdot 0.05 = \$20{,}000{,}000/\text{year}$$

> **All values in Table 2 are projections based on the parameterized model.** The $\Delta\tau$ values are target throughput gains — not measured speedups. Our Gate 5 measurements on Gemma 4 E2B at 4K tokens show $\Delta\tau < 0$ (16× slowdown ✓). Achieving $\Delta\tau > 0$ requires the three preconditions enumerated in §3.3. The OpEx savings assume that power reduction from block eviction scales linearly with $S$ — a simplification that does not account for minimum chip idle power draw. The fleet parameters ($N_{\text{chips}} = 200{,}000$, $\phi_{\text{inf}} = 0.40$) are engineering estimates, not disclosed Google operational data.

---

## 5. Model Generalizability

The economic value of this framework scales deterministically with the deployment parameters:

$$\Delta \text{Total}(S, \Delta \tau) = \left( N_{\text{chips}} \cdot \phi_{\text{inf}} \right) \cdot \left[ S \cdot \gamma_{\text{net}} \cdot P_{\text{chip}} \cdot \text{PUE} \cdot 8760 \cdot \text{Rate}_{\text{kWh}} + \text{Cost}_{\text{amortized}} \cdot \Delta \tau \right]$$

where $\text{Cost}_{\text{amortized}} = \$5,000 \text{ / chip-year}$ (i.e., $\$15,000$ per chip over a 3-year lifespan).

This provides infrastructure leads with a transparent spreadsheet tool: input the measured $S$ and $\Delta \tau$ from any workload class to calculate the precise annual cash yield.

**This framework is deliberately parameterized.** The measured inputs — block sparsity $S$, TV distance $\text{TV}(\alpha, \hat{\alpha})$, and reconstruction error — are empirically validated on Gemma 4 E2B at $N = 4096$ tokens on TPU v5e-8 (✓). The throughput parameter $\Delta\tau$ is an open engineering target: currently negative on our prototype (✓), projected positive under the preconditions of §3.3 (⊘). The fleet-scale constants ($N_{\text{chips}}$, $\phi_{\text{inf}}$, $\gamma_{\text{net}}$, $P_{\text{chip}}$, $\text{PUE}$, $\text{Rate}_{\text{kWh}}$) are engineering estimates derived from public Alphabet filings and standard data center modeling — they require production validation before any specific dollar figure can be treated as a forecast.

The separation between **what we have measured** (the accuracy–eviction tradeoff curve in Table 1) and **what we project** (the fleet economics in Table 2) is the epistemic core of this document. Table 1 stands on its own. Table 2 is a parameterized model awaiting its inputs.
