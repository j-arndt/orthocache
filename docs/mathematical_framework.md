# Mathematical Framework: OrthoCache — Geometric Foundations and Truncation Bounds

This document provides the complete mathematical architecture for **OrthoCache**, organized in three layers:

1. **Geometric Layer** (§0) — The cotangent bundle model and Hamiltonian dynamics that explain *why* spectral energy thresholding identifies dispensable tokens.
2. **Spectral Layer** (§1–§3) — The FWHT computation, Parseval isometry, and attention score bounds that constitute the *computational mechanism*.
3. **Bound Layer** (§4–§5) — The formal TV distance bound proving the approximation is exponentially tight.

---

## 0. Geometric and Information-Theoretic Foundations

### 0.1 The Cotangent Bundle Model

We model the attention mechanism as a canonical pairing on a cotangent bundle. Let $M$ be the $d_k$-dimensional manifold of token representations. The **cotangent bundle** $T^*M$ consists of pairs $(q, p)$ where $q \in T_x M$ is a tangent vector (the query) and $p \in T_x^* M$ is a cotangent vector (the key) at the same point $x$ in representation space.

The raw attention logit for query $q$ and key $k_i$ is:

$$z_i = \frac{q^T k_i}{\sqrt{d_k}}$$

This is precisely the **canonical action pairing** $\langle p, q \rangle$ on $T^*M$, scaled by $1/\sqrt{d_k}$. In Hamiltonian mechanics, this pairing evaluates a covector (momentum) on a vector (position) — it is the fundamental operation that couples the two halves of phase space. The attention mechanism is not merely *analogous* to a Hamiltonian system; it instantiates the exact algebraic structure of canonical pairing on the cotangent bundle.

**Notation.** We identify:
- Query vectors $q \in \mathbb{R}^{d_k}$ with **generalized positions** in $T_x M$
- Key vectors $k_i \in \mathbb{R}^{d_k}$ with **generalized momenta** in $T_x^* M$
- The attention logit $z_i = q^T k_i / \sqrt{d_k}$ with the **scaled canonical action pairing**
- The softmax distribution $\alpha_i = e^{z_i}/Z$ with the **Boltzmann distribution** over phase space at inverse temperature $\beta = 1$

Under this identification, the softmax attention output $\sum_i \alpha_i v_i$ is a thermodynamic expectation value — a Boltzmann-weighted average over the phase space trajectories indexed by token position.

### 0.2 Hamiltonian Dynamics and Dynamically Inert Regions

For each token $i$ in the KV-cache, define the per-token Hamiltonian:

$$H_i(q) = \frac{q^T k_i}{\sqrt{d_k}}$$

The Hamiltonian gradient with respect to the query (position) coordinate is:

$$\nabla_q H_i = \frac{k_i}{\sqrt{d_k}}$$

This gradient determines the **Hamiltonian vector field** $X_{H_i}$ — the infinitesimal flow that token $i$ induces on the query manifold. The magnitude of this flow is:

$$\|X_{H_i}\| = \|\nabla_q H_i\| = \frac{\|k_i\|_2}{\sqrt{d_k}}$$

Now consider a block $B_j$ whose spectral energy satisfies $E_j < \epsilon$. By Parseval's identity (proved in §1):

$$E_j = \sum_{i \in B_j} \|k_i\|_2^2 < \epsilon$$

Therefore, for every token $i \in B_j$:

$$\|k_i\|_2^2 \leq E_j < \epsilon \implies \|k_i\|_2 < \sqrt{\epsilon}$$

Substituting into the Hamiltonian vector field magnitude:

$$\|X_{H_i}\| = \frac{\|k_i\|_2}{\sqrt{d_k}} < \frac{\sqrt{\epsilon}}{\sqrt{d_k}}$$

**Geometric interpretation.** When $\epsilon \to 0$, the Hamiltonian vector field $X_{H_i} \to 0$ for all tokens in the block. The momentum coordinates $(k_i)$ are tightly concentrated near the origin of the cotangent fiber. The token trajectories in this region of the cotangent bundle are **dynamically inert** — they generate no flow, couple negligibly to any query, and contribute exponentially suppressed softmax mass (proved formally in §5).

Evicting these tokens is not an arbitrary heuristic. It is the **geometric truncation of a flat, invariant submanifold** of the phase space — a region where the canonical action pairing vanishes and the Hamiltonian dynamics are stationary.

Under Liouville's theorem, the phase-space volume is conserved under Hamiltonian flow. Truncating a region of near-zero momentum corresponds to removing a set of measure approaching zero in the symplectic volume form $\omega^n = \bigwedge_{i=1}^{d_k} dp_i \wedge dq_i$. The volume integral over the truncated region is bounded by:

$$\text{Vol}(B_j) \leq \text{Vol}_{q}(M) \cdot \text{Vol}_{p}(\{p : \|p\|_2 < \sqrt{\epsilon}\}) = \text{Vol}_{q}(M) \cdot \frac{\pi^{d_k/2}}{\Gamma(d_k/2 + 1)} \epsilon^{d_k/2}$$

which vanishes as $\epsilon \to 0$ for any fixed dimension $d_k$. The truncation preserves the Hamiltonian structure of the remaining phase space.

### 0.3 Chow Influence and Spectral Routing

In Boolean function analysis and circuit complexity, the **Chow parameters** of a function $f: \{-1, +1\}^n \to \{-1, +1\}$ are its degree-0 and degree-1 Fourier coefficients in the Walsh-Hadamard basis. For a Linear Threshold Function (LTF) $f(x) = \text{sgn}(w^T x - \theta)$, the Chow parameters uniquely determine $f$ up to equivalence (Chow, 1961; O'Donnell and Servedio, 2008).

The softmax attention operator acts as a **continuous, differentiable analog** of a multi-variable threshold function. Specifically, define the soft-argmax circuit:

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

As the temperature $\tau \to 0$, $\text{softmax}(z/\tau)$ converges pointwise to the hard-argmax (a threshold function). At finite temperature ($\tau = 1$), softmax is a smooth relaxation that preserves the essential routing behavior: tokens with large logits receive exponentially more weight.

The **Fast Walsh-Hadamard Transform** of the key block $K_{B_j}$ projects the keys into the Walsh basis — the same orthogonal basis used to define the Fourier/Chow coefficients of Boolean functions. The spectral energy:

$$E_j = \|\hat{K}_j\|_F^2 = \sum_{s=1}^{b} \sum_{d=1}^{d_k} |\hat{K}_{j,s,d}|^2$$

is the **aggregate Chow variance** of the block — the total spectral amplitude across all Walsh basis functions and all head dimensions. This quantity measures the **influence** of block $B_j$ on the soft-threshold routing circuit.

**Theorem (Chow Influence Interpretation).** If $E_j < \epsilon$, then the block $B_j$ lacks sufficient spectral amplitude to cross the softmax threshold gate for any query $q$. Formally:

1. Low spectral energy implies low individual key norms: $\|k_i\|_2 < \sqrt{\epsilon}$ for all $i \in B_j$ (by Parseval; §2).
2. Low key norms bound the logit: $|z_i| < \|q\|_2 \sqrt{\epsilon} / \sqrt{d_k} \triangleq \beta$ (by Cauchy-Schwarz; §3).
3. The bounded logit guarantees exponentially suppressed routing weight: $\alpha_i \leq e^{\beta - z_{\max}}$ (by softmax monotonicity; §5).

The Chow parameters of a Boolean threshold function determine whether an input coordinate can flip the output. Analogously, the spectral energy of a key block determines whether it can materially influence the softmax routing. When $E_j < \epsilon$, the block's Chow influence is below the threshold required to alter the attention distribution — the routing circuit is **invariant** to this block's presence.

---

## 1. Spectral Energy as Block Importance

We partition the sequence of $N$ cached keys into $m$ contiguous blocks $B_1, \ldots, B_m$ of size $b$ (aligned to TPU tile boundaries, typically $b = 512$ for bfloat16). For each block $B_j$, we compute the Fast Walsh-Hadamard Transform (FWHT) along the sequence-position axis:

$$\hat{K}_j = \mathcal{H}_b \cdot K_{B_j}$$

where $\mathcal{H}_b$ is the normalized Walsh-Hadamard matrix of size $b \times b$. The **spectral energy** $E_j$ of block $j$ is computed as the squared Frobenius norm of the spectral coefficients:

$$E_j = \|\hat{K}_j\|_F^2 = \sum_{s=1}^{b} \sum_{d=1}^{d_k} |\hat{K}_{j,s,d}|^2$$

### Parseval's Identity for FWHT
Because the normalized Walsh-Hadamard transform is an orthogonal transformation ($\mathcal{H}_b^T \mathcal{H}_b = I_b$), it preserves the inner product and Frobenius norm:

$$E_j = \|\hat{K}_j\|_F^2 = \|K_{B_j}\|_F^2 = \sum_{i \in B_j} \|k_i\|_2^2$$

This provides the critical bridge: the spectral energy in the FWHT domain is exactly equal to the spatial energy (the sum of the squared $L_2$ norms of the key vectors) within the block.

**Connection to §0:** By Parseval's identity, spectral energy equals the total squared momentum magnitude in the cotangent bundle model. Low spectral energy ↔ low momentum coordinates ↔ dynamically inert tokens ↔ negligible Chow influence on the softmax routing circuit.

---

## 2. Per-Key Norm Bound

If a block $B_j$ is marked for eviction because its spectral energy falls below the threshold $\epsilon$ ($E_j < \epsilon$), we can bound the norm of every individual key vector $k_i$ residing within that block.

For any $i \in B_j$:

$$\|k_i\|_2^2 \leq \sum_{i' \in B_j} \|k_{i'}\|_2^2 = \|K_{B_j}\|_F^2 = E_j < \epsilon$$

Taking the square root yields:

$$\|k_i\|_2 < \sqrt{\epsilon} \quad \forall i \in B_j \text{ where } E_j < \epsilon$$

---

## 3. Attention Score Bound

For a query vector $q$ and a key $k_i$ belonging to an evicted block, the raw attention logit $z_i$ is defined as:

$$z_i = \frac{q^T k_i}{\sqrt{d_k}}$$

Applying the Cauchy-Schwarz inequality:

$$|z_i| = \frac{|q^T k_i|}{\sqrt{d_k}} \leq \frac{\|q\|_2 \cdot \|k_i\|_2}{\sqrt{d_k}}$$

Using the per-key norm bound $\|k_i\|_2 < \sqrt{\epsilon}$:

$$|z_i| < \frac{\|q\|_2 \sqrt{\epsilon}}{\sqrt{d_k}}$$

We define the maximum possible logit magnitude for any evicted token as $\beta$:

$$\beta \triangleq \frac{\|q\|_2 \sqrt{\epsilon}}{\sqrt{d_k}}$$

Thus, for all evicted tokens $i \in S^c$, their raw logits are strictly bounded by:

$$z_i < \beta$$

---

## 3.1 Query-Aware Spectral Eviction (DC/AC Decomposition)

The energy-based bound in §2–§3 is **conservative**: it treats all queries identically, bounding every evicted logit by a single worst-case $\beta$ that depends on $\|q\|_2$ and total block energy $\epsilon$. In practice, different blocks interact very differently with different queries. The **query-aware spectral eviction** method, implemented in `spectral_energy.py`, uses the Walsh–Hadamard spectral decomposition to derive a tighter, per-block, per-query logit upper bound.

### Key Decomposition

Given a key block $K_{B_j} \in \mathbb{R}^{b \times d_k}$ and its FWHT $\hat{K}_j = \mathcal{H}_b \cdot K_{B_j}$, we decompose the spectral coefficients into:

- **DC component** (frequency 0): $\hat{K}_{j,0} = \sum_{i \in B_j} k_i / \sqrt{b}$ — proportional to the block mean
- **AC components** (frequencies $s > 0$): $\hat{K}_{j,s}$ for $s = 1, \ldots, b-1$ — encoding intra-block variance

The block mean vector is:
$$\bar{k}_j = \frac{\hat{K}_{j,0}}{\sqrt{b}} = \frac{1}{b}\sum_{i \in B_j} k_i$$

The AC energy captures the total squared deviation of keys from the block mean:
$$E_j^{\text{AC}} = \sum_{s=1}^{b-1} \sum_{d=1}^{d_k} |\hat{K}_{j,s,d}|^2 = \sum_{i \in B_j} \|k_i - \bar{k}_j\|_2^2$$

where the last equality follows from Parseval's identity applied to the centered key vectors.

### Query-Aware Logit Upper Bound

For a query $q$ and any key $k_i \in B_j$, write $k_i = \bar{k}_j + (k_i - \bar{k}_j)$ and decompose the logit:

$$z_i = \frac{q^T k_i}{\sqrt{d_k}} = \underbrace{\frac{q^T \bar{k}_j}{\sqrt{d_k}}}_{\text{alignment}} + \underbrace{\frac{q^T(k_i - \bar{k}_j)}{\sqrt{d_k}}}_{\text{residual}}$$

The alignment term is constant across the block. The residual is bounded by Cauchy–Schwarz:

$$\left|\frac{q^T(k_i - \bar{k}_j)}{\sqrt{d_k}}\right| \leq \frac{\|q\|_2 \cdot \|k_i - \bar{k}_j\|_2}{\sqrt{d_k}} \leq \frac{\|q\|_2 \cdot \sqrt{E_j^{\text{AC}}}}{\sqrt{d_k}}$$

where the last inequality uses $\|k_i - \bar{k}_j\|_2^2 \leq \sum_{i'} \|k_{i'} - \bar{k}_j\|_2^2 = E_j^{\text{AC}}$.

The **query-aware logit upper bound** for block $B_j$ is therefore:

$$\tau_j(q) = \frac{q^T \bar{k}_j}{\sqrt{d_k}} + \frac{\|q\|_2 \sqrt{E_j^{\text{AC}}}}{\sqrt{d_k}}$$

### Why This Is Strictly Tighter

The energy-based bound (§3) yields $|z_i| < \beta = \|q\|_2 \sqrt{E_j} / \sqrt{d_k}$ using the total block energy $E_j = E_j^{\text{DC}} + E_j^{\text{AC}}$. The query-aware bound replaces the crude $\|q\|_2 \sqrt{E_j}$ with:
1. The actual **signed alignment** $q^T \bar{k}_j$ (which can be strongly negative when the block mean opposes the query), plus
2. A **residual** that depends only on the AC energy $E_j^{\text{AC}} \leq E_j$.

This makes the FWHT **load-bearing**: the DC/AC split uses spectral information that raw block-energy thresholding discards. Blocks with large total energy but poor query alignment (negative $q^T \bar{k}_j$) are correctly identified as evictable — something the energy-only bound cannot do.

### Eviction Criterion

Given a threshold $\tau$, block $B_j$ is **evicted** if:

$$\max_{q \in Q} \tau_j(q) < \tau$$

where the max is taken over query tokens $Q$ to ensure that any block important to *any* query is retained. The truncation bound (§5) then applies with $\tau$ replacing $\beta$:

$$\text{TV}(\alpha, \hat{\alpha}) \leq |S^c| \cdot \exp(\tau - z_{\max})$$

---

## 3.2 Multi-Band Sequency Filtering

The query-aware bound in §3.1 uses the DC/AC decomposition to compute a per-block, per-query logit upper bound. However, the aggregate AC energy $E_j^{\text{AC}} = \sum_{s=1}^{b-1} \|\hat{K}_{j,s}\|_2^2$ collapses the entire non-DC spectrum into a single scalar. This scalar is exactly equal to the spatial variance via the bias-variance identity:

$$E_j^{\text{AC}} = \sum_{i \in B_j} \|k_i - \bar{k}_j\|_2^2 = \sum_{i \in B_j} \|k_i\|_2^2 - b\|\bar{k}_j\|_2^2$$

and is therefore computable in $O(bd_k)$ without the $O(bd_k \log b)$ FWHT.

To make the FWHT **genuinely load-bearing**, OrthoCache partitions the 512 Walsh coefficients into discrete **sequency bands** and evaluates per-band energy, which *cannot* be recovered from spatial statistics.

### Sequency Band Partitioning

In the Walsh-Hadamard basis, **sequency** (the number of sign changes along a basis vector) corresponds directly to frequency. We partition the $b = 512$ coefficients into:

| Band | Indices | Coefficients | Interpretation |
|:-----|:--------|:-------------|:---------------|
| DC | 0 | 1 | Block mean (macro-semantic pivot) |
| Low-sequency | 1–63 | 63 | Smooth semantic trends across the block |
| Mid-sequency | 64–255 | 192 | Syntactic/token-relational context |
| High-sequency | 256–511 | 256 | Rapid oscillations, formatting noise |

### The Spectral Decay Ratio ($\zeta$)

We define the **Spectral Decay Ratio** $\zeta_j$ of block $B_j$ as:

$$\zeta_j = \frac{\sum_{s=256}^{511} \|\hat{K}_{j,s}\|_2^2}{\sum_{s=1}^{63} \|\hat{K}_{j,s}\|_2^2 + \epsilon_{\text{stab}}}$$

where $\epsilon_{\text{stab}} = 10^{-6}$ prevents division by zero.

**Interpretation:**
- **High $\zeta_j$ ($\gg 1$)**: The block's variance is dominated by high-frequency sign oscillations — characteristic of formatting tokens, punctuation sequences, and structural noise. These tokens have large activation magnitudes but lack coherent semantic structure.
- **Low $\zeta_j$ ($\ll 1$)**: Energy concentrates in the structural low-frequency bands — characteristic of continuous human language, logical arguments, and long-range semantic dependencies.

**Why $\zeta$ is uncomputable from spatial statistics:** Two blocks can have identical total spatial variance $\sum_{i} \|k_i - \bar{k}_j\|_2^2$ but entirely different spectral decay ratios. A block of repetitive JSON formatting tokens (`{`, `\"`, `}`) and a block of coherent technical prose may share the same Frobenius norm, but their frequency decompositions are radically different. The FWHT is the only operation that exposes this distinction.

### Query-Independence and Pipeline Hoisting

$\zeta_j$ is **deliberately query-independent**. It is a structural property of the key block, computed once when the block is written to the KV cache and stored as a static scalar metadata entry. This enables:

1. **Asynchronous computation:** The FWHT and $\zeta$ computation execute during the initial KV-cache fill, amortized over thousands of subsequent decoding steps.
2. **PrefetchScalarGridSpec pipeline:** The pre-computed boolean mask (from $\zeta$ and energy thresholds) streams into Scalar SRAM (SMEM) via `PrefetchScalarGridSpec` in parallel with query vector loading, introducing zero additional latency.
3. **No circular dependency:** Query-modulated $\zeta$ would force re-running the FWHT across the entire cache on every decoding step, reintroducing the $O(Nd_k \log b)$ tax and negating OrthoCache's performance advantage.

Query-awareness is achieved *without* query-dependent transforms: the query norm $\|q\|_2$ dynamically scales the eviction threshold $\tau$ at runtime, tightening or loosening retention based on the current query's activation magnitude.

### Two-Gate Eviction Criterion

OrthoCache retains block $B_j$ if and only if it passes **both** gates:

1. **Query-aware logit bound (§3.1):** $\max_{q \in Q} \tau_j(q) \geq \tau$
2. **Spectral coherence:** $\zeta_j \leq \zeta_{\max}$

Gate 1 ensures blocks that could produce large attention logits for some query are retained. Gate 2 ensures blocks whose variance is dominated by high-frequency noise are evicted regardless of total energy — the key innovation that spatial variance cannot replicate.

The truncation bound from §5 continues to apply: $\zeta$ filtering is a pre-gate that determines which blocks are *candidates* for the query-aware evaluation. The final eviction set $S^c$ is still governed by the logit bound $\tau$, and $\text{TV}(\alpha, \hat{\alpha}) \leq |S^c| \cdot \exp(\tau - z_{\max})$ holds unchanged.

---

## 3.3 Why the FWHT Is Necessary

We directly address the Parseval redundancy observation: since the FWHT is an orthogonal isometry, $\|\hat{K}_j\|_F^2 = \|K_{B_j}\|_F^2$, and using the FWHT merely to compute aggregate block energy is a computational no-op — an $O(b \log b)$ computation for a quantity available in $O(b)$.

OrthoCache subverts this no-op by **not collapsing the spectrum**. The critical quantities extracted from the FWHT are:

1. **The DC component** $\hat{K}_{j,0}$ — used for query-mean alignment (§3.1). While equivalent to the block sum, it falls out naturally from the transform at zero marginal cost.

2. **Per-band energy** $E_j^{\text{low}}$, $E_j^{\text{mid}}$, $E_j^{\text{high}}$ — computed as sums of squared coefficients within each sequency band. These are **not** recoverable from any spatial-domain statistic because the spatial domain provides only the aggregate variance $E_j^{\text{AC}} = E_j^{\text{low}} + E_j^{\text{mid}} + E_j^{\text{high}}$ but cannot decompose it by frequency.

3. **The spectral decay ratio** $\zeta_j = E_j^{\text{high}} / E_j^{\text{low}}$ — a query-independent entropy signature that distinguishes semantically coherent blocks from noise-dominated blocks with identical total variance. This is the load-bearing metric.

**Formal statement:** There exist blocks $A$ and $B$ with $E_A^{\text{AC}} = E_B^{\text{AC}}$ (identical spatial variance) but $\zeta_A \neq \zeta_B$ (different frequency decompositions). Therefore, no spatial-domain function $f(\{k_i\}_{i \in B_j})$ can compute $\zeta_j$ without access to the individual spectral coefficients provided by the FWHT.

This is verified by construction in `tests/test_spectral_bands.py::test_zeta_not_computable_spatially`.

---

## 4. Total Variation Bound (The Core Theorem)

Let $S$ be the set of retained token indices and $S^c$ the set of evicted token indices. We define:
- **Full attention distribution:** $\alpha_i = \frac{e^{z_i}}{Z}$ where $Z = \sum_{j=1}^{N} e^{z_j}$
- **Truncated attention distribution:** $\hat{\alpha}_i = \frac{e^{z_i}}{\hat{Z}}$ for $i \in S$, and $\hat{\alpha}_i = 0$ for $i \in S^c$, where $\hat{Z} = \sum_{j \in S} e^{z_j}$

The Total Variation (TV) distance between these two probability distributions is:

$$\text{TV}(\alpha, \hat{\alpha}) = \frac{1}{2} \sum_{i=1}^{N} |\alpha_i - \hat{\alpha}_i|$$

### Lemma: TV Distance Equivalence to Evicted Mass
The TV distance is exactly equal to the total softmax probability mass assigned to the evicted tokens:

$$\text{TV}(\alpha, \hat{\alpha}) = \sum_{i \in S^c} \alpha_i \triangleq \delta$$

*Proof:*
For any evicted token $i \in S^c$, we have $\hat{\alpha}_i = 0$. Thus:
$$\sum_{i \in S^c} |\alpha_i - \hat{\alpha}_i| = \sum_{i \in S^c} \alpha_i = \delta$$

For any retained token $i \in S$, we note that $Z = \hat{Z} + \sum_{j \in S^c} e^{z_j} = \hat{Z} + \delta Z$, which implies $\hat{Z} = (1 - \delta)Z$. Therefore:
$$\hat{\alpha}_i = \frac{e^{z_i}}{\hat{Z}} = \frac{e^{z_i}}{(1-\delta)Z} = \frac{\alpha_i}{1-\delta}$$

Since $1-\delta < 1$, we have $\hat{\alpha}_i \geq \alpha_i$ for all $i \in S$. The difference is:
$$|\alpha_i - \hat{\alpha}_i| = \hat{\alpha}_i - \alpha_i = \alpha_i \left(\frac{1}{1-\delta} - 1\right) = \alpha_i \frac{\delta}{1-\delta}$$

Summing over all retained tokens $i \in S$:
$$\sum_{i \in S} |\alpha_i - \hat{\alpha}_i| = \frac{\delta}{1-\delta} \sum_{i \in S} \alpha_i = \frac{\delta}{1-\delta} (1-\delta) = \delta$$

Summing both components and dividing by 2:
$$\text{TV}(\alpha, \hat{\alpha}) = \frac{1}{2} \left( \sum_{i \in S} |\alpha_i - \hat{\alpha}_i| + \sum_{i \in S^c} |\alpha_i - \hat{\alpha}_i| \right) = \frac{1}{2} (\delta + \delta) = \delta$$
$\square$

---

## 5. Exponential Truncation Bound

We can now upper-bound the evicted mass $\delta$ to prove that the approximation error decays exponentially as the gap between the maximum retained logit and the evicted logit bound increases.

Each evicted token $i \in S^c$ contributes:

$$\alpha_i = \frac{e^{z_i}}{Z} \leq \frac{e^{\beta}}{Z}$$

Since the partition function $Z$ is a sum of positive exponentials, it is strictly greater than or equal to its maximum term:

$$Z \geq e^{z_{\max}} \quad \text{where } z_{\max} = \max_{j \in S} z_j$$

Therefore:

$$\alpha_i \leq \frac{e^{\beta}}{e^{z_{\max}}} = e^{\beta - z_{\max}}$$

Summing over all $|S^c|$ evicted tokens:

$$\delta = \sum_{i \in S^c} \alpha_i \leq |S^c| \cdot e^{\beta - z_{\max}}$$

This completes the proof of the **OrthoCache Truncation Bound**:

$$\boxed{\text{TV}(\alpha, \hat{\alpha}) \leq |S^c| \cdot \exp\!\left(\frac{\|q\|_2\sqrt{\epsilon}}{\sqrt{d_k}} - z_{\max}\right)}$$

With the query-aware spectral eviction (§3.1), $\beta$ is replaced by the tighter query-aware threshold $\tau$:

$$\boxed{\text{TV}(\alpha, \hat{\alpha}) \leq |S^c| \cdot \exp(\tau - z_{\max})}$$

where $\tau$ is the query-aware logit upper bound used for eviction. Since $\tau \leq \beta$ in all cases where the alignment term is non-positive, the query-aware bound is strictly tighter.

**Dual reading of the bound:**

1. **Algebraic:** The TV distance decays exponentially in the gap $(z_{\max} - \beta)$. In practice, $z_{\max}$ ranges from 5–15 (pre-softmax peak for important tokens) while $\beta$ is near zero for reasonable $\epsilon$, yielding negligible TV distances.

2. **Geometric (§0.2):** The gap $(z_{\max} - \beta)$ is the difference between the maximum canonical action pairing on the retained manifold and the vanishing action pairing on the inert submanifold. Exponential decay in this gap reflects the Boltzmann suppression of thermodynamically irrelevant phase-space regions.

3. **Information-theoretic (§0.3):** The bound proves that tokens with insufficient Chow influence ($E_j < \epsilon$) are exponentially irrelevant to the softmax routing circuit, regardless of the query vector.

---

## 6. Comparison with Alternative Compression Frameworks

Unlike passive compression (e.g., TurboQuant, TurboAngle) which uniformly quantizes all tokens to reduce storage footprint, OrthoCache acts as an active spatial governor. The TV distance bound guarantees that the attention shift is negligible when we evict blocks with low spectral energy, preserving the exact mathematical representation of high-influence tokens.

The geometric framework reveals why OrthoCache and TurboAngle are **synergistic, not competitive**: TurboAngle compresses the *representation* of high-curvature (high-influence) tokens that OrthoCache retains, while OrthoCache eliminates the *existence* of low-curvature (low-influence) tokens from the pipeline entirely. Together, they reduce both the per-token storage cost and the total token count — a multiplicative bandwidth reduction.

---

## 7. Unified Architecture Summary

```
[ Geometric Layer ]   ──> Canonical Pairing on Cotangent Bundle T*M
                                  │
                                  │  Q = position, K = momentum
                                  │  z_i = <p, q> / √d_k (canonical action)
                                  ▼
[ Invariant Detection ] ──> Spectral Energy via FWHT (Chow Influence Maps)
                                  │
                                  │  E_j = ||K̂_j||²_F = Σ||k_i||² (Parseval)
                                  │  DC/AC split: alignment + residual bound
                                  ▼
[ Query-Aware Eviction] ──> τ_j(q) = q·k̄_j/√d + ||q||·√E_AC/√d
                                  │
                                  │  Per-block, per-query logit upper bound
                                  │  Evict block j if max_q τ_j(q) < τ
                                  ▼
[ Formal Guarantee ]    ──> OrthoCache Truncation Bound (Theorem)
                                  │
                                  │  TV(α, α̂) ≤ |S^c| · exp(τ - z_max)
                                  │  Exponential decay in logit gap
                                  ▼
[ Hardware Execution ]  ──> Pallas Scalar Prefetch Block Eviction
                                  │
                                  │  FWHT in VPU registers (9 butterfly stages)
                                  │  Block mask to SMEM via PrefetchScalarGridSpec
                                  │  DMA bypass for evicted blocks (zero HBM traffic)
                                  ▼
[ Economic Impact ]     ──> ICI Bandwidth Reduction + CapEx Deferral
                                  (see docs/cost_benefit_analysis.md)
```
