# Mathematical Framework: OrthoCache Truncation Bounds

This document provides the formal mathematical derivations for **OrthoCache** (formerly Project Ironclad), establishing the proof chain that bounds the attention distribution shift when evicting low-influence Key-Value cache blocks.

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

---

## 6. Comparison with Alternative Compression Frameworks

Unlike passive compression (e.g., TurboQuant, TurboAngle) which uniformly quantizes all tokens to reduce storage footprint, OrthoCache acts as an active spatial governor. The TV distance bound guarantees that the attention shift is negligible when we evict blocks with low spectral energy, preserving the exact mathematical representation of high-influence tokens.
