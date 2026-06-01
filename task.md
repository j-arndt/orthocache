# OrthoCache Execution Tasks

This task list serves as the central coordination document for the implementation, validation, and profiling of **OrthoCache** (formerly Project Ironclad).

## Phase 1: Mathematical Foundation & Formal Proofs
- [x] Draft `docs/mathematical_framework.md` containing full derivations for:
  - [x] Parseval's identity for FWHT bridging spectral and spatial energy
  - [x] Per-key norm upper bounds based on block energy thresholds ($\epsilon$)
  - [x] Cauchy-Schwarz logit bound ($\beta$)
  - [x] Softmax TV distance equality to evicted mass ($\delta$)
  - [x] Exponential TV distance upper bound: $\text{TV}(\alpha, \hat{\alpha}) \leq |S^c| \cdot \exp(\beta - z_{\max})$
- [x] Draft `docs/cost_benefit_analysis.md` containing macroeconomic infrastructure model
- [x] Establish Lean 4 environment:
  - [x] Create `proofs/lakefile.lean` with Mathlib4 dependency
  - [x] Configure Lean toolchain version file (`proofs/lean-toolchain`)
  - [x] Verify `lake build` compiles without errors (sorry warnings expected)
- [ ] Implement Lean 4 proofs:
  - [ ] `proofs/OrthoCacheMath/ParsevalWHT.lean`: Fix WHT matrix definition (use 2^n dimensions + Kronecker product), then prove $H_n^T H_n = n \cdot I$
  - [ ] `proofs/OrthoCacheMath/TruncationBound.lean`: Fix `S_c_card` binding (derive from `S_c` cardinality), then prove the exponential TV distance bound
- [ ] Review documentation and math for consistent nomenclature (OrthoCache, sequence sparsity $S$, throughput speedup $\Delta \tau$, error bounds).
  - [ ] Purge ~12 remaining "Ironclad" references from `implementation_plan.md`

## Phase 2: Compilable Pallas Kernels & Reference Implementation
- [x] Set up project structure and packaging:
  - [x] Create `pyproject.toml` with packaging metadata (JAX, NumPy, Transformers, Pytest)
  - [x] Create `src/orthocache/__init__.py` exposing core user API (5 public symbols)
- [x] Implement reference CPU models:
  - [x] `src/orthocache/reference.py`: NumPy-based iterative butterfly FWHT (`numpy_fwht_1d`, `numpy_fwht`)
  - [x] `src/orthocache/reference.py`: Block energy reference (`compute_block_energy_reference`)
  - [x] `src/orthocache/reference.py`: TV distance calculator (`compute_tv_distance`)
- [x] Develop Pallas TPU v5e/v5p compatible kernels:
  - [x] `src/orthocache/fwht.py`: Pure tensor-reshaping 512-row radix-2 FWHT (9 unrolled stages)
  - [x] `src/orthocache/spectral_energy.py`: Block energy summation and boolean threshold mask builder
  - [x] `src/orthocache/sparse_attention.py`: JAX block-sparse attention + Pallas kernel wrapper
  - [x] **Fix**: Replaced Python `if not mask:` with `jnp.where()` for TPU-compilable conditional execution

## Phase 3: Automated Testing & Equivalence Validation
- [x] Write pytest correctness suites:
  - [x] `tests/test_fwht.py`: 2D + 1D FWHT correctness vs NumPy reference (2 tests)
  - [x] `tests/test_energy.py`: Block energy correctness + threshold mask logic (2 tests)
  - [x] `tests/test_attention.py`: Sparse vs dense attention equivalence + TV distance sanity (1 test)
- [x] Execute pytest suite in JAX CPU emulation mode: **5 passed in 6.91s**
- [ ] Add TV-bound validation test: verify measured TV ≤ theoretical upper bound for random inputs

## Phase 4: Kaggle TPU v5e-8 Execution & Telemetry Profiling
- [ ] Spin up Kaggle TPU v5e-8 environment and clone codebase.
- [ ] **Gate 1: Compilation Test**
  - [ ] Compile and run a basic JIT execution trace for all three Pallas kernels on TPU.
- [ ] **Gate 2: Correctness Test**
  - [ ] Run the pytest suite directly on TPU v5e hardware (bfloat16: rtol=1e-2, atol=1e-3).
- [ ] **Gate 3: Empirical Telemetry Collection (Gemma-2B)**
  - [ ] Create `benchmarks/spectral_analysis.py` to load Gemma-2B, extract KV-cache, run FWHT.
  - [ ] Feed long-context prompt sequences (8K to 32K tokens).
  - [ ] Generate block sparsity distributions (histograms + CDFs) per sequence class.
- [ ] **Gate 4: Accuracy vs. Sparsity Tradeoff Mapping**
  - [ ] Create `benchmarks/attention_accuracy.py` for TV/KL divergence at 10/30/50/70% eviction.
  - [ ] Output accuracy tradeoff curves and verify OrthoCache Truncation Bound holds.
- [ ] **Gate 5: Profiling and Latency Measurement**
  - [ ] Create `benchmarks/profiling.py` for `jax.profiler` sparse vs. dense timing.
  - [ ] Export raw `.xprof` profiling trace logs.

## Phase 5: Technical Report, Cost Model, & Outreach
- [ ] Create `README.md` with architecture diagrams, setup/execution commands, and benchmark placeholders.
- [ ] Create `.gitignore` for Python + Lean ignore patterns.
- [ ] Write `docs/technical_report.md` (TechRxiv pre-print):
  - [ ] Hardware-native block filtering architecture detail
  - [ ] Empirical benchmark charts and profiling metrics from Phase 4
  - [ ] Macroeconomic infrastructure model with measured parameters
- [ ] Write `docs/xla_pass_design.md` (proposed future compiler integration)
- [ ] Compile technical report into IEEE preprint format and submit to TechRxiv.
- [ ] Draft LinkedIn outreach message to Yannis Papakonstantinou.
- [ ] Open RFC issue on `google/jax` or `openxla/xla`.
