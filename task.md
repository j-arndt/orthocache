# OrthoCache Execution Task List

This task list tracks the execution of the OrthoCache plan.

## Phase 1: Code and Benchmarks (Problems P1-P7)
- [x] Implement query-aware spectral bound in `src/orthocache/spectral_energy.py`
- [x] Rewrite Pallas kernel to use online softmax loops in `src/orthocache/sparse_attention.py`
- [x] Update `src/orthocache/reference.py` to match the query-aware algorithm
- [x] Fix natural text generation and query-key separation in `benchmarks/attention_accuracy.py`
- [x] Correct the bound checks in `tests/test_truncation_bound.py` and `benchmarks/attention_accuracy.py` to use the true theoretical $\beta$
- [x] Run pytest to verify all test suites pass: `$env:PYTHONPATH="src"; python -m pytest -p no:dandi tests/ -v`

## Phase 2: Lean Theorems Verification
- [x] Thread the query-dependent $\beta$ bound through `proofs/OrthoCacheMath/TruncationBound.lean`
- [x] Verify both `ParsevalWHT.lean` and `TruncationBound.lean` compile completely without `sorry` warnings
- [x] Run `lake build` to confirm zero errors

## Phase 3: Gemma 31B Notebook Update
- [x] Update `orthocache-v2-31b_v3.ipynb` with the query-aware spectral eviction algorithm and fixed Pallas kernel
- [x] Execute notebook cells to collect non-degenerate telemetry
- [x] Add matplotlib chart cells (spectral energy distribution, TV distance Pareto curve, latency crossover)
- [x] Add a download/export cell at the end of the notebook to allow downloading the results

## Phase 4: Repo, Docs, and the Real Paper
- [x] Update `docs/mathematical_framework.md` to document query-aware spectral bounds
- [x] Update `README.md` to remove any obsolete references and add accurate diagrams/descriptions
- [x] Write the real preprint paper in `docs/technical_report.md`
- [x] Commit and clean up the repository
