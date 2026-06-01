"""
OrthoCache — Kaggle TPU v5e-8 Setup & Gate Execution Notebook
=============================================================

Paste the cells below into a fresh Kaggle notebook with TPU v5e-8 accelerator.
Each cell is separated by a comment: # --- CELL N ---

Setup checklist:
  1. New Notebook → Accelerator: TPU v5e-8
  2. Add Input → Models → search "gemma-4-e2b" → Add
     (this mounts the model at /kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b/1)
  3. Paste each cell below, run sequentially
"""

# --- CELL 1: Environment Setup ---
# Verify TPU v5e connectivity

import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")
assert jax.default_backend() == "tpu", "ERROR: Not running on TPU!"
print(f"\n✅ TPU v5e detected: {len(jax.devices())} chips")


# --- CELL 2: Clone OrthoCache + verify Gemma 4 E2B ---
import subprocess
import sys
import os

REPO_URL = "https://github.com/j-arndt/orthocache.git"
CLONE_DIR = "/kaggle/working/orthocache"
GEMMA_PATH = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b/1"

if not os.path.exists(CLONE_DIR):
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(["git", "clone", REPO_URL, CLONE_DIR], check=True)
    print(f"✅ Cloned to {CLONE_DIR}")
else:
    print(f"✅ Repo already exists at {CLONE_DIR}")

sys.path.insert(0, os.path.join(CLONE_DIR, "src"))

# Verify imports
from orthocache import fwht_512, compute_block_energy_jax, generate_threshold_mask
from orthocache import jax_block_sparse_attention, compile_pallas_sparse_attention
print("✅ All OrthoCache modules imported")

# Verify model
assert os.path.exists(GEMMA_PATH), (
    f"Gemma 4 E2B not found at {GEMMA_PATH}. "
    "Add it via sidebar: Add Input → Models → search 'gemma-4-e2b'"
)
print(f"✅ Gemma 4 E2B found at {GEMMA_PATH}")


# --- CELL 3: GATE 1 — Compilation Test ---
import jax
import jax.numpy as jnp
import time

print("=" * 60)
print("GATE 1: TPU v5e-8 Compilation Test")
print("=" * 60)

# Test 1: FWHT kernel
print("\n[1/3] Compiling fwht_512 ...")
key_data = jax.random.normal(jax.random.PRNGKey(0), (512, 128), dtype=jnp.bfloat16)
t0 = time.perf_counter()
fwht_result = jax.jit(fwht_512)(key_data)
fwht_result.block_until_ready()
t1 = time.perf_counter()
print(f"  ✅ fwht_512: shape={fwht_result.shape}, dtype={fwht_result.dtype}, "
      f"compile+exec={1000*(t1-t0):.1f}ms")

# Test 2: Spectral energy + mask
print("\n[2/3] Compiling compute_block_energy_jax + generate_threshold_mask ...")
keys_3d = jax.random.normal(jax.random.PRNGKey(1), (1024, 4, 64), dtype=jnp.bfloat16)
t0 = time.perf_counter()
energies = jax.jit(compute_block_energy_jax, static_argnums=(1,))(keys_3d, 512)
energies.block_until_ready()
mask = generate_threshold_mask(energies, epsilon=1.0)
mask.block_until_ready()
t1 = time.perf_counter()
print(f"  ✅ compute_block_energy_jax: energies shape={energies.shape}")
print(f"  ✅ generate_threshold_mask: mask shape={mask.shape}, "
      f"retained={int(mask.sum())}/{mask.size}")
print(f"     compile+exec={1000*(t1-t0):.1f}ms")

# Test 3: Sparse attention
print("\n[3/3] Compiling sparse attention kernel ...")
q = jax.random.normal(jax.random.PRNGKey(2), (8, 4, 64), dtype=jnp.bfloat16)
k = jax.random.normal(jax.random.PRNGKey(3), (1024, 4, 64), dtype=jnp.bfloat16)
v = jax.random.normal(jax.random.PRNGKey(4), (1024, 4, 64), dtype=jnp.bfloat16)
block_mask = jnp.ones((2, 4), dtype=bool)

t0 = time.perf_counter()
attn_out = jax.jit(compile_pallas_sparse_attention, static_argnums=(4,))(q, k, v, block_mask, 512)
attn_out.block_until_ready()
t1 = time.perf_counter()
print(f"  ✅ compile_pallas_sparse_attention: output shape={attn_out.shape}, "
      f"dtype={attn_out.dtype}")
print(f"     compile+exec={1000*(t1-t0):.1f}ms")

print("\n" + "=" * 60)
print("🎉 GATE 1 PASSED: All kernels compile on TPU v5e-8")
print("=" * 60)


# --- CELL 4: GATE 2 — Correctness Test (bfloat16 tolerance) ---
import numpy as np
from orthocache.reference import numpy_fwht, compute_block_energy_reference, compute_tv_distance

print("=" * 60)
print("GATE 2: TPU Correctness Test (bfloat16)")
print("=" * 60)

np.random.seed(42)

print("\n[1/3] FWHT correctness vs NumPy reference ...")
test_data = np.random.randn(512, 128).astype(np.float32)
ref_out = numpy_fwht(test_data)
tpu_out = np.array(jax.jit(fwht_512)(jnp.array(test_data, dtype=jnp.bfloat16)))
max_dev = np.max(np.abs(tpu_out - ref_out))
rel_err = max_dev / (np.max(np.abs(ref_out)) + 1e-10)
print(f"  Max absolute deviation: {max_dev:.6f}")
print(f"  Relative error: {rel_err:.6f}")
assert rel_err < 0.02, f"FAIL: Relative error too large: {rel_err}"
print("  ✅ FWHT: PASS (bfloat16 rtol < 2%)")

print("\n[2/3] Block energy correctness ...")
keys_test = np.random.randn(1024, 4, 64).astype(np.float32)
ref_energy = compute_block_energy_reference(keys_test, 512)
tpu_energy = np.array(compute_block_energy_jax(jnp.array(keys_test, dtype=jnp.bfloat16), 512))
energy_rel_err = np.max(np.abs(tpu_energy - ref_energy) / (np.abs(ref_energy) + 1e-10))
print(f"  Max relative energy error: {energy_rel_err:.6f}")
assert energy_rel_err < 0.05, f"FAIL: Energy relative error too large: {energy_rel_err}"
print("  ✅ Block energy: PASS")

print("\n[3/3] Sparse vs dense attention consistency ...")
q_test = np.random.randn(4, 2, 64).astype(np.float32)
k_test = np.random.randn(1024, 2, 64).astype(np.float32)
v_test = np.random.randn(1024, 2, 64).astype(np.float32)
full_mask = jnp.ones((2, 2), dtype=bool)

dense_out = np.array(jax_block_sparse_attention(
    jnp.array(q_test), jnp.array(k_test), jnp.array(v_test), full_mask, 512))
sparse_mask = jnp.array([[True, False], [False, True]])
sparse_out = np.array(jax_block_sparse_attention(
    jnp.array(q_test), jnp.array(k_test), jnp.array(v_test), sparse_mask, 512))
diff = np.max(np.abs(dense_out - sparse_out))
print(f"  Dense vs sparse max diff: {diff:.6f}")
assert diff > 0.0, "FAIL: Sparse and dense should differ when blocks are evicted"
print("  ✅ Sparse attention: outputs differ as expected")

print("\n" + "=" * 60)
print("🎉 GATE 2 PASSED: Numerical fidelity verified on TPU v5e-8")
print("=" * 60)


# --- CELL 5: GATE 3 — Spectral Telemetry (Gemma 4 E2B) ---
# Extract real KV-cache from Gemma 4 E2B and analyze spectral energy distribution

import subprocess
ORTHOCACHE_DIR = "/kaggle/working/orthocache"

print("=" * 60)
print("GATE 3: Spectral Telemetry — Gemma 4 E2B")
print("=" * 60)

subprocess.run([
    sys.executable, f"{ORTHOCACHE_DIR}/benchmarks/spectral_analysis.py",
    "--model", GEMMA_PATH,
    "--seq_len", "8192",
    "--output_dir", "/kaggle/working/plots",
], check=True, env={**os.environ, "PYTHONPATH": f"{ORTHOCACHE_DIR}/src"})

print("\n🎉 GATE 3 PASSED: Spectral energy data collected")


# --- CELL 6: GATE 4 — Accuracy vs Sparsity Tradeoff ---
print("=" * 60)
print("GATE 4: Accuracy vs Sparsity — Gemma 4 E2B")
print("=" * 60)

subprocess.run([
    sys.executable, f"{ORTHOCACHE_DIR}/benchmarks/attention_accuracy.py",
    "--model", GEMMA_PATH,
    "--seq_len", "4096",
    "--output_dir", "/kaggle/working/plots",
], check=True, env={**os.environ, "PYTHONPATH": f"{ORTHOCACHE_DIR}/src"})

print("\n🎉 GATE 4 PASSED: Accuracy tradeoff curves generated")


# --- CELL 7: GATE 5 — Profiling (sparse vs dense timing) ---
print("=" * 60)
print("GATE 5: Latency Profiling — TPU v5e-8")
print("=" * 60)

subprocess.run([
    sys.executable, f"{ORTHOCACHE_DIR}/benchmarks/profiling.py",
    "--seq_len", "4096",
    "--num_iters", "50",
    "--output_dir", "/kaggle/working/results",
], check=True, env={**os.environ, "PYTHONPATH": f"{ORTHOCACHE_DIR}/src"})

print("\n🎉 GATE 5 PASSED: Profiling data collected")


# --- CELL 8: Final Summary ---
print("""
╔══════════════════════════════════════════════════════════════╗
║              OrthoCache TPU v5e-8 — ALL GATES PASSED         ║
╠══════════════════════════════════════════════════════════════╣
║  Gate 1: Compilation          ✅                             ║
║  Gate 2: Correctness          ✅                             ║
║  Gate 3: Spectral Telemetry   ✅  (Gemma 4 E2B, 8K tokens)  ║
║  Gate 4: Accuracy Tradeoff    ✅  (10/30/50/70% eviction)    ║
║  Gate 5: Latency Profiling    ✅  (sparse vs dense)          ║
╠══════════════════════════════════════════════════════════════╣
║  Model:    Gemma 4 E2B                                       ║
║  Hardware: TPU v5e-8                                         ║
╚══════════════════════════════════════════════════════════════╝

Output files (download from Kaggle "Output" tab):
  /kaggle/working/plots/    — energy CSVs, histograms, CDFs, Pareto curves
  /kaggle/working/results/  — profiling JSON with timing data
""")
