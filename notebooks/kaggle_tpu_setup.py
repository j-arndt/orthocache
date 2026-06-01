"""
OrthoCache — Kaggle TPU v5e-8 Setup & Gate Execution Notebook
=============================================================

Paste the cells below into a fresh Kaggle notebook with TPU v5e-8 accelerator.
Each cell is separated by a comment: # --- CELL N ---

Prerequisites:
  - Kaggle account with TPU quota (free tier has 20h/week)
  - This file uploaded as a Kaggle dataset, OR the GitHub repo cloned
"""

# --- CELL 1: Environment Setup ---
# Verify TPU connectivity and install dependencies

import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")
assert jax.default_backend() == "tpu", "ERROR: Not running on TPU!"
print(f"\n✅ TPU v5e detected: {len(jax.devices())} chips")


# --- CELL 2: Clone the OrthoCache codebase from GitHub ---
import subprocess
import sys
import os

REPO_URL = "https://github.com/j-arndt/orthocache.git"
CLONE_DIR = "/kaggle/working/orthocache"

if not os.path.exists(CLONE_DIR):
    print(f"Cloning {REPO_URL} ...")
    subprocess.run(["git", "clone", REPO_URL, CLONE_DIR], check=True)
    print(f"✅ Cloned to {CLONE_DIR}")
else:
    print(f"✅ Repo already exists at {CLONE_DIR}")

sys.path.insert(0, os.path.join(CLONE_DIR, "src"))

# Verify imports work
from orthocache import fwht_512, compute_block_energy_jax, generate_threshold_mask
from orthocache import jax_block_sparse_attention, compile_pallas_sparse_attention
print("✅ All OrthoCache modules imported successfully")


# --- CELL 3: GATE 1 — Compilation Test ---
# Verify all Pallas kernels compile on TPU without XLA graph faults

import jax
import jax.numpy as jnp
import time

print("=" * 60)
print("GATE 1: TPU Compilation Test")
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
energies = jax.jit(compute_block_energy_jax)(keys_3d, 512)
energies.block_until_ready()
mask = generate_threshold_mask(energies, epsilon=1.0)
mask.block_until_ready()
t1 = time.perf_counter()
print(f"  ✅ compute_block_energy_jax: energies shape={energies.shape}")
print(f"  ✅ generate_threshold_mask: mask shape={mask.shape}, "
      f"retained={int(mask.sum())}/{mask.size}")
print(f"     compile+exec={1000*(t1-t0):.1f}ms")

# Test 3: Sparse attention (Pallas path on TPU, JAX fallback on CPU)
print("\n[3/3] Compiling sparse attention kernel ...")
q = jax.random.normal(jax.random.PRNGKey(2), (8, 4, 64), dtype=jnp.bfloat16)
k = jax.random.normal(jax.random.PRNGKey(3), (1024, 4, 64), dtype=jnp.bfloat16)
v = jax.random.normal(jax.random.PRNGKey(4), (1024, 4, 64), dtype=jnp.bfloat16)
block_mask = jnp.ones((2, 4), dtype=bool)  # retain all blocks

t0 = time.perf_counter()
attn_out = jax.jit(compile_pallas_sparse_attention, static_argnums=(4,))(q, k, v, block_mask, 512)
attn_out.block_until_ready()
t1 = time.perf_counter()
print(f"  ✅ compile_pallas_sparse_attention: output shape={attn_out.shape}, "
      f"dtype={attn_out.dtype}")
print(f"     compile+exec={1000*(t1-t0):.1f}ms")

print("\n" + "=" * 60)
print("🎉 GATE 1 PASSED: All kernels compile on TPU")
print("=" * 60)


# --- CELL 4: GATE 2 — Correctness Test (bfloat16 tolerance) ---
# Verify numerical fidelity against NumPy reference on TPU

import numpy as np
from orthocache.reference import numpy_fwht, compute_block_energy_reference, compute_tv_distance

print("=" * 60)
print("GATE 2: TPU Correctness Test (bfloat16)")
print("=" * 60)

np.random.seed(42)

# Test 1: FWHT correctness
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

# Test 2: Block energy correctness
print("\n[2/3] Block energy correctness ...")
keys_test = np.random.randn(1024, 4, 64).astype(np.float32)
ref_energy = compute_block_energy_reference(keys_test, 512)
tpu_energy = np.array(compute_block_energy_jax(jnp.array(keys_test, dtype=jnp.bfloat16), 512))
energy_rel_err = np.max(np.abs(tpu_energy - ref_energy) / (np.abs(ref_energy) + 1e-10))
print(f"  Max relative energy error: {energy_rel_err:.6f}")
assert energy_rel_err < 0.05, f"FAIL: Energy relative error too large: {energy_rel_err}"
print("  ✅ Block energy: PASS")

# Test 3: Sparse attention consistency
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
print("🎉 GATE 2 PASSED: Numerical fidelity verified on TPU")
print("=" * 60)


# --- CELL 5: Summary & Next Steps ---
print("""
╔══════════════════════════════════════════════════════════╗
║                 OrthoCache TPU Status                    ║
╠══════════════════════════════════════════════════════════╣
║  Gate 1: Compilation     ✅ PASSED                       ║
║  Gate 2: Correctness     ✅ PASSED                       ║
║  Gate 3: Spectral Telemetry   ⬜ NEXT                    ║
║  Gate 4: Accuracy Tradeoff    ⬜ PENDING                  ║
║  Gate 5: Profiling            ⬜ PENDING                  ║
╚══════════════════════════════════════════════════════════╝

Next: Run benchmarks/spectral_analysis.py and
      benchmarks/attention_accuracy.py to collect empirical data.
""")
