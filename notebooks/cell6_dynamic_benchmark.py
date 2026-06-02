"""Phase D Benchmark: dynamic while_loop attention vs Pallas unrolled loop.

Paste this as a new cell in your Kaggle notebook after Cell 2 (clone repo).
This tests whether jax.lax.while_loop achieves true dynamic loop elision
on TPU v5e-8.
"""

# %% Cell 6: Phase D — while_loop Kernel Benchmark on TPU
# =========================================================

import time
import jax
import jax.numpy as jnp
from orthocache.sparse_attention import compile_pallas_sparse_attention
from orthocache.dynamic_attention import dynamic_compact_attention, dynamic_multihead_attention
from orthocache.compaction import stream_compact

BLOCK_SIZE = 512
SEQ_LEN_K = 32768
NUM_HEADS = 16
HEAD_DIM = 256
NUM_BLOCKS = SEQ_LEN_K // BLOCK_SIZE
NUM_ITERS = 20
WARMUP = 3

key = jax.random.PRNGKey(42)
q = jax.random.normal(key, (1, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16) / jnp.sqrt(HEAD_DIM)
keys = jax.random.normal(key, (SEQ_LEN_K, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
values = jax.random.normal(key, (SEQ_LEN_K, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16)

def make_mask(eviction_pct):
    n_evict = int(NUM_BLOCKS * eviction_pct / 100)
    mask = jnp.ones((NUM_BLOCKS, NUM_HEADS), dtype=jnp.bool_)
    if n_evict > 0:
        mask = mask.at[-n_evict:, :].set(False)
    return mask

def bench(label, fn, num_iters=NUM_ITERS, warmup=WARMUP):
    for _ in range(warmup):
        out = fn()
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = fn()
        if hasattr(out, 'block_until_ready'):
            out.block_until_ready()
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / num_iters * 1000
    print(f"  {label}: {avg_ms:.3f} ms")
    return avg_ms

print("=" * 70)
print("PHASE D: while_loop vs Pallas BENCHMARK (TPU v5e-8)")
print(f"Shape: Q=(1,{NUM_HEADS},{HEAD_DIM}), KV=({SEQ_LEN_K},{NUM_HEADS},{HEAD_DIM})")
print(f"Blocks: {NUM_BLOCKS}, Block size: {BLOCK_SIZE}")
print("=" * 70)

# --- Pallas Unrolled Loop (Phase A/B kernel) ---
print("\n>>> PALLAS KERNEL (unrolled for loop):")
pallas_results = {}
for pct in [0, 50, 90]:
    mask = make_mask(pct)
    fn = jax.jit(lambda m=mask: compile_pallas_sparse_attention(q, keys, values, m, BLOCK_SIZE))
    ms = bench(f"pallas_evict_{pct}%", fn)
    pallas_results[pct] = ms

# --- Dynamic while_loop Kernel (Phase D) ---
print("\n>>> DYNAMIC while_loop KERNEL (Phase D):")
dynamic_results = {}
for pct in [0, 50, 90]:
    mask = make_mask(pct)
    fn = jax.jit(lambda m=mask: dynamic_multihead_attention(q, keys, values, m, block_size=BLOCK_SIZE))
    ms = bench(f"dynamic_evict_{pct}%", fn)
    dynamic_results[pct] = ms

# --- Correctness Check ---
print("\n>>> CORRECTNESS CHECK (0% eviction, dense baseline):")
mask_full = make_mask(0)
pallas_out = compile_pallas_sparse_attention(q, keys, values, mask_full, BLOCK_SIZE)
dynamic_out = dynamic_multihead_attention(q, keys, values, mask_full, block_size=BLOCK_SIZE)

# Dense reference
scale = jnp.sqrt(jnp.float32(HEAD_DIM))
logits = jnp.einsum('qhd,khd->qkh', q.astype(jnp.float32), keys.astype(jnp.float32)) / scale
weights = jax.nn.softmax(logits, axis=1)
dense_out = jnp.einsum('qkh,khd->qhd', weights, values.astype(jnp.float32))

err_pallas = jnp.max(jnp.abs(pallas_out.astype(jnp.float32) - dense_out))
err_dynamic = jnp.max(jnp.abs(dynamic_out.astype(jnp.float32) - dense_out))
print(f"  Pallas vs Dense:  max abs error = {err_pallas:.6f}")
print(f"  Dynamic vs Dense: max abs error = {err_dynamic:.6f}")

# --- Summary ---
print("\n" + "=" * 70)
print("PHASE D SUMMARY")
print("=" * 70)
print(f"{'Eviction %':<12} | {'Pallas (ms)':<14} | {'Dynamic (ms)':<14} | {'Δτ (ms)':<10} | {'Speedup':<10}")
print("-" * 65)
for pct in [0, 50, 90]:
    p = pallas_results[pct]
    d = dynamic_results[pct]
    delta = p - d
    speedup = p / d if d > 0 else float('inf')
    print(f"{pct:<12} | {p:<14.3f} | {d:<14.3f} | {delta:<10.3f} | {speedup:<9.2f}x")

print("\n>>> KEY QUESTION: Does the Dynamic kernel's latency DECREASE")
print("    proportionally with eviction rate?")
d0 = dynamic_results[0]
d50 = dynamic_results[50]
d90 = dynamic_results[90]
print(f"    0% → 50%: {d50/d0*100:.1f}% of baseline (ideal: 50%)")
print(f"    0% → 90%: {d90/d0*100:.1f}% of baseline (ideal: 10%)")
