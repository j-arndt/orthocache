"""Phase D Benchmark: compacted vs predicated vs dense attention.

Paste this ENTIRE cell into a Kaggle TPU notebook. Zero imports from the repo.
Uses standard JAX ops only — no Pallas, no BlockSpec, no Mosaic.

This measures the ACTUAL Δτ from stream compaction: fewer blocks gathered
and matmul'd means fewer FLOPs means less wall-clock time.
"""

import jax
import jax.numpy as jnp
import time

BS = 512; HD = 128; NH = 4; WARMUP = 10; REPS = 30

# ============================================================================
# Data generation
# ============================================================================
def make_data(sl):
    k1,k2,k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    return (jax.random.normal(k1, (1,NH,HD), dtype=jnp.bfloat16),
            jax.random.normal(k2, (sl,NH,HD), dtype=jnp.bfloat16),
            jax.random.normal(k3, (sl,NH,HD), dtype=jnp.bfloat16))

def make_mask(nb, pct):
    if pct == 0: return jnp.ones((nb,NH), dtype=jnp.bool_)
    n = int(nb * pct / 100)
    m = jnp.ones(nb, dtype=jnp.bool_).at[:n].set(False)
    return jnp.broadcast_to(
        m[jax.random.permutation(jax.random.PRNGKey(42), nb)][:, None],
        (nb, NH))

def bench(label, fn):
    for _ in range(WARMUP): fn().block_until_ready()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn().block_until_ready()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort(); med = ts[len(ts)//2]
    print(f"  {label}: {med:.3f} ms")
    return med

# ============================================================================
# Stream compaction (pure JAX — emulates the HLO pass prefix sum)
# ============================================================================
def stream_compact(mask):
    if mask.ndim == 2: mask_1d = jnp.any(mask, axis=1)
    else: mask_1d = mask
    nb = mask_1d.shape[0]
    iota = jnp.arange(nb, dtype=jnp.int32)
    keys = jnp.where(mask_1d, iota, nb + iota)
    return jnp.argsort(keys, stable=True), jnp.sum(mask_1d).astype(jnp.int32)

# ============================================================================
# Three attention implementations
# ============================================================================

@jax.jit
def dense_attn(q, k, v):
    """Baseline: full dense attention, no masking."""
    sc = jnp.sqrt(jnp.float32(HD))
    lo = jnp.einsum('qhd,khd->qkh', q.astype(jnp.float32), k.astype(jnp.float32)) / sc
    w = jax.nn.softmax(lo, axis=1)
    return jnp.einsum('qkh,khd->qhd', w, v.astype(jnp.float32)).astype(jnp.bfloat16)

@jax.jit
def predicated_attn(q, k, v, mask):
    """v1: compute ALL blocks, mask logits. MXU still fires for evicted blocks."""
    sc = jnp.sqrt(jnp.float32(HD))
    lo = jnp.einsum('qhd,khd->qkh', q.astype(jnp.float32), k.astype(jnp.float32)) / sc
    token_mask = jnp.repeat(mask, BS, axis=0)  # (seq_k, NH)
    lo = jnp.where(token_mask[None,:,:], lo, -1e9)
    w = jax.nn.softmax(lo, axis=1)
    return jnp.einsum('qkh,khd->qhd', w, v.astype(jnp.float32)).astype(jnp.bfloat16)

@jax.jit
def compacted_attn(q, k, v, mask):
    """v2: stream compact → gather active blocks → dense attention on K blocks only."""
    nb = k.shape[0] // BS
    active_idx, num_active = stream_compact(mask)

    # Gather active blocks
    k_blocks = k.reshape(nb, BS, NH, HD)
    v_blocks = v.reshape(nb, BS, NH, HD)
    k_comp = k_blocks[active_idx]  # (nb, BS, NH, HD) — first K are valid
    v_comp = v_blocks[active_idx]
    k_flat = k_comp.reshape(nb * BS, NH, HD)
    v_flat = v_comp.reshape(nb * BS, NH, HD)

    # Validity mask: only first num_active * BS tokens are real
    valid = jnp.arange(nb * BS) < (num_active * BS)

    sc = jnp.sqrt(jnp.float32(HD))
    lo = jnp.einsum('qhd,khd->qkh', q.astype(jnp.float32), k_flat.astype(jnp.float32)) / sc
    lo = jnp.where(valid[None,:,None], lo, -1e9)
    w = jax.nn.softmax(lo, axis=1)
    return jnp.einsum('qkh,khd->qhd', w, v_flat.astype(jnp.float32)).astype(jnp.bfloat16)

# ============================================================================
# Run benchmarks
# ============================================================================
print(f"Platform: {jax.devices()[0].device_kind}, JAX {jax.__version__}")
print(f"Config: Q=1, heads={NH}, d={HD}, block={BS}, warmup={WARMUP}, reps={REPS}")

results = {}
for seq in [8192, 16384, 32768, 65536]:
    nb = seq // BS
    q, k, v = make_data(seq)
    print(f"\n{'='*70}")
    print(f"  {seq} tokens ({nb} blocks)")
    print(f"{'='*70}")

    t_dense = bench("dense_0%", lambda: dense_attn(q, k, v))

    for pct in [0, 25, 50, 75, 90]:
        m = make_mask(nb, pct)
        t_pred = bench(f"predicated_{pct}%", lambda m=m: predicated_attn(q, k, v, m))
        t_comp = bench(f"compacted_{pct}%",  lambda m=m: compacted_attn(q, k, v, m))
        results[(seq, pct)] = {'dense': t_dense, 'pred': t_pred, 'comp': t_comp}

# Summary
print(f"\n{'='*70}")
print(f"  SUMMARY: Δτ = (predicated - compacted) / predicated")
print(f"{'='*70}")
print(f"{'seq':>6} | {'evict%':>6} | {'pred ms':>8} | {'comp ms':>8} | {'Δτ':>8} | {'speedup':>8}")
print("-" * 60)
for (seq, pct), r in sorted(results.items()):
    dt = (r['pred'] - r['comp']) / r['pred'] * 100
    spd = r['pred'] / r['comp'] if r['comp'] > 0 else 0
    print(f"{seq:>6} | {pct:>6} | {r['pred']:>8.3f} | {r['comp']:>8.3f} | {dt:>7.1f}% | {spd:>7.2f}x")
