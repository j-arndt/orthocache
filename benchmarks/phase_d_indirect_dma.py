"""Phase D.5: XLA Loop Indirection Benchmark — fori_loop + dynamic_slice.

NO Pallas. NO intermediate buffer. NO libtpu version dependency.
DMA engine fetches directly from original KV-cache via computed address.
Loop runs exactly num_active iterations (dynamic bound, one JIT).

Paste this entire cell into a Kaggle TPU v5 notebook.
"""
import jax
import jax.numpy as jnp
from functools import partial
import time

BS = 512; HD = 128; NH = 4; WARMUP = 10; REPS = 30

# ============================================================
# Data & utilities (identical to previous benchmarks)
# ============================================================
def make_data(sl):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    return (
        jax.random.normal(k1, (1, NH, HD), dtype=jnp.bfloat16),
        jax.random.normal(k2, (sl, NH, HD), dtype=jnp.bfloat16),
        jax.random.normal(k3, (sl, NH, HD), dtype=jnp.bfloat16),
    )

def make_mask(nb, pct):
    if pct == 0: return jnp.ones(nb, dtype=jnp.bool_)
    n = int(nb * pct / 100)
    m = jnp.ones(nb, dtype=jnp.bool_).at[:n].set(False)
    return m[jax.random.permutation(jax.random.PRNGKey(42), nb)]

def stream_compact(mask):
    nb = mask.shape[0]
    iota = jnp.arange(nb, dtype=jnp.int32)
    keys = jnp.where(mask, iota, nb + iota)
    return jnp.argsort(keys, stable=True), jnp.sum(mask).astype(jnp.int32)

def bench(label, fn):
    for _ in range(WARMUP): fn().block_until_ready()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn().block_until_ready()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort(); med = ts[len(ts)//2]
    print(f"  {label}: {med:.3f} ms")
    return med

# ============================================================
# XLA loop indirection: fori_loop + dynamic_slice
# ============================================================
@jax.jit
def indirect_loop_attn(q, k_cache, v_cache, active_indices, num_active):
    """OrthoCache attention via native XLA loop indirection.

    - fori_loop compiles to a hardware loop on TPU
    - dynamic_slice compiles to DMA with computed address
    - No intermediate buffer, no gather, no Pallas
    - num_active is a TRACED int — one JIT handles all eviction rates
    """
    seq_q, num_heads, head_dim = q.shape
    scale = jnp.sqrt(jnp.float32(head_dim))

    # Online softmax accumulators
    m_init = jnp.full((seq_q, num_heads), -1e30, dtype=jnp.float32)
    l_init = jnp.zeros((seq_q, num_heads), dtype=jnp.float32)
    o_init = jnp.zeros((seq_q, num_heads, head_dim), dtype=jnp.float32)

    def body(i, carry):
        m_prev, l_prev, o_prev = carry

        # Scalar register read: which block to fetch
        real_idx = active_indices[i]

        # ONE HBM TOUCH: DMA fetches directly from original cache
        k_block = jax.lax.dynamic_slice(
            k_cache,
            (real_idx * BS, 0, 0),
            (BS, num_heads, head_dim)
        )
        v_block = jax.lax.dynamic_slice(
            v_cache,
            (real_idx * BS, 0, 0),
            (BS, num_heads, head_dim)
        )

        # Logits: (seq_q, BS, NH)
        logits = jnp.einsum('qhd,khd->qkh',
                            q.astype(jnp.float32),
                            k_block.astype(jnp.float32)) / scale

        # Online softmax update
        m_block = jnp.max(logits, axis=1)              # (seq_q, NH)
        m_next = jnp.maximum(m_prev, m_block)

        exp_logits = jnp.exp(logits - m_next[:, None, :])
        exp_prev = jnp.exp(m_prev - m_next)

        l_block = jnp.sum(exp_logits, axis=1)           # (seq_q, NH)
        l_next = l_prev * exp_prev + l_block

        # Weighted V: (seq_q, NH, HD)
        v_agg = jnp.einsum('qkh,khd->qhd',
                           exp_logits,
                           v_block.astype(jnp.float32))
        o_next = o_prev * exp_prev[:, :, None] + v_agg

        return m_next, l_next, o_next

    m_f, l_f, o_f = jax.lax.fori_loop(0, num_active, body, (m_init, l_init, o_init))
    return (o_f / l_f[:, :, None]).astype(jnp.bfloat16)


# ============================================================
# Baselines (identical to previous benchmarks)
# ============================================================
@jax.jit
def dense_attn(q, k, v):
    sc = jnp.sqrt(jnp.float32(HD))
    lo = jnp.einsum('qhd,khd->qkh', q.astype(jnp.float32), k.astype(jnp.float32)) / sc
    w = jax.nn.softmax(lo, axis=1)
    return jnp.einsum('qkh,khd->qhd', w, v.astype(jnp.float32)).astype(jnp.bfloat16)

@jax.jit
def predicated_attn(q, k, v, mask):
    sc = jnp.sqrt(jnp.float32(HD))
    lo = jnp.einsum('qhd,khd->qkh', q.astype(jnp.float32), k.astype(jnp.float32)) / sc
    lo = jnp.where(jnp.repeat(mask, BS)[None, :, None], lo, -1e9)
    w = jax.nn.softmax(lo, axis=1)
    return jnp.einsum('qkh,khd->qhd', w, v.astype(jnp.float32)).astype(jnp.bfloat16)


# ============================================================
# Correctness check
# ============================================================
print(f"Platform: {jax.devices()[0].device_kind}, JAX {jax.__version__}")
print(f"Config: Q=1, heads={NH}, d={HD}, block={BS}")
print(f"Method: jax.lax.fori_loop + jax.lax.dynamic_slice (native XLA)\n")

print("=== Correctness check (16K, 50% eviction) ===")
q_c, k_c, v_c = make_data(16384)
m_c = make_mask(32, 50)
aidx_c, nact_c = stream_compact(m_c)

out_pred  = predicated_attn(q_c, k_c, v_c, m_c)
out_loop  = indirect_loop_attn(q_c, k_c, v_c, aidx_c, nact_c)

err = jnp.max(jnp.abs(out_pred.astype(jnp.float32) - out_loop.astype(jnp.float32)))
print(f"  predicated vs loop-indirect max err: {err:.6f}")
assert err < 0.05, f"Output mismatch: {err}"
print("  PASSED\n")

# ============================================================
# Benchmark
# ============================================================
results = {}
for seq in [8192, 16384, 32768, 65536]:
    nb = seq // BS
    q, k, v = make_data(seq)
    print(f"{'='*70}")
    print(f"  {seq} tokens ({nb} blocks)")
    print(f"{'='*70}")

    t_dense = bench("dense", lambda: dense_attn(q, k, v))

    for pct in [0, 50, 75, 90]:
        m = make_mask(nb, pct)
        t_pred = bench(f"pred_{pct}%", lambda m=m: predicated_attn(q, k, v, m))

        aidx, nact = stream_compact(m)

        t_loop = bench(f"loop_{pct}% (K={int(nact)})",
                      lambda ai=aidx, na=nact: indirect_loop_attn(q, k, v, ai, na))

        dt = (t_pred - t_loop) / t_pred * 100
        spd = t_pred / t_loop if t_loop > 0 else 0
        print(f"    -> Δτ = {dt:+.1f}%  ({spd:.2f}x)")
        results[(seq, pct)] = {
            'dense': t_dense, 'pred': t_pred,
            'loop': t_loop, 'K': int(nact)
        }

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print(f"  SUMMARY: XLA Loop Indirection vs Predicated")
print(f"  Key: 0% eviction should be ~1.00x (not 0.33x)")
print(f"{'='*70}")
print(f"{'seq':>6} | {'evict%':>6} | {'K':>4} | {'pred ms':>8} | {'loop ms':>8} | {'Δτ':>8} | {'speedup':>8}")
print("-" * 68)
for (seq, pct), r in sorted(results.items()):
    dt = (r['pred'] - r['loop']) / r['pred'] * 100
    spd = r['pred'] / r['loop'] if r['loop'] > 0 else 0
    print(f"{seq:>6} | {pct:>6} | {r['K']:>4} | {r['pred']:>8.3f} | {r['loop']:>8.3f} | {dt:>7.1f}% | {spd:>7.2f}x")
