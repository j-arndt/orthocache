"""Phase D.7: Vectorized Indirect Attention — vmap + fori_loop + dynamic_slice.

Tests whether vmap over heads erases the 0% eviction tax by:
1. Merging per-head DMA descriptors into wide vector-aligned bursts
2. Packing the MXU operands across heads for optimal systolic array density
3. Hiding per-iteration DMA latency behind cross-head parallelism

Compares:
  - D.5 baseline: fori_loop with multi-head einsum inside loop body
  - D.7 vmap-heads: single-head fori_loop vmapped over NH heads
  - D.7 vmap-batch: batch + heads vmapped (B=4)

Paste this entire cell into a Kaggle TPU v5 notebook.
"""
import jax
import jax.numpy as jnp
from functools import partial
import time

BS = 512; HD = 128; NH = 4; WARMUP = 10; REPS = 30

# ============================================================
# Data & utilities
# ============================================================
def make_data(sl, batch=1):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    return (
        jax.random.normal(k1, (batch, 1, NH, HD), dtype=jnp.bfloat16),
        jax.random.normal(k2, (batch, sl, NH, HD), dtype=jnp.bfloat16),
        jax.random.normal(k3, (batch, sl, NH, HD), dtype=jnp.bfloat16),
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
    return jnp.argsort(keys, stable=True), int(jnp.sum(mask))

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
# CORE: Single-head indirection kernel
# ============================================================
def single_head_indirect_attn(q, k_cache, v_cache, active_indices, num_active):
    """Single head, single batch element.
    q: (1, HD), k_cache: (SL, HD), v_cache: (SL, HD)
    active_indices: (M,) int32, num_active: scalar int
    """
    seq_q, head_dim = q.shape
    scale = jnp.sqrt(jnp.float32(head_dim))

    m_init = jnp.full((seq_q,), -1e30, dtype=jnp.float32)
    l_init = jnp.zeros((seq_q,), dtype=jnp.float32)
    o_init = jnp.zeros((seq_q, head_dim), dtype=jnp.float32)

    def body(i, carry):
        m_prev, l_prev, o_prev = carry
        real_idx = active_indices[i]

        k_block = jax.lax.dynamic_slice(k_cache, (real_idx * BS, 0), (BS, head_dim))
        v_block = jax.lax.dynamic_slice(v_cache, (real_idx * BS, 0), (BS, head_dim))

        logits = jnp.einsum('qd,kd->qk',
                            q.astype(jnp.float32),
                            k_block.astype(jnp.float32)) / scale

        m_block = jnp.max(logits, axis=1)
        m_next = jnp.maximum(m_prev, m_block)

        exp_logits = jnp.exp(logits - m_next[:, None])
        exp_prev = jnp.exp(m_prev - m_next)

        l_block = jnp.sum(exp_logits, axis=1)
        l_next = l_prev * exp_prev + l_block

        v_agg = jnp.einsum('qk,kd->qd', exp_logits, v_block.astype(jnp.float32))
        o_next = o_prev * exp_prev[:, None] + v_agg

        return m_next, l_next, o_next

    m_f, l_f, o_f = jax.lax.fori_loop(0, num_active, body, (m_init, l_init, o_init))
    return (o_f / l_f[:, None]).astype(jnp.bfloat16)


# ============================================================
# LAYER 1: vmap over heads (shared mask)
# ============================================================
# q: (1, NH, HD), k: (SL, NH, HD), v: (SL, NH, HD)
# active_indices: (M,) shared, num_active: scalar shared
_multi_head_indirect = jax.vmap(
    single_head_indirect_attn,
    in_axes=(1, 1, 1, None, None),
    out_axes=1
)

# ============================================================
# LAYER 2: vmap over batch (per-element mask)
# ============================================================
# q: (B, 1, NH, HD), k: (B, SL, NH, HD), v: (B, SL, NH, HD)
# active_indices: (B, M), num_active: scalar (same for all — bench uses uniform eviction)
_batch_multi_head_indirect = jax.vmap(
    _multi_head_indirect,
    in_axes=(0, 0, 0, 0, None),
    out_axes=0
)

# ============================================================
# JIT wrappers (num_active is static for optimal loop compilation)
# ============================================================
@partial(jax.jit, static_argnums=(4,))
def vmap_heads_attn(q, k, v, active_indices, num_active):
    """B=1: q (1, NH, HD), k (SL, NH, HD), v (SL, NH, HD)"""
    return _multi_head_indirect(q, k, v, active_indices, num_active)

@partial(jax.jit, static_argnums=(4,))
def vmap_batch_attn(q, k, v, active_indices, num_active):
    """B>1: q (B, 1, NH, HD), k (B, SL, NH, HD), v (B, SL, NH, HD)"""
    return _batch_multi_head_indirect(q, k, v, active_indices, num_active)


# ============================================================
# D.5 baseline (for comparison — multi-head einsum inside loop)
# ============================================================
@jax.jit
def d5_loop_attn(q, k, v, active_indices, num_active):
    """Phase D.5: fori_loop with multi-head einsum inside."""
    seq_q, num_heads, head_dim = q.shape
    scale = jnp.sqrt(jnp.float32(head_dim))
    m_init = jnp.full((seq_q, num_heads), -1e30, dtype=jnp.float32)
    l_init = jnp.zeros((seq_q, num_heads), dtype=jnp.float32)
    o_init = jnp.zeros((seq_q, num_heads, head_dim), dtype=jnp.float32)

    def body(i, carry):
        m_prev, l_prev, o_prev = carry
        real_idx = active_indices[i]
        k_block = jax.lax.dynamic_slice(k, (real_idx * BS, 0, 0), (BS, num_heads, head_dim))
        v_block = jax.lax.dynamic_slice(v, (real_idx * BS, 0, 0), (BS, num_heads, head_dim))
        logits = jnp.einsum('qhd,khd->qkh', q.astype(jnp.float32), k_block.astype(jnp.float32)) / scale
        m_block = jnp.max(logits, axis=1)
        m_next = jnp.maximum(m_prev, m_block)
        exp_logits = jnp.exp(logits - m_next[:, None, :])
        exp_prev = jnp.exp(m_prev - m_next)
        l_block = jnp.sum(exp_logits, axis=1)
        l_next = l_prev * exp_prev + l_block
        v_agg = jnp.einsum('qkh,khd->qhd', exp_logits, v_block.astype(jnp.float32))
        o_next = o_prev * exp_prev[:, :, None] + v_agg
        return m_next, l_next, o_next

    m_f, l_f, o_f = jax.lax.fori_loop(0, num_active, body, (m_init, l_init, o_init))
    return (o_f / l_f[:, :, None]).astype(jnp.bfloat16)


# Dense baseline
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
print(f"Method: vmap(single_head_fori_loop) over heads + batch\n")

print("=== Correctness check (16K, 50% eviction) ===")
q_c1, k_c1, v_c1 = make_data(16384, batch=1)
q_1 = q_c1[0]; k_1 = k_c1[0]; v_1 = v_c1[0]  # squeeze batch dim
m_c = make_mask(32, 50)
aidx_c, nact_c = stream_compact(m_c)

out_d5   = d5_loop_attn(q_1, k_1, v_1, aidx_c, nact_c)
out_vmap = vmap_heads_attn(q_1, k_1, v_1, aidx_c, nact_c)

err = jnp.max(jnp.abs(out_d5.astype(jnp.float32) - out_vmap.astype(jnp.float32)))
print(f"  D.5 loop vs vmap-heads max err: {err:.6f}")
assert err < 0.01, f"Output mismatch: {err}"
print("  PASSED\n")

# Batch correctness
print("=== Batch correctness (B=4, 16K, 50%) ===")
q_b4, k_b4, v_b4 = make_data(16384, batch=4)
aidx_b4 = jnp.broadcast_to(aidx_c[None, :], (4, aidx_c.shape[0]))
out_batch = vmap_batch_attn(q_b4, k_b4, v_b4, aidx_b4, nact_c)
out_single = vmap_heads_attn(q_b4[0], k_b4[0], v_b4[0], aidx_c, nact_c)
err_b = jnp.max(jnp.abs(out_batch[0].astype(jnp.float32) - out_single.astype(jnp.float32)))
print(f"  batch[0] vs single max err: {err_b:.6f}")
assert err_b < 0.01, f"Batch mismatch: {err_b}"
print("  PASSED\n")

# ============================================================
# Benchmark: B=1 comparison (D.5 vs vmap-heads vs dense)
# ============================================================
print("=" * 70)
print("  SECTION 1: B=1 — D.5 loop vs vmap-heads vs dense/predicated")
print("=" * 70)

results_b1 = {}
for seq in [8192, 16384, 32768, 65536]:
    nb = seq // BS
    q_b, k_b, v_b = make_data(seq, batch=1)
    q_1 = q_b[0]; k_1 = k_b[0]; v_1 = v_b[0]
    print(f"\n{'='*70}")
    print(f"  {seq} tokens ({nb} blocks)")
    print(f"{'='*70}")

    t_dense = bench("dense", lambda: dense_attn(q_1, k_1, v_1))

    for pct in [0, 50, 75, 90]:
        m = make_mask(nb, pct)
        aidx, nact = stream_compact(m)

        t_pred = bench(f"pred_{pct}%",
                      lambda m=m: predicated_attn(q_1, k_1, v_1, m))

        # Warmup D.5
        for _ in range(3): d5_loop_attn(q_1, k_1, v_1, aidx, nact).block_until_ready()
        t_d5 = bench(f"d5_loop_{pct}% (K={nact})",
                    lambda ai=aidx, n=nact: d5_loop_attn(q_1, k_1, v_1, ai, n))

        # Warmup vmap-heads
        for _ in range(3): vmap_heads_attn(q_1, k_1, v_1, aidx, nact).block_until_ready()
        t_vh = bench(f"vmap_heads_{pct}% (K={nact})",
                    lambda ai=aidx, n=nact: vmap_heads_attn(q_1, k_1, v_1, ai, n))

        dt_d5 = (t_pred - t_d5) / t_pred * 100
        dt_vh = (t_pred - t_vh) / t_pred * 100
        print(f"    D.5:  Δτ = {dt_d5:+.1f}%  ({t_pred/t_d5:.2f}x)")
        print(f"    vmap: Δτ = {dt_vh:+.1f}%  ({t_pred/t_vh:.2f}x)")
        results_b1[(seq, pct)] = {
            'dense': t_dense, 'pred': t_pred,
            'd5': t_d5, 'vmap': t_vh, 'K': nact
        }

# ============================================================
# Benchmark: B=4 batched
# ============================================================
print(f"\n{'='*70}")
print("  SECTION 2: B=4 — vmap-batch")
print(f"{'='*70}")

results_b4 = {}
for seq in [8192, 16384, 32768, 65536]:
    nb = seq // BS
    q_b4, k_b4, v_b4 = make_data(seq, batch=4)
    print(f"\n  {seq} tokens ({nb} blocks), B=4")

    for pct in [0, 50, 75, 90]:
        m = make_mask(nb, pct)
        aidx, nact = stream_compact(m)
        aidx_batched = jnp.broadcast_to(aidx[None, :], (4, aidx.shape[0]))

        for _ in range(3):
            vmap_batch_attn(q_b4, k_b4, v_b4, aidx_batched, nact).block_until_ready()
        t_vb = bench(f"  vmap_batch_{pct}% (K={nact})",
                    lambda ai=aidx_batched, n=nact: vmap_batch_attn(q_b4, k_b4, v_b4, ai, n))
        results_b4[(seq, pct)] = {'vmap_batch': t_vb, 'K': nact}


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print("  SUMMARY: B=1 — D.5 loop vs vmap-heads vs predicated")
print(f"{'='*70}")
print(f"{'seq':>6} | {'ev%':>4} | {'K':>4} | {'pred':>7} | {'D.5':>7} | {'vmap':>7} | {'D.5 Δτ':>7} | {'vmap Δτ':>7}")
print("-" * 72)
for (seq, pct), r in sorted(results_b1.items()):
    dt_d5 = (r['pred'] - r['d5']) / r['pred'] * 100
    dt_vh = (r['pred'] - r['vmap']) / r['pred'] * 100
    print(f"{seq:>6} | {pct:>4} | {r['K']:>4} | {r['pred']:>6.3f} | {r['d5']:>6.3f} | {r['vmap']:>6.3f} | {dt_d5:>6.1f}% | {dt_vh:>6.1f}%")

print(f"\n{'='*70}")
print("  SUMMARY: B=4 — vmap-batch (per-element ms)")
print(f"{'='*70}")
print(f"{'seq':>6} | {'ev%':>4} | {'K':>4} | {'total ms':>9} | {'per-elem':>9}")
print("-" * 50)
for (seq, pct), r in sorted(results_b4.items()):
    total = r['vmap_batch']
    per_elem = total / 4
    print(f"{seq:>6} | {pct:>4} | {r['K']:>4} | {total:>8.3f} | {per_elem:>8.3f}")
