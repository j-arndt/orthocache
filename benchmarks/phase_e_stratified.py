"""Phase E.2: Stratified AllGather via shard_map — True ICI Bandwidth Reduction.

Platform: TPU v5e-8 (8 chips, 128 GB HBM), JAX 0.10.1
Method: shard_map + lax.switch over 4 pre-compiled AllGather capacity buckets

Architecture:
  1. Each device compacts local KV blocks (fancy-index reorder)
  2. pmax consensus → global_k_max (nanosecond scalar sync)
  3. lax.switch selects bucket (25%, 50%, 75%, 100% capacity)
  4. Selected branch: slice[:target_k] → all_gather → masked einsum
  5. Output: (1, NH, HD) replicated — softmax is complete on each device

Key difference from Strategy C (pmap):
  - Slice happens BEFORE the collective → physically fewer bytes on ICI
  - Dense einsum (not fori_loop) → MXU-parallel attention
  - lax.switch with static shapes → zero recompilation across masks

Paste this entire cell into a Kaggle TPU v5e notebook.
"""

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as PS
from functools import partial
import time

# ============================================================
# Constants
# ============================================================
BS = 512    # Block size (tokens per block)
HD = 128    # Head dimension
NH = 4      # Attention heads (local per device)
WARMUP = 5
REPS = 20

print(f"Platform: {jax.devices()[0].device_kind}, JAX {jax.__version__}")
NDev = jax.device_count()
devices = jax.devices()[:NDev]
print(f"Devices: {NDev} ({', '.join(str(d) for d in devices)})")
print(f"Config: Q=1, heads={NH}, d={HD}, block={BS}, devices={NDev}")
print(f"Method: shard_map + Stratified AllGather (4 buckets)\n")

import numpy as np

mesh = Mesh(np.array(devices).reshape(-1), ('tp',))

# ============================================================
# Utilities
# ============================================================
def make_mask(nb, pct):
    """Create block eviction mask. True = active (kept)."""
    if pct == 0:
        return jnp.ones(nb, dtype=jnp.bool_)
    n_evict = max(1, int(nb * pct / 100))
    n_evict = min(n_evict, nb - 1)  # keep at least 1
    m = jnp.ones(nb, dtype=jnp.bool_).at[:n_evict].set(False)
    return m[jax.random.permutation(jax.random.PRNGKey(42), nb)]


def stream_compact(mask):
    """Sort active blocks to front. Returns (sorted_indices, num_active)."""
    nb = mask.shape[0]
    iota = jnp.arange(nb, dtype=jnp.int32)
    keys = jnp.where(mask, iota, nb + iota)
    return jnp.argsort(keys, stable=True), jnp.sum(mask).astype(jnp.int32)


def bench(label, fn):
    """Median-of-sorted latency benchmark."""
    for _ in range(WARMUP):
        fn().block_until_ready()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn().block_until_ready()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    med = ts[len(ts) // 2]
    return med


def make_data(sl):
    """Generate random Q, K, V tensors."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(k1, (1, NH, HD), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (sl, NH, HD), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (sl, NH, HD), dtype=jnp.bfloat16)
    return q, k, v


def prepare_inputs(nb_per_dev, pct):
    """Build global index/count arrays for shard_map sharding.

    Returns:
        global_indices: (NDev * nb_per_dev,) — sharded P('tp') → (nb_per_dev,) per device
        global_counts:  (NDev,)             — sharded P('tp') → (1,) per device
    """
    idx_list, cnt_list = [], []
    for d in range(NDev):
        mask = make_mask(nb_per_dev, pct)
        idx, n = stream_compact(mask)
        idx_list.append(idx)
        cnt_list.append(n.reshape(1))
    return jnp.concatenate(idx_list), jnp.concatenate(cnt_list)


# ============================================================
# Single-device references
# ============================================================
@jax.jit
def dense_single(q, k, v):
    """Full dense attention — gold reference."""
    sc = jnp.sqrt(jnp.float32(HD))
    lo = jnp.einsum('qhd,khd->qkh',
                     q.astype(jnp.float32), k.astype(jnp.float32)) / sc
    w = jax.nn.softmax(lo, axis=1)
    return jnp.einsum('qkh,khd->qhd', w,
                       v.astype(jnp.float32)).astype(jnp.bfloat16)


@jax.jit
def predicated_single(q, k, v, block_mask):
    """Masked attention — reference for eviction correctness."""
    sc = jnp.sqrt(jnp.float32(HD))
    tmask = jnp.repeat(block_mask, BS)
    lo = jnp.einsum('qhd,khd->qkh',
                     q.astype(jnp.float32), k.astype(jnp.float32)) / sc
    lo = jnp.where(tmask[None, :, None], lo, jnp.float32(-1e9))
    w = jax.nn.softmax(lo, axis=1)
    return jnp.einsum('qkh,khd->qhd', w,
                       v.astype(jnp.float32)).astype(jnp.bfloat16)


# ============================================================
# Sharded Dense Baseline (shard_map — full AllGather)
# ============================================================
def build_sharded_dense(mesh_):
    """Baseline: AllGather full K/V, dense einsum. No eviction."""

    @jax.jit
    @partial(shard_map, mesh=mesh_,
             in_specs=(PS(None, None, None),    # q: replicated (1, NH, HD)
                       PS('tp', None, None),     # k: seq-sharded (SL, NH, HD)
                       PS('tp', None, None)),    # v: seq-sharded
             out_specs=PS(None, None, None),      # output: replicated
             check_rep=False)
    def fn(q, k, v):
        k_all = jax.lax.all_gather(k, axis_name='tp', axis=0, tiled=True)
        v_all = jax.lax.all_gather(v, axis_name='tp', axis=0, tiled=True)
        sc = jnp.sqrt(jnp.float32(HD))
        lo = jnp.einsum('qhd,khd->qkh',
                         q.astype(jnp.float32),
                         k_all.astype(jnp.float32)) / sc
        w = jax.nn.softmax(lo, axis=1)
        return jnp.einsum('qkh,khd->qhd', w,
                           v_all.astype(jnp.float32)).astype(jnp.bfloat16)
    return fn


# ============================================================
# Stratified OrthoCache (shard_map + Bucketed AllGather)
# ============================================================
def build_stratified(mesh_, nb_per_dev):
    """Builds a JIT-compiled stratified AllGather attention kernel.

    Pre-compiles 4 discrete AllGather capacity profiles (b1..b4).
    At runtime, pmax consensus selects the tightest bucket that fits
    the global maximum active count. The selected branch:
      1. Slices K/V to target_k blocks (ICI savings)
      2. AllGathers the slice (physically smaller transfer)
      3. Builds validity mask from per-device counts
      4. Runs dense masked einsum (compute proportional to bucket)

    Args:
        mesh_: JAX Mesh object
        nb_per_dev: number of blocks per device (compile-time constant)

    Returns:
        JIT-compiled shard_map kernel
    """
    b1 = max(1, nb_per_dev // 4)
    b2 = max(1, nb_per_dev // 2)
    b3 = max(1, (nb_per_dev * 3) // 4)
    b4 = nb_per_dev

    # Deduplicate bucket values (at small nb, some may collide)
    buckets = []
    seen = set()
    for b in [b1, b2, b3, b4]:
        if b not in seen:
            buckets.append(b)
            seen.add(b)
    # Pad to exactly 4 entries (lax.switch needs fixed branch count)
    while len(buckets) < 4:
        buckets.append(buckets[-1])
    b1, b2, b3, b4 = buckets

    def make_branch(target_k):
        """Factory: creates one AllGather+Attention branch for a fixed capacity."""
        def branch(q, k_act, v_act, all_counts):
            # 1. Slice to bucket capacity (BEFORE the collective)
            k_sl = k_act[:target_k]   # (target_k, BS, NH, HD) — static shape
            v_sl = v_act[:target_k]

            # 2. AllGather: each device sends target_k blocks
            k_g = jax.lax.all_gather(k_sl, axis_name='tp', axis=0, tiled=True)
            v_g = jax.lax.all_gather(v_sl, axis_name='tp', axis=0, tiled=True)
            # Shape: (target_k * NDev, BS, NH, HD)

            # 3. Build block validity mask from per-device counts
            ntb = target_k * NDev
            bi = jnp.arange(ntb, dtype=jnp.int32)
            did = bi // target_k           # which device contributed this block
            lpos = bi % target_k           # local position within device's chunk
            sdid = jnp.minimum(did, NDev - 1)
            valid = (did < NDev) & (lpos < all_counts[sdid])
            tmask = jnp.repeat(valid, BS)  # token-level mask

            # 4. Dense masked einsum attention
            kf = k_g.reshape(-1, NH, HD)   # (ntb * BS, NH, HD)
            vf = v_g.reshape(-1, NH, HD)
            sc = jnp.sqrt(jnp.float32(HD))
            lo = jnp.einsum('qhd,khd->qkh',
                             q.astype(jnp.float32),
                             kf.astype(jnp.float32)) / sc
            lo = jnp.where(tmask[None, :, None], lo, jnp.float32(-1e9))
            w = jax.nn.softmax(lo, axis=1)
            return jnp.einsum('qkh,khd->qhd', w,
                               vf.astype(jnp.float32)).astype(jnp.bfloat16)
        return branch

    @jax.jit
    @partial(shard_map, mesh=mesh_,
             in_specs=(PS(None, None, None),    # q: replicated
                       PS('tp', None, None),     # k: seq-sharded
                       PS('tp', None, None),     # v: seq-sharded
                       PS('tp'),                 # active_indices: per-device (nb_per_dev,)
                       PS('tp')),                # num_active: per-device (1,)
             out_specs=PS(None, None, None),      # output: replicated
             check_rep=False)
    def kernel(q, k, v, active_indices, num_active):
        nb = k.shape[0] // BS

        # Reshape to block layout
        kb = k.reshape(nb, BS, NH, HD)
        vb = v.reshape(nb, BS, NH, HD)

        # Reindex: active blocks to front via fancy indexing (XLA gather)
        ka = kb[active_indices]
        va = vb[active_indices]

        # Scalar consensus: global maximum active count
        na = num_active[0]  # squeeze (1,) → scalar
        gmax = jax.lax.pmax(na, axis_name='tp')

        # AllGather per-device counts for mask building
        ac = jax.lax.all_gather(num_active, axis_name='tp',
                                axis=0, tiled=True)  # (NDev,)

        # Bucket selection (same on all devices → same branch)
        bidx = jnp.where(gmax <= b1, 0,
               jnp.where(gmax <= b2, 1,
               jnp.where(gmax <= b3, 2, 3)))

        # Dispatch to pre-compiled branch
        return jax.lax.switch(bidx, [
            make_branch(b1),
            make_branch(b2),
            make_branch(b3),
            make_branch(b4),
        ], q, ka, va, ac)

    return kernel, (b1, b2, b3, b4)


# ════════════════════════════════════════════════════════════════
#   GATE E.1b: Correctness
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("  GATE E.1b: Correctness — shard_map Stratified AllGather")
print("=" * 70)

seq_c = 16384
q_c, k_c, v_c = make_data(seq_c)
nb_c = seq_c // (BS * NDev)

# Dense reference
out_ref = dense_single(q_c, k_c, v_c)

# Sharded dense baseline
sd_fn = build_sharded_dense(mesh)
out_sd = sd_fn(q_c, k_c, v_c)
err_sd = float(jnp.max(jnp.abs(out_ref.astype(jnp.float32) -
                                 out_sd.astype(jnp.float32))))
print(f"  Dense single vs shard_map dense: max err = {err_sd:.6f}")

# Stratified 0% eviction (should match dense exactly)
strat_fn, bkts = build_stratified(mesh, nb_c)
idx_0, cnt_0 = prepare_inputs(nb_c, 0)
out_s0 = strat_fn(q_c, k_c, v_c, idx_0, cnt_0)
err_s0 = float(jnp.max(jnp.abs(out_ref.astype(jnp.float32) -
                                 out_s0.astype(jnp.float32))))
print(f"  Dense single vs stratified (0% evict): max err = {err_s0:.6f}")

# Stratified 50% eviction (should match predicated reference)
idx_50, cnt_50 = prepare_inputs(nb_c, 50)
out_s50 = strat_fn(q_c, k_c, v_c, idx_50, cnt_50)
# Build global mask for predicated reference (same mask repeated per device)
gmask_50 = jnp.concatenate([make_mask(nb_c, 50) for _ in range(NDev)])
out_pred = predicated_single(q_c, k_c, v_c, gmask_50)
err_s50 = float(jnp.max(jnp.abs(out_pred.astype(jnp.float32) -
                                  out_s50.astype(jnp.float32))))
print(f"  Predicated single vs stratified (50% evict): max err = {err_s50:.6f}")
print()

# ════════════════════════════════════════════════════════════════
#   GATE E.2b: ICI Data Volume Accounting
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("  GATE E.2b: ICI Data Volume — Stratified Buckets vs Dense")
print("=" * 70)

for seq_v in [16384, 32768, 65536]:
    nb_v = seq_v // (BS * NDev)
    _, bk_v = build_stratified(mesh, nb_v)
    bv1, bv2, bv3, bv4 = bk_v

    # Dense: each device sends nb_v blocks via AllGather
    # Total ICI per device = nb_v * BS * NH * HD * dtype_bytes
    bytes_per_block = BS * NH * HD * 2  # bf16
    dense_ici = nb_v * bytes_per_block  # per-device send volume

    print(f"\n  Seq={seq_v} ({seq_v // BS} blocks, {nb_v}/dev, "
          f"buckets=[{bv1},{bv2},{bv3},{bv4}])")

    for pct in [0, 50, 75, 90]:
        mask = make_mask(nb_v, pct)
        _, na = stream_compact(mask)
        na = int(na)

        # Bucket selection logic (mirrors kernel)
        if na <= bv1:   bucket, bn = bv1, f"b1={bv1}"
        elif na <= bv2: bucket, bn = bv2, f"b2={bv2}"
        elif na <= bv3: bucket, bn = bv3, f"b3={bv3}"
        else:           bucket, bn = bv4, f"b4={bv4}"

        strat_ici = bucket * bytes_per_block
        saved_pct = (1 - strat_ici / dense_ici) * 100 if dense_ici > 0 else 0
        print(f"    {pct:>2}% evict: K={na} → {bn}  "
              f"dense={dense_ici * NDev / 1e6:.1f}MB  "
              f"strat={strat_ici * NDev / 1e6:.1f}MB  "
              f"ICI saved={saved_pct:.0f}%")
print()

# ════════════════════════════════════════════════════════════════
#   GATE E.3b: Latency Benchmark
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("  GATE E.3b: Latency — shard_map Dense vs Stratified OrthoCache")
print("=" * 70)

results = {}
for seq_b in [8192, 16384, 32768, 65536]:
    nb_b = seq_b // (BS * NDev)
    if nb_b < 1:
        print(f"\n  {seq_b} tokens: skipped (< 1 block/dev)")
        continue

    q_b, k_b, v_b = make_data(seq_b)
    d_fn = build_sharded_dense(mesh)
    o_fn, bk_b = build_stratified(mesh, nb_b)

    print(f"\n  {seq_b} tokens ({nb_b} blocks/dev, "
          f"buckets=[{bk_b[0]},{bk_b[1]},{bk_b[2]},{bk_b[3]}])")

    t_dense = bench("shard_dense", lambda: d_fn(q_b, k_b, v_b))
    print(f"  shard_dense: {t_dense:.3f} ms")

    for pct in [0, 50, 75, 90]:
        idx_b, cnt_b = prepare_inputs(nb_b, pct)
        # Capture loop vars
        t_o = bench(f"strat_{pct}%",
                    lambda _i=idx_b, _c=cnt_b: o_fn(q_b, k_b, v_b, _i, _c))

        mask_b = make_mask(nb_b, pct)
        _, na_b = stream_compact(mask_b)
        na_b = int(na_b)

        # Determine bucket
        if na_b <= bk_b[0]:   bkt = bk_b[0]
        elif na_b <= bk_b[1]: bkt = bk_b[1]
        elif na_b <= bk_b[2]: bkt = bk_b[2]
        else:                 bkt = bk_b[3]
        ici_pct = (1 - bkt / nb_b) * 100

        dt = (t_dense - t_o) / t_dense * 100
        spd = t_dense / t_o
        print(f"  strat_{pct}% (K={na_b}, bucket={bkt}): {t_o:.3f} ms  "
              f"Δτ = {dt:+.1f}%  ({spd:.2f}x)  ICI saved={ici_pct:.0f}%")

        results[(seq_b, pct)] = {
            'dense': t_dense, 'ortho': t_o, 'dtau': dt,
            'ici_saved': ici_pct, 'K': na_b, 'bucket': bkt
        }

# ════════════════════════════════════════════════════════════════
#   Summary Table
# ════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("  SUMMARY: shard_map Stratified AllGather")
print(f"{'=' * 70}")
print(f"{'seq':>6} | {'ev%':>4} | {'K':>3} | {'bkt':>3} | "
      f"{'dense':>8} | {'strat':>8} | {'Δτ':>8} | {'ICI↓':>5}")
print("-" * 65)
for (seq_s, pct_s), r in sorted(results.items()):
    print(f"{seq_s:>6} | {pct_s:>4} | {r['K']:>3} | {r['bucket']:>3} | "
          f"{r['dense']:>7.3f} | {r['ortho']:>7.3f} | "
          f"{r['dtau']:>+7.1f}% | {r['ici_saved']:>4.0f}%")
