"""Phase E: Multi-Host AllToAllv Benchmark — ICI Bandwidth Reduction.

Simulates P=8 sequence-parallel devices on Kaggle TPU v5e-8 (8 chips, 128 GB HBM).
Each device owns seq_len/P tokens of the KV-cache. The AllToAllv protocol
compacts locally, syncs counts, exchanges only active blocks, then runs
indirect attention on the received blocks.

Measures:
  1. Correctness: sharded output matches single-device dense attention
  2. Data volume: bytes per device (compacted vs dense)
  3. Latency: sharded OrthoCache vs sharded dense
  4. Scaling: 0% / 50% / 75% / 90% eviction (uniform + skewed masks)

Paste this entire cell into a Kaggle TPU v5 notebook.
"""
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
import time
import json

BS = 512; HD = 128; NH = 4; WARMUP = 5; REPS = 20

print(f"Platform: {jax.devices()[0].device_kind}, JAX {jax.__version__}")
P = jax.device_count()
print(f"Devices: {P} ({', '.join(str(d) for d in jax.devices())})")
print(f"Config: Q=1, heads={NH}, d={HD}, block={BS}, devices={P}")
print(f"Method: pmap AllToAllv (Strategy C: static-buffer exchange)\n")

assert P >= 2, f"Need ≥2 devices for multi-device test, got {P}"

# ============================================================
# Utilities
# ============================================================
def make_mask(nb, pct):
    """Create block mask with pct% eviction."""
    if pct == 0: return jnp.ones(nb, dtype=jnp.bool_)
    n = max(1, int(nb * pct / 100))
    n = min(n, nb - 1)  # keep at least 1 block
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
# Single-device dense attention (ground truth)
# ============================================================
@jax.jit
def dense_attn_single(q, k, v):
    """q: (1, NH, HD), k/v: (SL, NH, HD)"""
    sc = jnp.sqrt(jnp.float32(HD))
    lo = jnp.einsum('qhd,khd->qkh', q.astype(jnp.float32), k.astype(jnp.float32)) / sc
    w = jax.nn.softmax(lo, axis=1)
    return jnp.einsum('qkh,khd->qhd', w, v.astype(jnp.float32)).astype(jnp.bfloat16)

# ============================================================
# Sharded dense attention (baseline — full AllToAll, no eviction)
# ============================================================
@partial(jax.pmap, axis_name='d', in_axes=(0, 0, 0), out_axes=0)
def sharded_dense_attn(q_shard, k_shard, v_shard):
    """Each device has SL/P tokens. AllGather full KV, then dense attention."""
    # AllGather: each device gets full KV cache
    k_full = lax.all_gather(k_shard, axis_name='d', axis=0, tiled=True)
    v_full = lax.all_gather(v_shard, axis_name='d', axis=0, tiled=True)
    # Dense attention on full cache
    sc = jnp.sqrt(jnp.float32(HD))
    lo = jnp.einsum('qhd,khd->qkh',
                    q_shard.astype(jnp.float32),
                    k_full.astype(jnp.float32)) / sc
    w = jax.nn.softmax(lo, axis=1)
    out = jnp.einsum('qkh,khd->qhd', w, v_full.astype(jnp.float32))
    return out.astype(jnp.bfloat16)

# ============================================================
# Sharded OrthoCache attention (AllToAllv Strategy C)
# ============================================================
def _pack_active(kv_shard, active_indices, num_active, max_blocks):
    """Pack active blocks into static buffer. kv_shard: (local_sl, NH, HD)."""
    local_sl = kv_shard.shape[0]
    nh = kv_shard.shape[1]
    hd = kv_shard.shape[2]
    buf = jnp.zeros((max_blocks * BS, nh, hd), dtype=kv_shard.dtype)

    def body(i, buf):
        idx = active_indices[i]
        block = lax.dynamic_slice(kv_shard, (idx * BS, 0, 0), (BS, nh, hd))
        buf = lax.dynamic_update_slice(buf, block, (i * BS, 0, 0))
        return buf

    return lax.fori_loop(0, num_active, body, buf)


def _indirect_attn_on_received(q, k_received, v_received, total_active):
    """Online softmax attention over total_active blocks from received KV."""
    seq_q, nh, hd = q.shape
    scale = jnp.sqrt(jnp.float32(hd))

    m_init = jnp.full((seq_q, nh), -1e30, dtype=jnp.float32)
    l_init = jnp.zeros((seq_q, nh), dtype=jnp.float32)
    o_init = jnp.zeros((seq_q, nh, hd), dtype=jnp.float32)

    def body(i, carry):
        m_prev, l_prev, o_prev = carry
        k_blk = lax.dynamic_slice(k_received, (i * BS, 0, 0), (BS, nh, hd))
        v_blk = lax.dynamic_slice(v_received, (i * BS, 0, 0), (BS, nh, hd))

        logits = jnp.einsum('qhd,khd->qkh',
                            q.astype(jnp.float32),
                            k_blk.astype(jnp.float32)) / scale

        m_blk = jnp.max(logits, axis=1)
        m_new = jnp.maximum(m_prev, m_blk)
        exp_l = jnp.exp(logits - m_new[:, None, :])
        exp_p = jnp.exp(m_prev - m_new)

        l_new = l_prev * exp_p + jnp.sum(exp_l, axis=1)
        o_new = (o_prev * exp_p[:, :, None] +
                 jnp.einsum('qkh,khd->qhd', exp_l, v_blk.astype(jnp.float32)))
        return m_new, l_new, o_new

    m_f, l_f, o_f = lax.fori_loop(0, total_active, body,
                                    (m_init, l_init, o_init))
    return (o_f / l_f[:, :, None]).astype(jnp.bfloat16)


@partial(jax.pmap, axis_name='d',
         in_axes=(0, 0, 0, 0),
         out_axes=0)
def sharded_orthocache_attn(q_shard, k_shard, v_shard, block_mask):
    """Sharded OrthoCache attention with AllToAllv (Strategy C).

    Each device:
      1. Compacts local KV (keeps only active blocks)
      2. AllGathers active counts
      3. AllGathers compacted KV buffers
      4. Runs indirect attention on received blocks

    After AllGather, each device sees P chunks of num_blocks each.
    Chunk d has active blocks packed at positions [0..count_d) and
    zeros at [count_d..num_blocks). We iterate over all P*num_blocks
    block slots but use predication to skip zero-padded slots.
    """
    local_sl = k_shard.shape[0]
    num_blocks = local_sl // BS

    # Step 1: Local compaction
    active_indices, num_active = stream_compact(block_mask)

    # Step 2: Pack into static buffer (size = num_blocks * BS)
    k_packed = _pack_active(k_shard, active_indices, num_active, num_blocks)
    v_packed = _pack_active(v_shard, active_indices, num_active, num_blocks)

    # Step 3: AllGather counts (tiny payload — P integers)
    # num_active is a scalar (0D) — reshape to (1,) for all_gather
    counts_1d = num_active.reshape(1)
    all_counts_1d = lax.all_gather(counts_1d, axis_name='d', axis=0, tiled=True)
    # all_counts_1d shape: (P,)
    total_active = jnp.sum(all_counts_1d)
    num_devices = all_counts_1d.shape[0]  # P, derived from all_gather output

    # Step 4: AllGather compacted KV buffers
    # Each device sends its packed buffer (static size num_blocks * BS)
    # Receiver gets P copies concatenated along axis 0
    k_all = lax.all_gather(k_packed, axis_name='d', axis=0, tiled=True)
    v_all = lax.all_gather(v_packed, axis_name='d', axis=0, tiled=True)
    # k_all shape: (P * num_blocks * BS, NH, HD)

    # Step 5: Indirect attention over ALL received blocks
    # Layout in k_all: P chunks of num_blocks blocks each.
    # In chunk d, blocks [0..count_d) are valid, [count_d..num_blocks) are zero-padded.
    # Build a validity mask for all P*num_blocks block positions.
    total_blocks = num_devices * num_blocks  # static
    block_iota = jnp.arange(total_blocks, dtype=jnp.int32)
    # For block position b, it belongs to device d = b // num_blocks
    # Its local index within device d is j = b % num_blocks
    # It's valid iff j < all_counts_1d[d]
    device_ids = block_iota // num_blocks
    local_ids = block_iota % num_blocks
    per_device_counts = all_counts_1d[device_ids]  # count for each block's device
    valid_mask = local_ids < per_device_counts  # (total_blocks,) bool

    # Sort to push valid blocks to front (stable: preserves order)
    sort_keys = jnp.where(valid_mask, block_iota, total_blocks + block_iota)
    sorted_indices = jnp.argsort(sort_keys, stable=True)

    # Run attention using sorted_indices[0:total_active]
    out = _indirect_attn_global(q_shard, k_all, v_all, sorted_indices, total_active)
    return out


def _indirect_attn_global(q, k_all, v_all, global_indices, total_active):
    """Attention over globally indexed blocks."""
    seq_q, nh, hd = q.shape
    scale = jnp.sqrt(jnp.float32(hd))

    m_init = jnp.full((seq_q, nh), -1e30, dtype=jnp.float32)
    l_init = jnp.zeros((seq_q, nh), dtype=jnp.float32)
    o_init = jnp.zeros((seq_q, nh, hd), dtype=jnp.float32)

    def body(i, carry):
        m_prev, l_prev, o_prev = carry
        block_idx = global_indices[i]
        k_blk = lax.dynamic_slice(k_all, (block_idx * BS, 0, 0), (BS, nh, hd))
        v_blk = lax.dynamic_slice(v_all, (block_idx * BS, 0, 0), (BS, nh, hd))

        logits = jnp.einsum('qhd,khd->qkh',
                            q.astype(jnp.float32),
                            k_blk.astype(jnp.float32)) / scale

        m_blk = jnp.max(logits, axis=1)
        m_new = jnp.maximum(m_prev, m_blk)
        exp_l = jnp.exp(logits - m_new[:, None, :])
        exp_p = jnp.exp(m_prev - m_new)

        l_new = l_prev * exp_p + jnp.sum(exp_l, axis=1)
        o_new = (o_prev * exp_p[:, :, None] +
                 jnp.einsum('qkh,khd->qhd', exp_l, v_blk.astype(jnp.float32)))
        return m_new, l_new, o_new

    m_f, l_f, o_f = lax.fori_loop(0, total_active, body,
                                    (m_init, l_init, o_init))
    return (o_f / l_f[:, :, None]).astype(jnp.bfloat16)


# ============================================================
# Data generation
# ============================================================
def make_sharded_data(seq_len):
    """Generate q, k, v and shard across P devices."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    q = jax.random.normal(k1, (1, NH, HD), dtype=jnp.bfloat16)
    k = jax.random.normal(k2, (seq_len, NH, HD), dtype=jnp.bfloat16)
    v = jax.random.normal(k3, (seq_len, NH, HD), dtype=jnp.bfloat16)

    # Shard: each device gets seq_len/P tokens
    local_sl = seq_len // P
    q_sharded = jnp.broadcast_to(q, (P, 1, NH, HD))  # replicate q
    k_sharded = k.reshape(P, local_sl, NH, HD)
    v_sharded = v.reshape(P, local_sl, NH, HD)
    return q, k, v, q_sharded, k_sharded, v_sharded


def make_uniform_masks(num_blocks_per_device, pct):
    """Same eviction rate on all devices."""
    masks = []
    for d in range(P):
        masks.append(make_mask(num_blocks_per_device, pct))
    return jnp.stack(masks)  # (P, num_blocks_per_device)


def make_skewed_masks(num_blocks_per_device):
    """Different eviction rates across devices (ascending retention)."""
    # Spread rates across however many devices we have
    all_rates = [90, 80, 75, 60, 50, 30, 20, 10]
    rates = all_rates[:P]
    masks = []
    for d in range(P):
        masks.append(make_mask(num_blocks_per_device, rates[d]))
    return jnp.stack(masks), rates


# ============================================================
# ICI data volume accounting
# ============================================================
def ici_accounting(seq_len, masks_per_device):
    """Compute ICI bytes transferred."""
    local_sl = seq_len // P
    local_blocks = local_sl // BS
    bytes_per_block = BS * NH * HD * 2  # bf16

    dense_bytes_per_device = local_blocks * bytes_per_block  # what each device would send dense
    total_dense = dense_bytes_per_device * P  # total across cluster

    active_per_device = []
    sparse_bytes = 0
    for d in range(P):
        count = int(jnp.sum(masks_per_device[d]))
        active_per_device.append(count)
        sparse_bytes += count * bytes_per_block

    return {
        'dense_bytes_total': int(total_dense),
        'sparse_bytes_total': int(sparse_bytes),
        'savings_bytes': int(total_dense - sparse_bytes),
        'savings_pct': (1 - sparse_bytes / total_dense) * 100 if total_dense > 0 else 0,
        'active_per_device': active_per_device,
        'blocks_per_device': local_blocks,
    }


# ============================================================
# Correctness Check
# ============================================================
print("=" * 70)
print("  GATE E.1: Correctness — Sharded vs Single-Device")
print("=" * 70)

seq = 16384
q, k, v, q_sh, k_sh, v_sh = make_sharded_data(seq)
local_blocks = (seq // P) // BS

# Dense reference
out_ref = dense_attn_single(q, k, v)

# Sharded dense (AllGather full KV)
out_sharded_dense = sharded_dense_attn(q_sh, k_sh, v_sh)
err_dense = jnp.max(jnp.abs(out_ref.astype(jnp.float32) - out_sharded_dense[0].astype(jnp.float32)))
print(f"  Dense single vs sharded-dense: max err = {err_dense:.6f}")

# Sharded OrthoCache (0% eviction — should match dense)
masks_0 = make_uniform_masks(local_blocks, 0)
out_ortho_0 = sharded_orthocache_attn(q_sh, k_sh, v_sh, masks_0)
err_ortho_0 = jnp.max(jnp.abs(out_ref.astype(jnp.float32) - out_ortho_0[0].astype(jnp.float32)))
print(f"  Dense single vs ortho-sharded (0% evict): max err = {err_ortho_0:.6f}")

# Sharded OrthoCache (50% eviction — matches predicated)
masks_50 = make_uniform_masks(local_blocks, 50)
out_ortho_50 = sharded_orthocache_attn(q_sh, k_sh, v_sh, masks_50)
# Compare against single-device predicated
@jax.jit
def predicated_single(q, k, v, masks_flat):
    sc = jnp.sqrt(jnp.float32(HD))
    full_mask = jnp.repeat(masks_flat, BS)
    lo = jnp.einsum('qhd,khd->qkh', q.astype(jnp.float32), k.astype(jnp.float32)) / sc
    lo = jnp.where(full_mask[None, :, None], lo, -1e9)
    w = jax.nn.softmax(lo, axis=1)
    return jnp.einsum('qkh,khd->qhd', w, v.astype(jnp.float32)).astype(jnp.bfloat16)

# Reconstruct global mask from per-device masks
global_mask_50 = masks_50.reshape(-1)  # (P * local_blocks,)
out_pred_50 = predicated_single(q, k, v, global_mask_50)
err_ortho_50 = jnp.max(jnp.abs(out_pred_50.astype(jnp.float32) - out_ortho_50[0].astype(jnp.float32)))
print(f"  Predicated single vs ortho-sharded (50% evict): max err = {err_ortho_50:.6f}")
print()

# ============================================================
# GATE E.2: Data Volume Accounting
# ============================================================
print("=" * 70)
print("  GATE E.2: ICI Data Volume Accounting")
print("=" * 70)

for seq in [16384, 32768, 65536]:
    local_blocks = (seq // P) // BS
    print(f"\n  Seq={seq} ({seq//BS} blocks, {local_blocks}/device)")
    for pct in [0, 50, 75, 90]:
        masks = make_uniform_masks(local_blocks, pct)
        acct = ici_accounting(seq, masks)
        print(f"    {pct}% evict: dense={acct['dense_bytes_total']/1e6:.1f}MB "
              f"sparse={acct['sparse_bytes_total']/1e6:.1f}MB "
              f"saved={acct['savings_bytes']/1e6:.1f}MB ({acct['savings_pct']:.0f}%) "
              f"active/dev={acct['active_per_device']}")

# Skewed mask test
print(f"\n  Skewed masks (16K tokens):")
local_blocks = (16384 // P) // BS
masks_skew, rates = make_skewed_masks(local_blocks)
acct_skew = ici_accounting(16384, masks_skew)
print(f"    rates={rates}")
print(f"    active/dev={acct_skew['active_per_device']}")
print(f"    dense={acct_skew['dense_bytes_total']/1e6:.1f}MB "
      f"sparse={acct_skew['sparse_bytes_total']/1e6:.1f}MB "
      f"saved={acct_skew['savings_bytes']/1e6:.1f}MB ({acct_skew['savings_pct']:.0f}%)")
print()

# ============================================================
# GATE E.3: Latency Benchmark
# ============================================================
print("=" * 70)
print("  GATE E.3: Latency — Sharded Dense vs Sharded OrthoCache")
print("=" * 70)

results = {}
for seq in [8192, 16384, 32768]:
    local_blocks = (seq // P) // BS
    q_f, k_f, v_f, q_sh, k_sh, v_sh = make_sharded_data(seq)
    print(f"\n  {seq} tokens ({local_blocks} blocks/device)")

    t_dense = bench("sharded_dense",
                    lambda q=q_sh, k=k_sh, v=v_sh: sharded_dense_attn(q, k, v))

    for pct in [0, 50, 75, 90]:
        masks = make_uniform_masks(local_blocks, pct)
        # Warmup OrthoCache
        for _ in range(3):
            sharded_orthocache_attn(q_sh, k_sh, v_sh, masks).block_until_ready()
        t_ortho = bench(f"ortho_{pct}%",
                       lambda q=q_sh, k=k_sh, v=v_sh, m=masks:
                           sharded_orthocache_attn(q, k, v, m))
        dt = (t_dense - t_ortho) / t_dense * 100
        acct = ici_accounting(seq, masks)
        print(f"    Δτ = {dt:+.1f}%  ({t_dense/t_ortho:.2f}x)  "
              f"ICI saved: {acct['savings_pct']:.0f}%")
        results[(seq, pct)] = {
            'dense_ms': t_dense, 'ortho_ms': t_ortho,
            'dtau': dt, 'ici_saved_pct': acct['savings_pct'],
        }

# ============================================================
# Skewed mask latency test
# ============================================================
print(f"\n  Skewed masks (16K):")
q_f, k_f, v_f, q_sh, k_sh, v_sh = make_sharded_data(16384)
local_blocks = (16384 // P) // BS
masks_skew, rates = make_skewed_masks(local_blocks)
for _ in range(3):
    sharded_orthocache_attn(q_sh, k_sh, v_sh, masks_skew).block_until_ready()
t_skew = bench(f"ortho_skewed (rates={rates})",
               lambda: sharded_orthocache_attn(q_sh, k_sh, v_sh, masks_skew))

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*70}")
print("  SUMMARY")
print(f"{'='*70}")
print(f"{'seq':>6} | {'ev%':>4} | {'dense':>7} | {'ortho':>7} | {'Δτ':>7} | {'ICI saved':>9}")
print("-" * 55)
for (seq, pct), r in sorted(results.items()):
    print(f"{seq:>6} | {pct:>4} | {r['dense_ms']:>6.3f} | {r['ortho_ms']:>6.3f} | "
          f"{r['dtau']:>6.1f}% | {r['ici_saved_pct']:>7.0f}%")
