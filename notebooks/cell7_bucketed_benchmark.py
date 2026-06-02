# %% Cell 7: BUCKETED PALLAS BENCHMARK
# =======================================
# The killer combo: stream compaction + Pallas with reduced block count.
# No while_loop overhead. No predication. Just fewer unrolled iterations.

import importlib
# Force reimport
for mod_name in ['orthocache.bucketed_attention', 'orthocache.compaction', 
                  'orthocache.sparse_attention', 'orthocache.dynamic_attention']:
    if mod_name in sys.modules:
        importlib.reload(sys.modules[mod_name])

import time
import jax
import jax.numpy as jnp
from orthocache.sparse_attention import compile_pallas_sparse_attention
from orthocache.bucketed_attention import bucketed_attention

BLOCK_SIZE = 512
NUM_HEADS = 16
HEAD_DIM = 256
NUM_ITERS = 20
WARMUP = 3

def make_mask(num_blocks, eviction_pct):
    n_evict = int(num_blocks * eviction_pct / 100)
    mask = jnp.ones((num_blocks, NUM_HEADS), dtype=jnp.bool_)
    if n_evict > 0:
        mask = mask.at[-n_evict:, :].set(False)
    return mask

def bench(label, fn, num_iters=NUM_ITERS, warmup=WARMUP):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple): out = out[0]
        out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(num_iters):
        out = fn()
        if isinstance(out, tuple): out = out[0]
        out.block_until_ready()
    ms = (time.perf_counter() - t0) / num_iters * 1000
    print(f"  {label}: {ms:.3f} ms")
    return ms

# ============================================================
# Test at multiple sequence lengths
# ============================================================
for SEQ_LEN_K in [32768, 65536, 131072]:
    NUM_BLOCKS = SEQ_LEN_K // BLOCK_SIZE
    
    key = jax.random.PRNGKey(42)
    q = jax.random.normal(key, (1, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16) / jnp.sqrt(HEAD_DIM)
    keys = jax.random.normal(key, (SEQ_LEN_K, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
    values = jax.random.normal(key, (SEQ_LEN_K, NUM_HEADS, HEAD_DIM), dtype=jnp.bfloat16)
    
    print("=" * 70)
    print(f"BUCKETED PALLAS vs PREDICATED PALLAS — {SEQ_LEN_K//1024}K context ({NUM_BLOCKS} blocks)")
    print("=" * 70)
    
    pallas_results = {}
    bucket_results = {}
    
    for pct in [0, 25, 50, 75, 90]:
        mask = make_mask(NUM_BLOCKS, pct)
        
        # Predicated Pallas (existing kernel, flat latency)
        pf = lambda m=mask: compile_pallas_sparse_attention(q, keys, values, m, BLOCK_SIZE)
        pallas_results[pct] = bench(f"pallas_{pct}%", pf)
        
        # Bucketed Pallas (compaction + reduced block count)
        bf = lambda m=mask: bucketed_attention(q, keys, values, m, block_size=BLOCK_SIZE)
        bucket_results[pct] = bench(f"bucket_{pct}%", bf)
    
    # Summary table
    print(f"\n{'Evict %':<10} | {'Pallas (ms)':<14} | {'Bucketed (ms)':<14} | {'Bucket':<8} | {'Winner'}")
    print("-" * 65)
    for pct in [0, 25, 50, 75, 90]:
        n_active = int(NUM_BLOCKS * (1 - pct/100))
        # Compute bucket
        bkt = 1
        for b in [1,2,4,8,16,32,64,128,256,512]:
            if b >= n_active:
                bkt = b
                break
        winner = "BUCKET ✓" if bucket_results[pct] < pallas_results[pct] else "Pallas"
        print(f"{pct:<10} | {pallas_results[pct]:<14.3f} | {bucket_results[pct]:<14.3f} | {bkt:<8} | {winner}")
    
    if pallas_results[0] > 0:
        print(f"\nPallas scaling:  0%→50%: {pallas_results[50]/pallas_results[0]*100:.1f}%")
        print(f"Bucketed scaling: 0%→50%: {bucket_results[50]/bucket_results[0]*100:.1f}%")
        print(f"Bucketed scaling: 0%→90%: {bucket_results[90]/bucket_results[0]*100:.1f}%")
    print()
