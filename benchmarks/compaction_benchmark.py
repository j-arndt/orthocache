import time
import argparse
import jax
import jax.numpy as jnp
from orthocache.pipeline import orthocache_forward

def benchmark(
    seq_len_q: int,
    seq_len_k: int,
    num_heads: int,
    head_dim: int,
    block_size: int,
    zeta_max: float,
    num_iters: int = 10,
    warmup_iters: int = 3
):
    print(f"--- OrthoCache Compaction Benchmark ---")
    print(f"Device: {jax.devices()[0].device_kind}")
    print(f"Shapes: Q=({seq_len_q}, {num_heads}, {head_dim}), K/V=({seq_len_k}, {num_heads}, {head_dim})")
    print(f"Block size: {block_size}, Zeta max: {zeta_max}")
    print(f"Iterations: {num_iters} (Warmup: {warmup_iters})\n")

    key = jax.random.PRNGKey(42)
    q_key, k_key, v_key = jax.random.split(key, 3)

    # Initialize synthetic tensors
    # Normalizing variance to avoid extreme softmax saturation
    q = jax.random.normal(q_key, (seq_len_q, num_heads, head_dim), dtype=jnp.bfloat16) / jnp.sqrt(head_dim)
    keys = jax.random.normal(k_key, (seq_len_k, num_heads, head_dim), dtype=jnp.bfloat16)
    values = jax.random.normal(v_key, (seq_len_k, num_heads, head_dim), dtype=jnp.bfloat16)

    modes = ['dense', 'sparse', 'compact']
    results = {}

    for mode in modes:
        print(f"Benchmarking mode: {mode.upper()}...")
        
        def run_forward():
            out, meta = orthocache_forward(
                q, keys, values,
                block_size=block_size,
                zeta_max=zeta_max,
                tau=None,  # Auto-compute tau
                mode=mode
            )
            return out, meta
            
        # Warmup
        for _ in range(warmup_iters):
            out, meta = run_forward()
            out.block_until_ready()
            
        # Benchmark
        start_time = time.perf_counter()
        for _ in range(num_iters):
            out, meta = run_forward()
            out.block_until_ready()
        
        end_time = time.perf_counter()
        avg_latency_ms = ((end_time - start_time) / num_iters) * 1000
        
        # Fetch metadata from the last iteration
        eviction_rate = meta.get('eviction_rate', 0.0) * 100
        
        results[mode] = {
            'latency_ms': avg_latency_ms,
            'eviction_rate': eviction_rate,
        }
        print(f"  Latency: {avg_latency_ms:.2f} ms | Eviction Rate: {eviction_rate:.1f}%")

    # Summary Table
    dense_lat = results['dense']['latency_ms']
    print("\n--- Summary ---")
    print(f"{'Mode':<10} | {'Latency (ms)':<15} | {'Speedup':<10} | {'Eviction %':<10} | {'Δτ (ms)':<10}")
    print("-" * 65)
    for mode in modes:
        lat = results[mode]['latency_ms']
        speedup = dense_lat / lat
        evict = results[mode]['eviction_rate']
        delta_tau = dense_lat - lat
        print(f"{mode:<10} | {lat:<15.2f} | {speedup:<9.2f}x | {evict:<10.1f} | {delta_tau:<10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OrthoCache Stream Compaction Benchmark")
    parser.add_argument("--seq_len_q", type=int, default=1, help="Query sequence length")
    parser.add_argument("--seq_len_k", type=int, default=32768, help="KV cache sequence length")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of heads (KV)")
    parser.add_argument("--head_dim", type=int, default=256, help="Head dimension")
    parser.add_argument("--block_size", type=int, default=512, help="Tokens per block")
    parser.add_argument("--zeta_max", type=float, default=5.0, help="Maximum spectral decay ratio")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    
    args = parser.parse_args()
    
    # We enforce JAX to use hardware if available (will fallback to CPU if not, but good for reporting)
    benchmark(
        seq_len_q=args.seq_len_q,
        seq_len_k=args.seq_len_k,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        block_size=args.block_size,
        zeta_max=args.zeta_max,
        num_iters=args.iters
    )
