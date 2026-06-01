"""OrthoCache Profiling Benchmark.

Profiles wall-clock execution time of dense attention versus OrthoCache
block-sparse attention at various eviction rates.  Uses ``jax.block_until_ready()``
for accurate timing on accelerator backends.

Outputs
-------
* Timing comparison table on stdout
* JSON results file → benchmarks/results/

Usage
-----
    python benchmarks/profiling.py
    python benchmarks/profiling.py --seq_len 8192 --num_heads 8 --head_dim 64 --num_iters 50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from orthocache.spectral_energy import compute_block_energy_jax, generate_threshold_mask
from orthocache.sparse_attention import jax_block_sparse_attention


BLOCK_SIZE = 512


# ---------------------------------------------------------------------------
# Attention implementations under test
# ---------------------------------------------------------------------------

def dense_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
) -> jax.Array:
    """Standard dense scaled-dot-product attention."""
    head_dim = q.shape[-1]
    logits = jnp.einsum("qhd,khd->qkh", q, k) / jnp.sqrt(head_dim)
    weights = jax.nn.softmax(logits, axis=1)
    return jnp.einsum("qkh,khd->qhd", weights, v)


def sparse_attention_with_eviction(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    eviction_rate: float,
) -> jax.Array:
    """OrthoCache pipeline: spectral energy → threshold → sparse attention."""
    energies = compute_block_energy_jax(k, block_size=BLOCK_SIZE)
    energies_np = np.asarray(energies)

    # Find threshold for the requested eviction rate
    threshold = float(np.percentile(energies_np.flatten(), eviction_rate * 100))
    block_mask = generate_threshold_mask(energies, epsilon=threshold)

    return jax_block_sparse_attention(q, k, v, block_mask, block_size=BLOCK_SIZE)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

def time_fn(fn, num_warmup: int, num_iters: int) -> dict[str, float]:
    """Time *fn* over *num_iters* measured iterations after *num_warmup* warm-ups.

    Uses ``jax.block_until_ready()`` to ensure async dispatch is flushed.
    """
    # Warm-up
    for _ in range(num_warmup):
        out = fn()
        jax.block_until_ready(out)

    # Measured iterations
    times: list[float] = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    arr = np.array(times) * 1000  # convert to milliseconds
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "num_iters": num_iters,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OrthoCache profiling: dense vs. sparse attention timing.",
    )
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="KV-cache sequence length (default: 4096). Must be a multiple of 512.")
    parser.add_argument("--query_len", type=int, default=16,
                        help="Number of query tokens (default: 16).")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads (default: 8).")
    parser.add_argument("--head_dim", type=int, default=64,
                        help="Dimension per head (default: 64).")
    parser.add_argument("--num_warmup", type=int, default=5,
                        help="Number of warm-up iterations (default: 5).")
    parser.add_argument("--num_iters", type=int, default=30,
                        help="Number of measured iterations (default: 30).")
    parser.add_argument("--output_dir", type=str, default="benchmarks/results",
                        help="Directory to write JSON results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_len = args.seq_len
    query_len = args.query_len
    num_heads = args.num_heads
    head_dim = args.head_dim

    assert seq_len % BLOCK_SIZE == 0, (
        f"seq_len ({seq_len}) must be a multiple of block size ({BLOCK_SIZE})"
    )

    print("=== OrthoCache Profiling Benchmark ===")
    print(f"  Backend    : {jax.default_backend()}")
    print(f"  Devices    : {jax.devices()}")
    print(f"  Seq length : {seq_len}")
    print(f"  Query len  : {query_len}")
    print(f"  Num heads  : {num_heads}")
    print(f"  Head dim   : {head_dim}")
    print(f"  Warm-up    : {args.num_warmup}")
    print(f"  Iterations : {args.num_iters}")
    print()

    # Synthetic KV-cache ---------------------------------------------------
    rng = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(rng, 3)

    keys = jax.random.normal(k1, (seq_len, num_heads, head_dim), dtype=jnp.float32)
    values = jax.random.normal(k2, (seq_len, num_heads, head_dim), dtype=jnp.float32)
    queries = jax.random.normal(k3, (query_len, num_heads, head_dim), dtype=jnp.float32)

    # Pre-materialise to avoid counting allocation in timing
    jax.block_until_ready(keys)
    jax.block_until_ready(values)
    jax.block_until_ready(queries)

    # Configurations to profile
    configs: list[dict[str, object]] = [
        {"label": "Dense attention", "eviction": None},
        {"label": "OrthoCache sparse (30% eviction)", "eviction": 0.30},
        {"label": "OrthoCache sparse (50% eviction)", "eviction": 0.50},
    ]

    all_results: list[dict[str, object]] = []

    for cfg in configs:
        label: str = cfg["label"]  # type: ignore[assignment]
        eviction = cfg["eviction"]

        if eviction is None:
            fn = lambda: dense_attention(queries, keys, values)
        else:
            ev = eviction  # capture for closure
            fn = lambda ev=ev: sparse_attention_with_eviction(queries, keys, values, ev)

        print(f"  Profiling: {label} …", end=" ", flush=True)
        stats = time_fn(fn, args.num_warmup, args.num_iters)
        print(f"median={stats['median_ms']:.2f} ms")

        result = {
            "label": label,
            "eviction_rate": eviction,
            "seq_len": seq_len,
            "query_len": query_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            **stats,
        }
        all_results.append(result)

    # JSON output ----------------------------------------------------------
    json_path = output_dir / "profiling_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table --------------------------------------------------------
    print()
    col_w = {
        "label": max(len(str(r["label"])) for r in all_results),
        "median": 10,
        "mean": 10,
        "std": 10,
        "min": 10,
        "p95": 10,
    }
    header = (
        f"{'Configuration':<{col_w['label']}}  "
        f"{'Median':>{col_w['median']}}  "
        f"{'Mean':>{col_w['mean']}}  "
        f"{'Std':>{col_w['std']}}  "
        f"{'Min':>{col_w['min']}}  "
        f"{'P95':>{col_w['p95']}}  "
        f"{'Speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    dense_median = all_results[0]["median_ms"]
    for r in all_results:
        speedup = dense_median / r["median_ms"] if r["median_ms"] > 0 else float("inf")
        print(
            f"{r['label']:<{col_w['label']}}  "
            f"{r['median_ms']:>{col_w['median']}.3f}  "
            f"{r['mean_ms']:>{col_w['mean']}.3f}  "
            f"{r['std_ms']:>{col_w['std']}.3f}  "
            f"{r['min_ms']:>{col_w['min']}.3f}  "
            f"{r['p95_ms']:>{col_w['p95']}.3f}  "
            f"{speedup:>7.2f}x"
        )

    print(f"\nResults written to {json_path.resolve()}")


if __name__ == "__main__":
    main()
