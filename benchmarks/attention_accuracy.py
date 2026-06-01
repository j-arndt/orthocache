"""OrthoCache Attention Accuracy Benchmark.

Measures how block-sparse attention quality degrades as blocks are evicted at
increasing rates. For each eviction target the script:

1. Computes the full dense attention output.
2. Uses query-aware spectral-energy bounds to build a block mask.
3. Computes sparse attention via ``jax_block_sparse_attention``.
4. Measures TV distance (``compute_tv_distance``) and KL divergence.
5. Verifies the OrthoCache Truncation Bound:
       measured TV  ≤  |S^c| · exp(τ − z_max)

Outputs
-------
* Accuracy–efficiency Pareto curves (PNG)  → benchmarks/plots/
* Per-threshold results (CSV)              → benchmarks/plots/
* Bound-violation report on stdout
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from orthocache.spectral_energy import (
    compute_block_energy_jax,
    generate_threshold_mask,
    compute_query_aware_bounds,
    compute_query_aware_mask,
)
from orthocache.sparse_attention import jax_block_sparse_attention
from orthocache.reference import compute_tv_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 512


def _load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    """Load a HuggingFace causal-LM and tokenizer in bfloat16 on CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.eval()
    print(f"  Model loaded on CPU ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params)")
    return model, tokenizer


def _build_prompt(tokenizer: Any, seq_len: int) -> Any:
    """Build a token tensor of exactly *seq_len* diverse tokens to prevent degenerate uniform attention."""
    import torch
    import random

    # High-entropy vocabulary to produce peaked, non-uniform attention logits
    words = [
        "Walsh", "Hadamard", "transform", "concentrates", "spectral", "energy", "into", "small",
        "number", "coefficients", "Blocks", "with", "low", "can", "be", "evicted", "from", "KV-cache",
        "without", "measurable", "accuracy", "loss", "TPU", "Pallas", "kernel", "compilation",
        "acceleration", "speedup", "latency", "bandwidth", "memory", "bandwidth-bound", "collective",
        "AllToAll", "AllGather", "distributed", "partition", "Vector", "Memory", "VMEM", "HBM",
        "systolic", "array", "Matrix", "Multiply", "Unit", "MXU", "Vector", "Processing", "Unit", "VPU",
        "coordinate", "distribution", "incoherent", "isotropic", "outlier", "quantization", "gauge",
        "invariance", "fiber", "bundle", "connection", "curvature", "information", "geometry", "cotangent",
        "Hamiltonian", "dynamics", "Liouville", "theorem", "symplectic", "volume", "form", "measure",
        "thermodynamic", "expectation", "Boltzmann", "distribution", "temperature", "entropy", "divergence",
        "Total", "Variation", "distance", "softmax", "partition", "function", "exponential", "bound",
        "decay", "retained", "reconstruction", "error", "Pareto", "curve", "efficiency", "macroeconomic",
        "infrastructure", "cost", "benefit", "model", "CapEx", "OpEx", "fleet", "datacenter", "cooling",
        "power", "megawatt", "silicon", "fabrication", "virtual", "Gemini", "Pro", "Ultra", "context",
        "length", "million", "tokens", "attention", "mechanism", "query", "key", "value", "projection"
    ]
    random.seed(42)
    generated_words = []
    while len(generated_words) < seq_len * 2:
        generated_words.append(random.choice(words))
        
    text = " ".join(generated_words)
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = tokens["input_ids"][:, :seq_len]
    if input_ids.shape[1] < seq_len:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        padding = torch.full((1, seq_len - input_ids.shape[1]), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, padding], dim=1)
    return input_ids


def _extract_qkv_cache(model: Any, input_ids: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass with hooks to capture queries, keys, and values from the global layer."""
    import torch

    # Select the layer index with the longest context sequence (global layer)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values

    seq_lens = []
    if hasattr(past_kv, "key_cache") and isinstance(past_kv.key_cache, list):
        seq_lens = [kv.shape[2] for kv in past_kv.key_cache]
    elif hasattr(past_kv, "layers") and isinstance(past_kv.layers, list):
        for lc in past_kv.layers:
            if hasattr(lc, "keys"):
                seq_lens.append(lc.keys.shape[2])
            elif hasattr(lc, "key_cache"):
                seq_lens.append(lc.key_cache.shape[2])
            else:
                seq_lens.append(0)
    else:
        seq_lens = [kv[0].shape[2] for kv in past_kv]

    layer_idx = int(np.argmax(seq_lens))
    print(f"  Selected layer {layer_idx} (cached seq_len={seq_lens[layer_idx]})")

    if hasattr(model, "model"):
        layers = model.model.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        layers = model.model.decoder.layers

    layer = layers[layer_idx]
    attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attn

    q_captured = []
    k_captured = []
    v_captured = []

    def hook_q(module, input, output):
        q_captured.append(output.detach().float().cpu())
    def hook_k(module, input, output):
        k_captured.append(output.detach().float().cpu())
    def hook_v(module, input, output):
        v_captured.append(output.detach().float().cpu())

    h_q = attn.q_proj.register_forward_hook(hook_q)
    h_k = attn.k_proj.register_forward_hook(hook_k)
    h_v = attn.v_proj.register_forward_hook(hook_v)

    with torch.no_grad():
        model(input_ids)

    h_q.remove()
    h_k.remove()
    h_v.remove()

    q_tensor = q_captured[0][0]  # (seq_len, num_heads * head_dim)
    k_tensor = k_captured[0][0]  # (seq_len, num_kv_heads * head_dim)
    v_tensor = v_captured[0][0]  # (seq_len, num_kv_heads * head_dim)

    tc = model.config.text_config if hasattr(model.config, "text_config") else model.config
    num_heads = tc.num_attention_heads
    num_kv_heads = tc.num_key_value_heads
    head_dim = q_tensor.shape[-1] // num_heads

    q = q_tensor.view(-1, num_heads, head_dim).numpy()
    k_dim = k_tensor.shape[-1] // num_kv_heads
    k = k_tensor.view(-1, num_kv_heads, k_dim).numpy()
    v = v_tensor.view(-1, num_kv_heads, k_dim).numpy()

    if num_kv_heads < num_heads:
        repeats = num_heads // num_kv_heads
        k = np.repeat(k, repeats, axis=1)
        v = np.repeat(v, repeats, axis=1)

    return q, k, v


def _pad_to_block(arr: np.ndarray) -> np.ndarray:
    seq_len = arr.shape[0]
    remainder = seq_len % BLOCK_SIZE
    if remainder == 0:
        return arr
    pad_len = BLOCK_SIZE - remainder
    pad = np.zeros((pad_len, *arr.shape[1:]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


# ---------------------------------------------------------------------------
# Dense attention
# ---------------------------------------------------------------------------

def _dense_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Standard dense scaled-dot-product attention."""
    head_dim = q.shape[-1]
    logits = jnp.einsum("qhd,khd->qkh", q, k) / jnp.sqrt(head_dim)
    weights = jax.nn.softmax(logits, axis=1)
    return jnp.einsum("qkh,khd->qhd", weights, v), weights


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_eviction(
    keys_jax: jnp.ndarray,
    values_jax: jnp.ndarray,
    query_jax: jnp.ndarray,
    target_eviction: float,
) -> dict[str, float]:
    """Run query-aware sparse attention at *target_eviction* and compare with dense."""
    seq_len_k, num_heads, head_dim = keys_jax.shape
    num_blocks = seq_len_k // BLOCK_SIZE

    # Dense attention output + weights
    dense_out, dense_weights = _dense_attention(query_jax, keys_jax, values_jax)
    dense_weights_np = np.asarray(dense_weights)

    # Query-aware bounds computation
    bounds = compute_query_aware_bounds(query_jax, keys_jax, block_size=BLOCK_SIZE)
    # Take max over queries to yield (num_blocks, num_heads)
    max_bounds = jnp.max(bounds, axis=0)
    max_bounds_np = np.asarray(max_bounds)
    
    # Determine the query-aware threshold tau for target eviction
    tau = float(np.percentile(max_bounds_np.flatten(), target_eviction * 100))
    block_mask = max_bounds >= tau

    # Actual eviction rate
    mask_np = np.asarray(block_mask)
    actual_eviction = 1.0 - mask_np.mean()
    num_evicted = int((~mask_np).sum())

    # Sparse attention output
    sparse_out = jax_block_sparse_attention(
        query_jax, keys_jax, values_jax, block_mask, block_size=BLOCK_SIZE,
    )

    # Reconstruct sparse attention weights for TV / KL measurement
    logits = jnp.einsum("qhd,khd->qkh", query_jax, keys_jax) / jnp.sqrt(head_dim)
    mask_seq = jnp.repeat(mask_np, BLOCK_SIZE, axis=0)  # (seq_len_k, num_heads)
    mask_broad = mask_seq[jnp.newaxis, :, :]             # (1, seq_len_k, num_heads)
    logits_masked = jnp.where(mask_broad, logits, -1e9)
    sparse_weights = jax.nn.softmax(logits_masked, axis=1)
    sparse_weights_np = np.asarray(sparse_weights)

    # TV distance (averaged over query positions and heads)
    tv_total = 0.0
    seq_len_q = dense_weights_np.shape[0]
    for qi in range(seq_len_q):
        for hi in range(num_heads):
            tv_total += compute_tv_distance(
                dense_weights_np[qi, :, hi],
                sparse_weights_np[qi, :, hi],
            )
    tv_mean = tv_total / (seq_len_q * num_heads)

    # KL divergence
    eps_kl = 1e-10
    dense_safe = np.clip(dense_weights_np, eps_kl, None)
    sparse_safe = np.clip(sparse_weights_np, eps_kl, None)
    kl_per_element = dense_safe * np.log(dense_safe / sparse_safe)
    kl_mean = float(np.mean(np.sum(kl_per_element, axis=1)))

    # ------------------------------------------------------------------
    # OrthoCache Truncation Bound verification
    #   TV  ≤  |S^c| · exp(τ − z_max)
    # where τ is the query-aware eviction threshold
    # ------------------------------------------------------------------
    logits_np = np.asarray(logits)  # (seq_len_q, seq_len_k, num_heads)
    mask_seq_np = np.asarray(mask_seq)  # (seq_len_k, num_heads)

    bound_violations = 0
    bound_value_max = 0.0
    for qi in range(seq_len_q):
        for hi in range(num_heads):
            retained = logits_np[qi, mask_seq_np[:, hi], hi]
            evicted = logits_np[qi, ~mask_seq_np[:, hi], hi]
            if evicted.size == 0 or retained.size == 0:
                continue
            z_max = float(np.max(retained))
            s_c = evicted.size
            bound = s_c * np.exp(tau - z_max)
            measured = compute_tv_distance(
                dense_weights_np[qi, :, hi],
                sparse_weights_np[qi, :, hi],
            )
            bound_value_max = max(bound_value_max, bound)
            if measured > bound + 1e-6:
                bound_violations += 1

    # Output reconstruction error (Frobenius)
    recon_err = float(jnp.linalg.norm(dense_out - sparse_out) / jnp.linalg.norm(dense_out))

    return {
        "target_eviction": target_eviction,
        "actual_eviction": actual_eviction,
        "num_evicted_blocks": num_evicted,
        "threshold_eps": tau,
        "tv_mean": tv_mean,
        "kl_mean": kl_mean,
        "recon_error_rel": recon_err,
        "bound_violations": bound_violations,
        "bound_max": bound_value_max,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OrthoCache attention-accuracy benchmark at multiple eviction rates.",
    )
    parser.add_argument("--model", type=str,
                        default="/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b/1",
                        help="HuggingFace model name or local path.")
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="Sequence length in tokens (default: 4096).")
    parser.add_argument("--query_len", type=int, default=16,
                        help="Number of query tokens to evaluate (default: 16).")
    parser.add_argument("--output_dir", type=str, default="benchmarks/plots",
                        help="Directory for output plots and CSVs.")
    parser.add_argument("--eviction_rates", type=str, default="0.1,0.3,0.5,0.7",
                        help="Comma-separated target eviction fractions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eviction_rates = [float(x) for x in args.eviction_rates.split(",")]

    print("=== OrthoCache Attention Accuracy Benchmark ===")
    print(f"  Model          : {args.model}")
    print(f"  Seq length     : {args.seq_len}")
    print(f"  Query length   : {args.query_len}")
    print(f"  Eviction rates : {eviction_rates}")
    print()

    # 1. Model + KV-cache -------------------------------------------------
    print("[1/3] Loading model …")
    model, tokenizer = _load_model_and_tokenizer(args.model)

    print("[2/3] Extracting QKV vectors …")
    input_ids = _build_prompt(tokenizer, args.seq_len)
    q_np, keys_np, values_np = _extract_qkv_cache(model, input_ids)

    # Pad to block boundary
    keys_np = _pad_to_block(keys_np)
    values_np = _pad_to_block(values_np)

    keys_jax = jnp.array(keys_np, dtype=jnp.float32)
    values_jax = jnp.array(values_np, dtype=jnp.float32)

    # Slice query vectors
    query_jax = jnp.array(q_np[-args.query_len:, :, :], dtype=jnp.float32)

    # 3. Evaluate at each eviction rate -----------------------------------
    print("[3/3] Evaluating eviction rates …")
    results: list[dict[str, float]] = []
    for rate in eviction_rates:
        print(f"  Eviction target {rate:.0%} …", end=" ", flush=True)
        row = evaluate_eviction(keys_jax, values_jax, query_jax, rate)
        results.append(row)
        print(f"TV={row['tv_mean']:.6f}  KL={row['kl_mean']:.6f}  "
              f"recon_err={row['recon_error_rel']:.6f}  "
              f"bound_violations={row['bound_violations']}")

    # CSV -----------------------------------------------------------------
    csv_path = output_dir / "attention_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Pareto plot ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        evictions = [r["actual_eviction"] for r in results]
        tvs = [r["tv_mean"] for r in results]
        kls = [r["kl_mean"] for r in results]
        recons = [r["recon_error_rel"] for r in results]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(evictions, tvs, "o-", linewidth=2, markersize=8)
        axes[0].set_xlabel("Eviction Rate")
        axes[0].set_ylabel("Mean TV Distance")
        axes[0].set_title("TV Distance vs Eviction")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(evictions, kls, "s-", linewidth=2, markersize=8, color="tab:orange")
        axes[1].set_xlabel("Eviction Rate")
        axes[1].set_ylabel("Mean KL Divergence")
        axes[1].set_title("KL Divergence vs Eviction")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(evictions, recons, "^-", linewidth=2, markersize=8, color="tab:green")
        axes[2].set_xlabel("Eviction Rate")
        axes[2].set_ylabel("Relative Reconstruction Error")
        axes[2].set_title("Output Error vs Eviction")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("OrthoCache Accuracy–Efficiency Pareto Curves", fontsize=13)
        fig.tight_layout()
        fig.savefig(output_dir / "attention_accuracy_pareto.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping Pareto plot.")

    # Summary table -------------------------------------------------------
    print()
    header = (f"{'Evict%':>7} {'Actual%':>7} {'TV':>10} {'KL':>10} "
              f"{'ReconErr':>10} {'Violations':>10}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['target_eviction']:>7.0%} {r['actual_eviction']:>7.1%} "
              f"{r['tv_mean']:>10.6f} {r['kl_mean']:>10.6f} "
              f"{r['recon_error_rel']:>10.6f} {r['bound_violations']:>10}")

    violations_total = sum(r["bound_violations"] for r in results)
    print()
    if violations_total == 0:
        print("✓ OrthoCache Truncation Bound holds at all eviction rates.")
    else:
        print(f"✗ {violations_total} bound violation(s) detected — investigate numerics.")

    print(f"\nResults written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
