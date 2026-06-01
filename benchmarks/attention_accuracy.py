"""OrthoCache Attention Accuracy Benchmark.

Measures how block-sparse attention quality degrades as blocks are evicted at
increasing rates.  For each eviction target the script:

1. Computes the full dense attention output.
2. Uses spectral-energy thresholding to build a block mask.
3. Computes sparse attention via ``jax_block_sparse_attention``.
4. Measures TV distance (``compute_tv_distance``) and KL divergence.
5. Verifies the OrthoCache Truncation Bound:
       measured TV  ≤  |S^c| · exp(β − z_max)

Outputs
-------
* Accuracy–efficiency Pareto curves (PNG)  → benchmarks/plots/
* Per-threshold results (CSV)              → benchmarks/plots/
* Bound-violation report on stdout

Usage
-----
    python benchmarks/attention_accuracy.py --seq_len 4096
    python benchmarks/attention_accuracy.py --model /kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b/1 --seq_len 8192
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

from orthocache.spectral_energy import compute_block_energy_jax, generate_threshold_mask
from orthocache.sparse_attention import jax_block_sparse_attention
from orthocache.reference import compute_tv_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BLOCK_SIZE = 512


def _load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    """Load a HuggingFace causal-LM and tokenizer in bfloat16 on CPU.

    CPU is intentional — JAX owns the TPU in the notebook process.
    The model is only used for one forward pass to extract KV-cache.
    """
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
    """Build a token tensor of exactly *seq_len* tokens."""
    import torch

    seed = (
        "The Walsh-Hadamard transform concentrates spectral energy into a "
        "small number of coefficients.  Blocks with low energy can be "
        "evicted from the KV-cache without measurable accuracy loss. "
    )
    repeated = seed * (seq_len // 8 + 1)
    tokens = tokenizer(repeated, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = tokens["input_ids"][:, :seq_len]
    if input_ids.shape[1] < seq_len:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        padding = torch.full((1, seq_len - input_ids.shape[1]), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, padding], dim=1)
    return input_ids


def _extract_kv_cache(model: Any, input_ids: Any) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass → (keys, values) each of shape (seq_len, num_kv_heads, head_dim).

    Handles every known ``past_key_values`` format:
      - ``DynamicCache`` with ``.key_cache`` / ``.value_cache`` lists
      - ``DynamicCache`` with ``._cache`` list-of-tuples (some versions)
      - Legacy tuple-of-tuples
    Falls back to introspecting ``__dict__`` if none of the above match.
    """
    import torch

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    past_kv = outputs.past_key_values

    # Collect (key_tensor, value_tensor) per layer — each (batch, heads, seq, dim)
    layer_kvs: list[tuple] = []

    if hasattr(past_kv, "key_cache") and isinstance(past_kv.key_cache, list) and len(past_kv.key_cache) > 0:
        # DynamicCache with key_cache / value_cache (transformers >= 4.36)
        for i in range(len(past_kv.key_cache)):
            layer_kvs.append((past_kv.key_cache[i], past_kv.value_cache[i]))
    elif hasattr(past_kv, "_cache") and isinstance(past_kv._cache, list):
        # Some DynamicCache versions store [(k, v), ...] in _cache
        for item in past_kv._cache:
            layer_kvs.append((item[0], item[1]))
    elif isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
        # Legacy tuple-of-tuples
        for item in past_kv:
            layer_kvs.append((item[0], item[1]))
    else:
        # Last resort: inspect __dict__ for lists of tensors
        print(f"  DEBUG: past_kv type = {type(past_kv).__name__}")
        print(f"  DEBUG: past_kv attrs = {[a for a in dir(past_kv) if not a.startswith('__')]}")
        cache_dict = vars(past_kv) if hasattr(past_kv, "__dict__") else {}
        print(f"  DEBUG: __dict__ keys = {list(cache_dict.keys())}")

        # Try to find any list of tensors
        for attr_name, attr_val in cache_dict.items():
            if isinstance(attr_val, list) and len(attr_val) > 0:
                first = attr_val[0]
                if hasattr(first, "shape"):
                    print(f"  DEBUG: Found tensor list in '{attr_name}', len={len(attr_val)}, shape[0]={first.shape}")

        raise TypeError(
            f"Cannot extract KV-cache from {type(past_kv).__name__}. "
            f"Attrs: {[a for a in dir(past_kv) if not a.startswith('__')]}"
        )

    if not layer_kvs:
        raise ValueError("No KV-cache layers found in model output")

    # Select the layer with the longest cached sequence (global attention layer)
    seq_lens = [kv[0].shape[2] for kv in layer_kvs]
    layer_idx = int(np.argmax(seq_lens))
    print(f"  Selected layer {layer_idx}/{len(layer_kvs)} "
          f"(seq_len={seq_lens[layer_idx]}, heads={layer_kvs[layer_idx][0].shape[1]}, "
          f"head_dim={layer_kvs[layer_idx][0].shape[3]})")

    k = layer_kvs[layer_idx][0][0]  # drop batch → (heads, seq, dim)
    v = layer_kvs[layer_idx][1][0]

    keys = k.permute(1, 0, 2).float().cpu().numpy()    # (seq, heads, dim)
    values = v.permute(1, 0, 2).float().cpu().numpy()
    return keys, values


def _pad_to_block(arr: np.ndarray) -> np.ndarray:
    seq_len = arr.shape[0]
    remainder = seq_len % BLOCK_SIZE
    if remainder == 0:
        return arr
    pad_len = BLOCK_SIZE - remainder
    pad = np.zeros((pad_len, *arr.shape[1:]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _find_threshold_for_eviction(energies: np.ndarray, target_eviction: float) -> float:
    """Return the energy threshold ε that evicts approximately *target_eviction* fraction of blocks."""
    flat = energies.flatten()
    # Evicting fraction p means the threshold is at percentile p*100 of the energies
    return float(np.percentile(flat, target_eviction * 100))


# ---------------------------------------------------------------------------
# Dense attention (reference)
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
    """Run sparse attention at *target_eviction* and compare with dense."""
    seq_len_k, num_heads, head_dim = keys_jax.shape

    # Dense attention output + weights
    dense_out, dense_weights = _dense_attention(query_jax, keys_jax, values_jax)
    dense_weights_np = np.asarray(dense_weights)

    # Spectral energy → threshold → mask
    energies = compute_block_energy_jax(keys_jax, block_size=BLOCK_SIZE)
    energies_np = np.asarray(energies)
    eps = _find_threshold_for_eviction(energies_np, target_eviction)
    block_mask = generate_threshold_mask(energies, epsilon=eps)

    # Actual eviction rate
    mask_np = np.asarray(block_mask)
    actual_eviction = 1.0 - mask_np.mean()
    num_evicted = int((~mask_np).sum())

    # Sparse attention output
    sparse_out = jax_block_sparse_attention(
        query_jax, keys_jax, values_jax, block_mask, block_size=BLOCK_SIZE,
    )

    # Reconstruct sparse attention weights for TV / KL measurement
    num_blocks = seq_len_k // BLOCK_SIZE
    logits = jnp.einsum("qhd,khd->qkh", query_jax, keys_jax) / jnp.sqrt(head_dim)
    # Mask out evicted blocks
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

    # KL divergence  D_KL(dense || sparse)  — averaged
    eps_kl = 1e-10
    dense_safe = np.clip(dense_weights_np, eps_kl, None)
    sparse_safe = np.clip(sparse_weights_np, eps_kl, None)
    kl_per_element = dense_safe * np.log(dense_safe / sparse_safe)
    kl_mean = float(np.mean(np.sum(kl_per_element, axis=1)))

    # ------------------------------------------------------------------
    # OrthoCache Truncation Bound verification
    #   TV  ≤  |S^c| · exp(β − z_max)
    # where β = max logit in evicted set, z_max = max logit in retained set
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
            beta = float(np.max(evicted))
            z_max = float(np.max(retained))
            s_c = evicted.size
            bound = s_c * np.exp(beta - z_max)
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
        "threshold_eps": eps,
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
                        help="HuggingFace model name or local path (default: Kaggle Gemma 4 E2B).")
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="Sequence length in tokens (default: 4096).")
    parser.add_argument("--query_len", type=int, default=16,
                        help="Number of query tokens to evaluate (default: 16).")
    parser.add_argument("--output_dir", type=str, default="benchmarks/plots",
                        help="Directory for output plots and CSVs.")
    parser.add_argument("--eviction_rates", type=str, default="0.1,0.3,0.5,0.7",
                        help="Comma-separated target eviction fractions (default: 0.1,0.3,0.5,0.7).")
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

    print("[2/3] Extracting KV-cache …")
    input_ids = _build_prompt(tokenizer, args.seq_len)
    keys_np, values_np = _extract_kv_cache(model, input_ids)

    # Pad to block boundary
    keys_np = _pad_to_block(keys_np)
    values_np = _pad_to_block(values_np)

    keys_jax = jnp.array(keys_np, dtype=jnp.float32)
    values_jax = jnp.array(values_np, dtype=jnp.float32)

    # Use the last `query_len` tokens as queries
    query_jax = keys_jax[-args.query_len:, :, :]

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
