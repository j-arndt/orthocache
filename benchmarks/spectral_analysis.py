"""OrthoCache Spectral Analysis Benchmark.

Loads a causal-LM model (default: Gemma 4 E2B on Kaggle), runs a long-context
forward pass, extracts KV-cache key tensors from intermediate decoder layers,
applies the 512-point Fast Walsh-Hadamard Transform, and reports the spectral
energy distribution across blocks.

The --model flag accepts both HuggingFace hub IDs and local filesystem paths
(e.g. a Kaggle model input directory).

Outputs
-------
* Per-layer CSV files with block energies  → benchmarks/plots/
* Histogram + CDF PNG plots               → benchmarks/plots/
* Summary statistics printed to stdout

Usage
-----
    python benchmarks/spectral_analysis.py --seq_len 8192
    python benchmarks/spectral_analysis.py --model /kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b/1 --seq_len 16384
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

from orthocache.fwht import fwht_512
from orthocache.spectral_energy import compute_block_energy_jax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(model_name: str) -> tuple[Any, Any]:
    """Load a HuggingFace causal-LM and its tokenizer in bfloat16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def _build_prompt(tokenizer: Any, seq_len: int) -> Any:
    """Construct a token tensor of exactly *seq_len* tokens.

    Repeats a seed paragraph until the target length is reached, then
    truncates to the requested size.
    """
    import torch

    seed = (
        "The Walsh-Hadamard transform is an orthogonal, involutory linear "
        "operator widely used in signal processing and error-correcting codes. "
        "When applied to the key cache of a transformer decoder, the transform "
        "concentrates energy into a small number of spectral coefficients, "
        "enabling aggressive block eviction without significant accuracy loss. "
    )
    # Repeat enough times to overshoot, then truncate
    repeated = seed * (seq_len // 10 + 1)
    tokens = tokenizer(repeated, return_tensors="pt", truncation=True, max_length=seq_len)
    input_ids = tokens["input_ids"][:, :seq_len]
    # Pad if the tokenizer produced fewer tokens than requested
    if input_ids.shape[1] < seq_len:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        padding = torch.full((1, seq_len - input_ids.shape[1]), pad_id, dtype=input_ids.dtype)
        input_ids = torch.cat([input_ids, padding], dim=1)
    return input_ids


def _extract_key_caches(model: Any, input_ids: Any) -> list[np.ndarray]:
    """Run a forward pass and return key tensors from every decoder layer.

    Each returned array has shape ``(seq_len, num_heads, head_dim)``.
    """
    import torch

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    past_kv = outputs.past_key_values
    key_arrays: list[np.ndarray] = []
    for layer_kv in past_kv:
        # layer_kv is a tuple (key, value); key shape: (batch, num_heads, seq_len, head_dim)
        key_tensor = layer_kv[0][0]  # drop batch dim
        # Transpose to (seq_len, num_heads, head_dim) to match OrthoCache convention
        key_np = key_tensor.permute(1, 0, 2).float().cpu().numpy()
        key_arrays.append(key_np)
    return key_arrays


def _pad_to_block_boundary(keys: np.ndarray, block_size: int = 512) -> np.ndarray:
    """Zero-pad keys along the sequence axis so length is a multiple of *block_size*."""
    seq_len = keys.shape[0]
    remainder = seq_len % block_size
    if remainder == 0:
        return keys
    pad_len = block_size - remainder
    pad = np.zeros((pad_len, *keys.shape[1:]), dtype=keys.dtype)
    return np.concatenate([keys, pad], axis=0)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_layer(
    keys_np: np.ndarray,
    layer_idx: int,
    output_dir: Path,
) -> dict[str, float]:
    """Compute spectral energy for one layer, emit CSV + plots, return stats."""
    keys_np = _pad_to_block_boundary(keys_np, block_size=512)
    keys_jax = jnp.array(keys_np, dtype=jnp.float32)

    energies = compute_block_energy_jax(keys_jax, block_size=512)  # (num_blocks, num_heads)
    energies_np: np.ndarray = np.asarray(energies)

    flat_energies = energies_np.flatten()

    # ---- CSV -----------------------------------------------------------
    csv_path = output_dir / f"spectral_energy_layer{layer_idx:02d}.csv"
    num_blocks, num_heads = energies_np.shape
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_idx", "head_idx", "energy"])
        for b in range(num_blocks):
            for h in range(num_heads):
                writer.writerow([b, h, float(energies_np[b, h])])

    # ---- Histogram + CDF plot ------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(12, 4))

        ax_hist.hist(flat_energies, bins=50, edgecolor="black", alpha=0.7)
        ax_hist.set_xlabel("Block Energy  $E_j = \\|\\hat{K}_{B_j}\\|_F^2$")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"Layer {layer_idx} — Energy Histogram")

        sorted_e = np.sort(flat_energies)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax_cdf.plot(sorted_e, cdf, linewidth=1.5)
        ax_cdf.set_xlabel("Block Energy")
        ax_cdf.set_ylabel("CDF")
        ax_cdf.set_title(f"Layer {layer_idx} — Energy CDF")

        fig.tight_layout()
        fig.savefig(output_dir / f"spectral_energy_layer{layer_idx:02d}.png", dpi=150)
        plt.close(fig)
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping plots.")

    # ---- Summary statistics --------------------------------------------
    stats = {
        "layer": layer_idx,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "mean": float(np.mean(flat_energies)),
        "std": float(np.std(flat_energies)),
        "min": float(np.min(flat_energies)),
        "p25": float(np.percentile(flat_energies, 25)),
        "median": float(np.median(flat_energies)),
        "p75": float(np.percentile(flat_energies, 75)),
        "max": float(np.max(flat_energies)),
    }
    return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OrthoCache spectral-energy analysis on real KV-cache activations.",
    )
    parser.add_argument("--model", type=str,
                        default="/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b/1",
                        help="HuggingFace model name or local path (default: Kaggle Gemma 4 E2B).")
    parser.add_argument("--seq_len", type=int, default=8192,
                        help="Sequence length in tokens (default: 8192).")
    parser.add_argument("--output_dir", type=str, default="benchmarks/plots",
                        help="Directory for output CSVs and plots.")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to analyse (default: all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== OrthoCache Spectral Analysis ===")
    print(f"  Model      : {args.model}")
    print(f"  Seq length : {args.seq_len}")
    print(f"  Output dir : {output_dir.resolve()}")
    print()

    # 1. Load model -------------------------------------------------------
    print("[1/3] Loading model and tokenizer …")
    model, tokenizer = _load_model_and_tokenizer(args.model)

    # 2. Forward pass -----------------------------------------------------
    print("[2/3] Running forward pass to extract KV-cache …")
    input_ids = _build_prompt(tokenizer, args.seq_len)
    key_caches = _extract_key_caches(model, input_ids)
    print(f"       Extracted keys from {len(key_caches)} layers, "
          f"shape per layer: {key_caches[0].shape}")

    # 3. Spectral analysis ------------------------------------------------
    print("[3/3] Computing spectral energy …")
    layer_indices: list[int]
    if args.layers is not None:
        layer_indices = [int(x) for x in args.layers.split(",")]
    else:
        layer_indices = list(range(len(key_caches)))

    all_stats: list[dict[str, float]] = []
    for idx in layer_indices:
        print(f"  Layer {idx} …", end=" ", flush=True)
        stats = analyse_layer(key_caches[idx], idx, output_dir)
        all_stats.append(stats)
        print(f"mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"min={stats['min']:.4f}  max={stats['max']:.4f}")

    # Summary table -------------------------------------------------------
    print()
    print(f"{'Layer':>5} {'Blocks':>6} {'Heads':>5} {'Mean':>10} {'Std':>10} "
          f"{'Min':>10} {'P25':>10} {'Median':>10} {'P75':>10} {'Max':>10}")
    print("-" * 97)
    for s in all_stats:
        print(f"{s['layer']:>5} {s['num_blocks']:>6} {s['num_heads']:>5} "
              f"{s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} "
              f"{s['p25']:>10.4f} {s['median']:>10.4f} {s['p75']:>10.4f} "
              f"{s['max']:>10.4f}")
    print()
    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
