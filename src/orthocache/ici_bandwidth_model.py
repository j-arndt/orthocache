"""ICI Bandwidth Model for OrthoCache.

Pure analytical model (no JAX distributed primitives) that computes exact
byte counts for ICI (Inter-Chip Interconnect) transfer volumes with and
without OrthoCache sparsity-based eviction.

Formulas derived from CBA §3.2:
    ICI_dense  = L × (S × d_k × H_kv / P) × dtype_bytes
    ICI_sparse = ICI_dense × (1 − sparsity)

where:
    L        = number of transformer layers
    S        = sequence length (tokens)
    d_k      = head dimension
    H_kv     = number of KV heads (after GQA grouping)
    P        = number of devices (tensor-parallel shards)
    sparsity = fraction of KV cache entries evicted by OrthoCache
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BYTES_PER_GB = 1e9   # decimal GB (SI)
_BYTES_PER_MB = 1e6   # decimal MB (SI)

_DEFAULT_SPARSITY_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.688, 0.75, 0.9]


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------


def ici_bytes_per_step(
    num_layers: int,
    seq_len: int,
    head_dim: int,
    num_kv_heads: int,
    num_devices: int,
    sparsity: float,
    dtype_bytes: int = 2,
) -> dict:
    """Compute ICI transfer bytes for a single decode step.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers (L).
    seq_len : int
        Sequence length in tokens (S).
    head_dim : int
        Per-head dimension (d_k).
    num_kv_heads : int
        Number of KV attention heads after GQA grouping (H_kv).
    num_devices : int
        Tensor-parallel device count (P).
    sparsity : float
        Fraction of KV entries evicted (0.0 = dense, 1.0 = fully sparse).
    dtype_bytes : int, optional
        Bytes per element (default 2 for bfloat16 / float16).

    Returns
    -------
    dict
        dense_bytes       – total ICI bytes without eviction
        sparse_bytes      – total ICI bytes with OrthoCache eviction
        savings_bytes     – absolute reduction
        savings_pct       – percentage reduction (0–100)
        savings_gb_per_1000_steps – cumulative GB saved over 1000 steps
    """
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

    # CBA §3.2 — Per-device ICI transfer volume, summed across all layers.
    # With tensor-parallel KV-head partitioning (H_kv / P) and context-parallel
    # sequence partitioning (S / P), each device's all-gather shard is:
    #   ICI_dense = L × S × d_k × H_kv / P² × dtype_bytes
    dense_bytes = int(
        num_layers * seq_len * head_dim * num_kv_heads * dtype_bytes
        / (num_devices * num_devices)
    )
    sparse_bytes = int(dense_bytes * (1.0 - sparsity))
    savings_bytes = dense_bytes - sparse_bytes

    savings_pct = (savings_bytes / dense_bytes * 100.0) if dense_bytes > 0 else 0.0
    savings_gb_per_1000 = (savings_bytes * 1000) / _BYTES_PER_GB

    return {
        "dense_bytes": dense_bytes,
        "sparse_bytes": sparse_bytes,
        "savings_bytes": savings_bytes,
        "savings_pct": round(savings_pct, 2),
        "savings_gb_per_1000_steps": round(savings_gb_per_1000, 3),
    }


# ---------------------------------------------------------------------------
# Bandwidth table
# ---------------------------------------------------------------------------


def ici_bandwidth_table(
    num_layers: int,
    seq_len: int,
    head_dim: int,
    num_kv_heads: int,
    num_devices: int,
    dtype_bytes: int = 2,
    sparsity_levels: list[float] | None = None,
) -> list[dict]:
    """Build a table of ICI bandwidth across multiple sparsity levels.

    Parameters
    ----------
    num_layers, seq_len, head_dim, num_kv_heads, num_devices, dtype_bytes
        Same as :func:`ici_bytes_per_step`.
    sparsity_levels : list[float] | None
        Sparsity values to tabulate.  Defaults to
        ``[0, 0.1, 0.3, 0.5, 0.688, 0.75, 0.9]``.

    Returns
    -------
    list[dict]
        Each dict contains: sparsity, dense_mb, sparse_mb, savings_mb,
        savings_pct.
    """
    if sparsity_levels is None:
        sparsity_levels = list(_DEFAULT_SPARSITY_LEVELS)

    rows: list[dict] = []
    for s in sparsity_levels:
        r = ici_bytes_per_step(
            num_layers=num_layers,
            seq_len=seq_len,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            num_devices=num_devices,
            sparsity=s,
            dtype_bytes=dtype_bytes,
        )
        rows.append(
            {
                "sparsity": s,
                "dense_mb": round(r["dense_bytes"] / _BYTES_PER_MB, 2),
                "sparse_mb": round(r["sparse_bytes"] / _BYTES_PER_MB, 2),
                "savings_mb": round(r["savings_bytes"] / _BYTES_PER_MB, 2),
                "savings_pct": r["savings_pct"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Pre-defined model configurations
# ---------------------------------------------------------------------------


def model_configs() -> dict:
    """Return pre-defined ICI configurations for common LLM architectures.

    Returns
    -------
    dict
        Mapping of model name → config dict with keys:
        num_layers, num_heads, num_kv_heads, head_dim, num_devices, label.
    """
    return {
        "gemma4_31b": {
            "num_layers": 50,
            "num_heads": 128,
            "num_kv_heads": 4,
            "head_dim": 256,
            "num_devices": 8,
            "label": "Gemma-4 31B (GQA 128/4, d_k=256, P=8)",
        },
        "llama3_70b": {
            "num_layers": 80,
            "num_heads": 64,
            "num_kv_heads": 8,
            "head_dim": 128,
            "num_devices": 8,
            "label": "LLaMA-3 70B (GQA 64/8, d_k=128, P=8)",
        },
        "llama3_405b": {
            "num_layers": 126,
            "num_heads": 128,
            "num_kv_heads": 8,
            "head_dim": 128,
            "num_devices": 16,
            "label": "LLaMA-3 405B (GQA 128/8, d_k=128, P=16)",
        },
    }


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------


def print_ici_report(model_name: str, seq_len: int) -> None:
    """Pretty-print the ICI bandwidth table for a named model configuration.

    Parameters
    ----------
    model_name : str
        Key into :func:`model_configs` (e.g. ``"llama3_70b"``).
    seq_len : int
        Sequence length in tokens.

    Raises
    ------
    KeyError
        If *model_name* is not in the pre-defined configs.
    """
    configs = model_configs()
    if model_name not in configs:
        raise KeyError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(configs.keys())}"
        )
    cfg = configs[model_name]

    table = ici_bandwidth_table(
        num_layers=cfg["num_layers"],
        seq_len=seq_len,
        head_dim=cfg["head_dim"],
        num_kv_heads=cfg["num_kv_heads"],
        num_devices=cfg["num_devices"],
    )

    seq_k = seq_len // 1024
    header = f"  ICI Bandwidth - {cfg['label']} @ {seq_k}K tokens"
    print()
    print("=" * 72)
    print(header)
    print("=" * 72)
    print(
        f"  {'Sparsity':>10s}  {'Dense (MB)':>12s}  {'Sparse (MB)':>12s}"
        f"  {'Saved (MB)':>12s}  {'Saved (%)':>10s}"
    )
    print("-" * 72)
    for row in table:
        print(
            f"  {row['sparsity']:>10.1%}  {row['dense_mb']:>12.2f}"
            f"  {row['sparse_mb']:>12.2f}  {row['savings_mb']:>12.2f}"
            f"  {row['savings_pct']:>9.2f}%"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_reference_70b() -> None:
    """Sanity-check against the CBA §3.2 reference value.

    70B model: 80 layers, 128K tokens, d_k=128, 8 KV heads, P=8
    Expected dense ICI ≈ 335.5 MB (decimal).
    """
    r = ici_bytes_per_step(
        num_layers=80,
        seq_len=128 * 1024,
        head_dim=128,
        num_kv_heads=8,
        num_devices=8,
        sparsity=0.0,
    )
    dense_mb = r["dense_bytes"] / _BYTES_PER_MB
    # Allow a small tolerance for integer rounding
    assert abs(dense_mb - 335.5) < 1.0, (
        f"Validation failed: expected ~335.5 MB dense, got {dense_mb:.2f} MB"
    )
    print(f"  [VALIDATE] 70B dense ICI = {dense_mb:.2f} MB  [OK]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("OrthoCache ICI Bandwidth Model")
    print("=" * 72)

    # Validate against known reference
    _validate_reference_70b()

    seq_len = 128 * 1024  # 128K tokens

    for name in model_configs():
        print_ici_report(name, seq_len)

    print()
    print("Done.")
