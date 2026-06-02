"""AllToAllv protocol for OrthoCache multi-device sequence-parallel attention.

Implements the three-step OrthoCache collective communication protocol that
enables efficient KV-cache exchange across sequence-parallel devices. On
Kaggle TPU v5e-8 we have P=8 TPU chips visible to JAX; jax.pmap with
axis_name='devices' maps each core to one sequence shard.

Protocol overview
─────────────────
  Step 1 — count_sync:   all_gather local active-block counts → every
                         device knows every peer's K_i.
  Step 2 — compute_offsets: parallel prefix-sum on the gathered counts →
                         per-device write offsets into the global buffer.
  Step 3 — alltoallv_exchange: all_to_all on packed (padded-to-max)
                         KV buffers (Strategy C).

Design constraints
──────────────────
* All tensor shapes MUST be static for XLA / pmap compilation.
* ``num_active`` is a *runtime* value; buffer allocation uses the static
  ``max_blocks`` upper bound.  Valid data is distinguished from padding
  via masking and ``num_active`` counts.
* Only ``jax.lax`` primitives are used inside pmapped functions — no
  Python-level control flow.

See also
────────
* ``orthocache.compaction`` — single-device stream compaction.
* ``orthocache.xla_bridge``  — XLA custom-call wrappers.
"""

import jax
import jax.numpy as jnp
from jax import lax

# ─── Module constants ────────────────────────────────────────────────────────

BLOCK_SIZE: int = 512
"""Number of tokens per KV-cache block."""


# ─── 1. Stream compaction (sort-based, single-device) ────────────────────────

def stream_compact(block_mask: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Sort-based stream compaction on a 1-D boolean block mask.

    Pushes active (True) block indices to the front of an index array using
    a stable argsort on the negated mask.  Active blocks therefore appear in
    their original relative order in ``active_indices[0:num_active]``.

    This is the lightweight variant used by the AllToAllv protocol — it
    operates on the mask only and does *not* gather KV data (see
    ``pack_active_blocks`` for that step).

    Args:
        block_mask: Boolean tensor of shape ``(num_blocks,)`` where True
            indicates a block that should be retained (is active).

    Returns:
        A tuple ``(active_indices, num_active)`` where:

        - **active_indices** — ``int32`` tensor of shape ``(num_blocks,)``.
          The first ``num_active`` entries hold the original block indices
          of the active blocks, in order.  Entries at positions
          ``>= num_active`` are indices of inactive blocks (defined but
          should be ignored by downstream consumers).
        - **num_active** — ``int32`` scalar.  Count of active blocks.
    """
    active_int = block_mask.astype(jnp.int32)            # (num_blocks,)
    num_active = jnp.sum(active_int)                     # scalar int32

    # Negate so that active blocks (−1) sort before inactive blocks (0).
    active_indices = jnp.argsort(-active_int, stable=True)  # (num_blocks,)

    return active_indices, num_active


# ─── 2. Step 1 — count synchronisation ──────────────────────────────────────

def count_sync(
    local_num_active: jax.Array,
    axis_name: str = 'devices',
) -> jax.Array:
    """Broadcast every device's active-block count to all peers.

    This is **Step 1** of the AllToAllv protocol.  Each device contributes
    its scalar ``local_num_active`` count, and ``jax.lax.all_gather``
    replicates the full vector on every device so that every peer knows
    every other peer's workload.

    Must be called inside a ``jax.pmap`` region with the matching
    ``axis_name``.

    Args:
        local_num_active: Scalar ``int32`` — number of active blocks on
            the calling device.
        axis_name: Name of the pmap axis (default ``'devices'``).

    Returns:
        ``all_counts`` — ``int32`` tensor of shape ``(P,)`` where ``P`` is
        the number of devices along ``axis_name``.  ``all_counts[i]`` is
        the active-block count reported by device *i*.
    """
    # all_gather expects at least 1-D; reshape scalar → (1,) then flatten.
    local_count_1d = jnp.expand_dims(local_num_active, axis=0)  # (1,)
    gathered = lax.all_gather(local_count_1d, axis_name=axis_name)  # (P, 1)
    all_counts = gathered.squeeze(axis=-1)  # (P,)
    return all_counts


# ─── 3. Step 2 — offset computation (parallel prefix sum) ───────────────────

def compute_offsets(
    all_counts: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute per-device write offsets via exclusive prefix sum.

    This is **Step 2** of the AllToAllv protocol.  Given the gathered
    active-block counts ``all_counts`` (one per device), computes

        offsets = [0, cumsum(all_counts[:-1])]

    so that device *i* writes its active blocks starting at
    ``offsets[i]`` in the global receive buffer.

    Args:
        all_counts: ``int32`` tensor of shape ``(P,)`` — per-device active
            counts (output of :func:`count_sync`).

    Returns:
        A tuple ``(offsets, total_active)`` where:

        - **offsets** — ``int32`` tensor of shape ``(P,)``.
          ``offsets[i]`` is the starting block index for device *i*
          in the consolidated output buffer.
        - **total_active** — ``int32`` scalar.  Sum of all active counts
          across all devices.
    """
    # Exclusive prefix sum: prepend 0, take cumsum, drop last element.
    prefix = jnp.cumsum(all_counts)             # (P,) inclusive
    offsets = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.int32), prefix[:-1]]
    )                                            # (P,)
    total_active = prefix[-1]                    # scalar
    return offsets, total_active


# ─── 4. Pack active blocks into a fixed-size send buffer ─────────────────────

def pack_active_blocks(
    kv_shard: jax.Array,
    active_indices: jax.Array,
    num_active: jax.Array,
    max_blocks: int,
) -> jax.Array:
    """Gather active KV blocks into a statically-shaped packed buffer.

    Iterates over ``max_blocks`` slots using ``jax.lax.fori_loop``.  For
    each slot *i < num_active*, the block at ``active_indices[i]`` is
    copied from ``kv_shard`` into the send buffer via
    ``jax.lax.dynamic_slice`` / ``jax.lax.dynamic_update_slice``.  Slots
    at *i >= num_active* are left as zeros (padding).

    The output shape is fully static —
    ``(max_blocks * BLOCK_SIZE, num_heads, head_dim)`` — satisfying XLA's
    requirement for pmap-compatible code.

    Args:
        kv_shard: Local KV shard of shape
            ``(local_seq, num_heads, head_dim)`` in ``bfloat16``.
            ``local_seq`` must equal ``max_blocks * BLOCK_SIZE``.
        active_indices: ``int32`` tensor of shape ``(max_blocks,)`` —
            output of :func:`stream_compact`.
        num_active: ``int32`` scalar — number of valid entries in
            ``active_indices``.
        max_blocks: Static (compile-time) upper bound on the number of
            blocks per device.

    Returns:
        Packed buffer of shape ``(max_blocks * BLOCK_SIZE, num_heads, head_dim)``
        with the same dtype as ``kv_shard``.  The first
        ``num_active * BLOCK_SIZE`` tokens contain real data; the rest are
        zero-padded.
    """
    _, num_heads, head_dim = kv_shard.shape
    total_tokens = max_blocks * BLOCK_SIZE

    packed = jnp.zeros((total_tokens, num_heads, head_dim), dtype=kv_shard.dtype)

    def _copy_block(i, buf):
        """Copy block *i* from kv_shard into buf if i < num_active."""
        src_block_idx = active_indices[i]
        src_start = src_block_idx * BLOCK_SIZE

        # dynamic_slice: extract one block from kv_shard
        block = lax.dynamic_slice(
            kv_shard,
            (src_start, 0, 0),
            (BLOCK_SIZE, num_heads, head_dim),
        )  # (BLOCK_SIZE, num_heads, head_dim)

        dst_start = i * BLOCK_SIZE

        # Only write if i < num_active; otherwise leave zeros.
        should_write = i < num_active
        block = jnp.where(should_write, block, jnp.zeros_like(block))

        buf = lax.dynamic_update_slice(buf, block, (dst_start, 0, 0))
        return buf

    packed = lax.fori_loop(0, max_blocks, _copy_block, packed)
    return packed


# ─── 5. Step 3 — AllToAll exchange (Strategy C) ─────────────────────────────

def alltoallv_exchange(
    k_packed: jax.Array,
    v_packed: jax.Array,
    axis_name: str = 'devices',
) -> tuple[jax.Array, jax.Array]:
    """All-to-all exchange of packed KV buffers across devices.

    This is **Step 3** (Strategy C) of the AllToAllv protocol.  Each
    device contributes its padded send buffer and receives an equally-sized
    buffer from every other device, concatenated along the sequence
    (token) axis.

    After the exchange, device *i* holds the packed KV blocks from *all*
    devices.  Use :func:`unpack_received_blocks` to extract only the
    valid (non-padding) tokens.

    Must be called inside a ``jax.pmap`` region with the matching
    ``axis_name``.

    Args:
        k_packed: Packed key buffer of shape
            ``(max_tokens_per_device, num_heads, head_dim)``.
        v_packed: Packed value buffer, same shape as ``k_packed``.
        axis_name: pmap axis name (default ``'devices'``).

    Returns:
        A tuple ``(k_received, v_received)`` where each tensor has shape
        ``(P * max_tokens_per_device, num_heads, head_dim)`` — the
        concatenated receive buffers from all ``P`` devices.
    """
    k_received = lax.all_to_all(
        k_packed,
        axis_name=axis_name,
        split_axis=0,
        concat_axis=0,
    )
    v_received = lax.all_to_all(
        v_packed,
        axis_name=axis_name,
        split_axis=0,
        concat_axis=0,
    )
    return k_received, v_received


# ─── 6. Unpack received blocks (post-trim) ──────────────────────────────────

def unpack_received_blocks(
    kv_received: jax.Array,
    all_counts: jax.Array,
    num_devices: int,
    max_blocks: int,
    block_size: int = BLOCK_SIZE,
) -> tuple[jax.Array, jax.Array]:
    """Extract valid (non-padding) tokens from the all_to_all output.

    After ``alltoallv_exchange``, the receive buffer is a concatenation of
    ``P`` equal-sized shards, each of size ``max_blocks * block_size``
    tokens.  Only the first ``all_counts[i] * block_size`` tokens in
    shard *i* are real data; the rest are zero padding.

    This function constructs a boolean validity mask and gathers valid
    tokens into a contiguous prefix, returning the active-only tensor and
    per-device offsets.

    Args:
        kv_received: Tensor of shape
            ``(num_devices * max_blocks * block_size, num_heads, head_dim)``
            — output of :func:`alltoallv_exchange`.
        all_counts: ``int32`` tensor of shape ``(P,)`` — per-device
            active-block counts.
        num_devices: Static device count ``P``.
        max_blocks: Static per-device block capacity.
        block_size: Tokens per block (default 512).

    Returns:
        A tuple ``(active_kv, device_offsets)`` where:

        - **active_kv** — Tensor of shape
          ``(num_devices * max_blocks * block_size, num_heads, head_dim)``.
          The first ``total_active_tokens`` entries contain valid data
          (gathered in device order), the rest are zero-padded.
          ``total_active_tokens = sum(all_counts) * block_size``.
        - **device_offsets** — ``int32`` tensor of shape ``(P,)``.
          ``device_offsets[i]`` is the token-level start offset for
          device *i* inside ``active_kv``.
    """
    shard_tokens = max_blocks * block_size       # tokens per shard
    total_tokens = num_devices * shard_tokens     # total buffer length
    _, num_heads, head_dim = kv_received.shape

    # Build per-token validity mask across all shards.
    # Token t in the flat buffer belongs to shard (t // shard_tokens).
    # It is valid iff its position within that shard < count[shard] * block_size.
    token_indices = jnp.arange(total_tokens, dtype=jnp.int32)
    shard_ids = token_indices // shard_tokens              # which device
    within_shard = token_indices - shard_ids * shard_tokens  # offset in shard
    # Active token counts per device (in tokens, not blocks)
    active_token_counts = all_counts * block_size          # (P,)
    shard_limits = active_token_counts[shard_ids]          # (total_tokens,)
    valid_mask = within_shard < shard_limits               # (total_tokens,)

    # Sort-based compaction: push valid tokens to front.
    sort_key = (~valid_mask).astype(jnp.int32)  # 0 for valid, 1 for padding
    gather_indices = jnp.argsort(sort_key, stable=True)    # (total_tokens,)

    active_kv = kv_received[gather_indices]                # gather
    # Zero-out the padding tail
    total_active_tokens = jnp.sum(active_token_counts)
    pad_mask = jnp.arange(total_tokens) >= total_active_tokens
    active_kv = jnp.where(
        pad_mask[:, None, None], jnp.zeros_like(active_kv), active_kv
    )

    # Compute per-device token offsets into the active buffer
    offsets_blocks, _ = compute_offsets(all_counts)
    device_offsets = offsets_blocks * block_size            # (P,)

    return active_kv, device_offsets


# ─── 7. High-level wrapper — full AllToAllv KV exchange ──────────────────────

def alltoallv_kv_exchange(
    k_shard: jax.Array,
    v_shard: jax.Array,
    block_mask: jax.Array,
    axis_name: str = 'devices',
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict]:
    """End-to-end AllToAllv KV-cache exchange across sequence-parallel devices.

    Orchestrates the full three-step protocol:

    1. **Stream compact** the local block mask to identify active blocks
       and their count.
    2. **Count sync** — ``all_gather`` so every device knows every peer's
       active count.
    3. **Compute offsets** — exclusive prefix sum for write positions.
    4. **Pack** active key and value blocks into static-sized send buffers.
    5. **AllToAll exchange** — collective all-to-all on both packed buffers.
    6. **Unpack** — trim padding from the received buffers.

    All intermediate shapes are **static** (determined at trace time from
    ``block_mask.shape``); only *counts* vary at runtime.

    Must be called inside a ``jax.pmap`` region with the matching
    ``axis_name``.

    Args:
        k_shard: Local key shard, shape
            ``(local_seq, num_heads, head_dim)``, dtype ``bfloat16``.
        v_shard: Local value shard, same shape / dtype as ``k_shard``.
        block_mask: Boolean tensor of shape ``(num_blocks,)`` indicating
            which blocks on this device are active (True = retained).
        axis_name: pmap axis name (default ``'devices'``).

    Returns:
        A 5-tuple ``(k_active, v_active, all_counts, offsets, stats_dict)``:

        - **k_active** — Keys containing all active blocks from every
          device, shape ``(P * max_tokens, num_heads, head_dim)``.  The
          first ``total_active * BLOCK_SIZE`` tokens are valid.
        - **v_active** — Values, same layout as ``k_active``.
        - **all_counts** — ``int32 (P,)`` — per-device active block counts.
        - **offsets** — ``int32 (P,)`` — per-device block-level write
          offsets into the global buffer.
        - **stats_dict** — Dictionary with bandwidth-saving statistics:

          * ``local_num_active``: active blocks on this device.
          * ``total_active``: active blocks summed across all devices.
          * ``dense_blocks``: ``P * max_blocks`` — what a naïve
            (no-compaction) exchange would send.
          * ``bytes_saved``: approximate bytes saved vs. dense exchange
            (assuming bfloat16, 2 bytes per element, keys + values).
          * ``bytes_dense``: bytes a dense exchange would transfer.
    """
    num_blocks = block_mask.shape[0]
    max_blocks = num_blocks  # static upper bound = total blocks per device
    local_seq, num_heads, head_dim = k_shard.shape

    # --- Step 0: Local stream compaction (mask only) -------------------------
    active_indices, num_active = stream_compact(block_mask)

    # --- Step 1: Count synchronisation (all_gather) --------------------------
    all_counts = count_sync(num_active, axis_name=axis_name)

    # --- Step 2: Offset computation (prefix sum) -----------------------------
    offsets, total_active = compute_offsets(all_counts)

    # --- Step 3a: Pack active blocks into send buffers -----------------------
    k_packed = pack_active_blocks(k_shard, active_indices, num_active, max_blocks)
    v_packed = pack_active_blocks(v_shard, active_indices, num_active, max_blocks)

    # --- Step 3b: AllToAll exchange -------------------------------------------
    num_devices = all_counts.shape[0]
    k_received, v_received = alltoallv_exchange(
        k_packed, v_packed, axis_name=axis_name,
    )

    # --- Step 4: Unpack / trim received buffers ------------------------------
    k_active, device_offsets_k = unpack_received_blocks(
        k_received, all_counts, num_devices, max_blocks,
    )
    v_active, _ = unpack_received_blocks(
        v_received, all_counts, num_devices, max_blocks,
    )

    # --- Bandwidth statistics ------------------------------------------------
    dense_blocks = num_devices * max_blocks
    elements_per_block = BLOCK_SIZE * num_heads * head_dim
    bytes_per_element = 2  # bfloat16
    bytes_dense = dense_blocks * elements_per_block * bytes_per_element * 2  # K+V
    bytes_active = total_active * elements_per_block * bytes_per_element * 2
    bytes_saved = bytes_dense - bytes_active

    stats_dict = {
        'local_num_active': num_active,
        'total_active': total_active,
        'dense_blocks': dense_blocks,
        'bytes_saved': bytes_saved,
        'bytes_dense': bytes_dense,
    }

    return k_active, v_active, all_counts, offsets, stats_dict
