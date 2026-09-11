# SPDX-License-Identifier: Apache-2.0
"""PR4 tiled argmax and winning softmax probability, without update policy.

CFG arithmetic, threshold decisions and mask updates remain in the algorithm.
Only FP32 logits are supported: promoting BF16 here would change its confidence
rounding relative to the reference softmax/gather path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_kernel(
    logits,
    maxima,
    sums,
    indices,
    stride_n,
    stride_v,
    V: tl.constexpr,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    ids = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(logits + row * stride_n + ids * stride_v, ids < V, -float("inf"))
    maximum = tl.max(x, axis=0)
    # Image-only masking can leave whole vocabulary tiles at -inf.
    shifted = tl.where(maximum == -float("inf"), -float("inf"), x - maximum)
    total = tl.sum(tl.exp(shifted), axis=0)
    index = tl.min(tl.where((ids < V) & (x == maximum), ids, 2147483647), axis=0)
    offset = row * TILES + tile
    tl.store(maxima + offset, maximum)
    tl.store(sums + offset, total)
    tl.store(indices + offset, index)


@triton.jit
def _finalize_kernel(
    maxima,
    sums,
    indices,
    output_ids,
    confidence,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tiles = tl.arange(0, BLOCK)
    maximum = tl.load(maxima + row * TILES + tiles, tiles < TILES, -float("inf"))
    total = tl.load(sums + row * TILES + tiles, tiles < TILES, 0.0)
    index = tl.load(indices + row * TILES + tiles, tiles < TILES, 2147483647)
    global_max = tl.max(maximum, axis=0)
    mass = tl.sum(total * tl.exp(maximum - global_max), axis=0)
    winner = tl.min(tl.where(maximum == global_max, index, 2147483647), axis=0)
    tl.store(output_ids + row, winner.to(tl.int64))
    tl.store(confidence + row, 1.0 / mass)


def argmax_confidence_triton(logits):
    """Return first argmax and its softmax probability; kernel errors propagate."""
    if not logits.is_cuda or logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError("Triton decode requires a 2D CUDA float32 logits tensor")
    rows, vocab = logits.shape
    if vocab < 1:
        raise ValueError("Triton decode requires a nonempty vocabulary")
    tiles = triton.cdiv(vocab, 1024)
    maxima = torch.empty((rows, tiles), dtype=torch.float32, device=logits.device)
    sums = torch.empty_like(maxima)
    indices = torch.empty_like(maxima, dtype=torch.int32)
    ids = torch.empty(rows, dtype=torch.int64, device=logits.device)
    confidence = torch.empty(rows, dtype=torch.float32, device=logits.device)
    if rows:
        _partial_kernel[(rows, tiles)](
            logits,
            maxima,
            sums,
            indices,
            logits.stride(0),
            logits.stride(1),
            vocab,
            tiles,
            1024,
        )
        _finalize_kernel[(rows,)](
            maxima,
            sums,
            indices,
            ids,
            confidence,
            tiles,
            triton.next_power_of_2(tiles),
        )
    return ids, confidence
