# SPDX-License-Identifier: Apache-2.0
"""PR4 selection-only router kernel; sigmoid and weight arithmetic stay in Torch.

Unlike torch.topk(sorted=False), ties choose the lowest index and outputs are
score-ordered. This backend is opt-in, not a bitwise-equivalent replacement.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_topk_kernel(
    scores_ptr,
    weights_ptr,
    ids_ptr,
    stride_n,
    stride_e,
    E: tl.constexpr,
    G: tl.constexpr,
    EPG: tl.constexpr,
    KG: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    experts = tl.arange(0, E)
    scores = tl.load(scores_ptr + row * stride_n + experts * stride_e)
    grouped = tl.reshape(scores, (G, EPG))
    first = tl.argmax(grouped, axis=1)
    second = tl.max(
        tl.where(tl.arange(0, EPG)[None, :] == first[:, None], -float("inf"), grouped),
        axis=1,
    )
    group_scores = tl.max(grouped, axis=1) + second
    groups = tl.arange(0, G)
    selected = groups < 0
    for _ in tl.static_range(KG):
        pick = tl.argmax(tl.where(selected, -float("inf"), group_scores), axis=0)
        selected = selected | (groups == pick)
    active = tl.reshape(tl.broadcast_to(selected[:, None], (G, EPG)), (E,))
    for k in tl.static_range(K):
        values = tl.where(active, scores, -float("inf"))
        pick = tl.argmax(values, axis=0)
        tl.store(weights_ptr + row * K + k, tl.max(values, axis=0))
        tl.store(ids_ptr + row * K + k, pick.to(tl.int64))
        active = active & (experts != pick)


def grouped_topk_triton(scores, num_experts_per_tok, n_group, topk_group):
    """Select from finite, bias-added FP32 scores, without casts or fallback."""
    if not scores.is_cuda or scores.dtype != torch.float32 or scores.ndim != 2:
        raise ValueError("Triton router requires a 2D CUDA float32 scores tensor")
    rows, experts = scores.shape
    if n_group < 1 or experts % n_group:
        raise ValueError("num_experts must be divisible by positive n_group")
    epg = experts // n_group
    if any(x < 1 or x & (x - 1) for x in (experts, n_group, epg)) or epg < 2:
        raise ValueError("Triton router requires power-of-two groups with >=2 experts")
    if not 1 <= topk_group <= n_group:
        raise ValueError("topk_group must be in [1, n_group]")
    if not 1 <= num_experts_per_tok <= topk_group * epg:
        raise ValueError("top-k exceeds the selected groups' expert capacity")
    weights = torch.empty(
        (rows, num_experts_per_tok), dtype=torch.float32, device=scores.device
    )
    ids = torch.empty_like(weights, dtype=torch.int64)
    if rows:
        _grouped_topk_kernel[(rows,)](
            scores,
            weights,
            ids,
            scores.stride(0),
            scores.stride(1),
            experts,
            n_group,
            epg,
            topk_group,
            num_experts_per_tok,
        )
    return weights, ids
