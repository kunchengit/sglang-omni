# SPDX-License-Identifier: Apache-2.0
"""CUDA numerical tests, deliberately not replaced by CPU kernel mocks."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_native_srt_forward_batch_preserves_pr3_metadata():
    from dataclasses import fields, replace

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    from sglang_omni.scheduling.dllm_scheduler import DllmForwardBatch

    tensor = torch.tensor([1], device="cuda", dtype=torch.int64)
    batch = DllmForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=tensor,
        req_pool_indices=tensor,
        seq_lens=tensor,
        out_cache_loc=tensor,
        seq_lens_sum=1,
        reqs=[object()],
        dllm_left_pad_lens=tensor.clone(),
    )
    assert isinstance(batch, ForwardBatch)
    assert {"reqs", "dllm_left_pad_lens"} <= {field.name for field in fields(batch)}
    rebuilt = replace(batch, input_ids=tensor.clone())
    assert rebuilt.reqs is batch.reqs
    assert rebuilt.dllm_left_pad_lens is batch.dllm_left_pad_lens


def test_shared_expert_overlap_cuda_graph_replay(monkeypatch):
    """Real CUDA streams/capture, with lightweight experts instead of a checkpoint."""
    from types import SimpleNamespace

    from sglang.srt.model_executor import runner
    from torch import nn

    from sglang_omni.models.llada2_uni.components.thinker import (
        LLaDA2MoeGate,
        LLaDA2MoeSparseMoeBlock,
    )

    class Routed(nn.Module):
        def forward(self, x, _routing):
            return x.mul_(2)

    class Shared(nn.Module):
        def forward(self, x):
            return x * 3

    block = LLaDA2MoeSparseMoeBlock.__new__(LLaDA2MoeSparseMoeBlock)
    nn.Module.__init__(block)
    block.tp_size = 1
    block.num_experts, block.n_group = 8, 2
    block.num_experts_per_tok, block.topk_group = 2, 1
    block.routed_scaling_factor = 1.7
    block.router_topk_backend = "torch"
    block.gate = LLaDA2MoeGate(SimpleNamespace(num_experts=8, hidden_size=4)).cuda()
    block.gate.weight.data.fill_(0.1)
    block.experts, block.shared_experts = Routed(), Shared()
    block.alt_stream = torch.cuda.Stream()
    # No full SRT runtime is bootstrapped by this stream primitive test.
    monkeypatch.setattr(
        runner, "get_is_capture_mode", torch.cuda.is_current_stream_capturing
    )
    inputs = torch.ones(8, 4, device="cuda", dtype=torch.bfloat16)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            block(inputs.clone())
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = block(inputs.clone())
    for value in (1.0, 2.0, 3.0):
        inputs.fill_(value)
        graph.replay()
        torch.testing.assert_close(result, inputs * 5, rtol=0, atol=0)


@pytest.mark.parametrize("vocab", [1, 1025, 4097, 165376])
@pytest.mark.parametrize("strided", [False, True])
def test_decode_confidence(vocab, strided):
    from sglang_omni.models.llada2_uni.algorithm.triton_decode import (
        argmax_confidence_triton,
    )

    torch.manual_seed(123)
    logits = torch.randn(7, vocab * (2 if strided else 1), device="cuda")
    if strided:
        logits = logits[:, ::2]
    # Whole tiles masked out, irregular last tile, and first-index tie semantics.
    if vocab > 1024:
        logits[:, :1024] = -float("inf")
    logits[0, -1] = 30.0
    if vocab > 4096:
        logits[1, 2048] = logits[1, 4096] = 10.0
    ids, confidence = argmax_confidence_triton(logits)
    ref_ids = logits.argmax(-1)
    ref_confidence = logits.softmax(-1).gather(-1, ref_ids[:, None]).squeeze(-1)
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(confidence, ref_confidence, rtol=3e-5, atol=1e-7)
    assert torch.equal(confidence > 0.95, ref_confidence > 0.95)


@pytest.mark.parametrize("experts,groups,k", [(64, 8, 6), (256, 8, 8)])
@pytest.mark.parametrize("strided", [False, True])
def test_router_selection_and_bias_free_weighted_output(experts, groups, k, strided):
    from sglang_omni.models.llada2_uni.components.triton_topk import grouped_topk_triton

    torch.manual_seed(42)
    logits = torch.randn(17, experts * (2 if strided else 1), device="cuda")
    scores = logits.sigmoid()
    if strided:
        scores = scores[:, ::2]
    bias = torch.randn(experts, device="cuda") * 0.01
    routing = scores + bias
    if strided:
        routing = routing.t().contiguous().t()
    grouped = routing.reshape(17, groups, -1)
    selected_groups = grouped.topk(2, dim=-1).values.sum(-1).topk(2, dim=-1).indices
    mask = torch.zeros(17, groups, device="cuda", dtype=torch.bool)
    mask.scatter_(1, selected_groups, True)
    mask = mask[:, :, None].expand_as(grouped).reshape_as(routing)
    ref_ids = routing.masked_fill(~mask, -float("inf")).topk(k, sorted=False).indices
    selected, ids = grouped_topk_triton(routing, k, groups, 2)
    torch.testing.assert_close(
        ids.sort(-1).values, ref_ids.sort(-1).values, rtol=0, atol=0
    )
    torch.testing.assert_close(selected, routing.gather(1, ids), rtol=0, atol=0)
    expert_outputs = torch.randn(17, experts, 8, device="cuda")

    def combine(indices):
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 1.7
        outputs = expert_outputs.gather(1, indices[:, :, None].expand(-1, -1, 8))
        return (weights[:, :, None] * outputs).sum(1)

    torch.testing.assert_close(combine(ids), combine(ref_ids), rtol=2e-5, atol=2e-6)


def test_router_ties_and_empty_batch():
    from sglang_omni.models.llada2_uni.components.triton_topk import grouped_topk_triton

    weights, ids = grouped_topk_triton(torch.ones(2, 8, device="cuda"), 3, 2, 1)
    assert ids.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert torch.equal(weights, torch.ones_like(weights))
    assert grouped_topk_triton(torch.empty(0, 8, device="cuda"), 3, 2, 1)[1].shape == (
        0,
        3,
    )


def test_explicit_backend_rejects_unsupported_dtype_and_device():
    from sglang_omni.models.llada2_uni.algorithm.triton_decode import (
        argmax_confidence_triton,
    )
    from sglang_omni.models.llada2_uni.components.triton_topk import grouped_topk_triton

    for tensor in (
        torch.zeros(2, 8),
        torch.zeros(2, 8, device="cuda", dtype=torch.bfloat16),
    ):
        with pytest.raises(ValueError, match="CUDA float32"):
            argmax_confidence_triton(tensor)
        with pytest.raises(ValueError, match="CUDA float32"):
            grouped_topk_triton(tensor, 2, 2, 1)
    with pytest.raises(ValueError, match="capacity"):
        grouped_topk_triton(torch.zeros(2, 8, device="cuda"), 5, 2, 1)
