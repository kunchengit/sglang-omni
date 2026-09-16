# SPDX-License-Identifier: Apache-2.0
"""CUDA integration tests for the LLaDA2-Uni runtime."""

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
