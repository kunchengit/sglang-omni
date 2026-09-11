# SPDX-License-Identifier: Apache-2.0
"""Real thinker Python/Torch contracts with explicit SGLang layer doubles."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn


@pytest.fixture
def thinker(monkeypatch):
    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    class Layer(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args, self.kwargs = args, kwargs

        def forward(self, x):
            return x, None

    class Experts(Layer):
        def forward(self, x, routing):
            self.routing = routing
            return x.mul_(2)  # Exercise the actual shared-input clone.

    stub("transformers", PretrainedConfig=object)
    stub("sglang_omni.models.weight_loader", default_weight_loader=lambda *_: None)
    stub("sglang_omni.vendor.sglang.core", ForwardBatch=object)
    stub(
        "sglang_omni.vendor.sglang.distributed",
        get_tensor_model_parallel_world_size=lambda: 1,
        tensor_model_parallel_all_reduce=lambda x: x,
    )
    stub(
        "sglang_omni.vendor.sglang.layers",
        AttentionType=NS(ENCODER_ONLY=1),
        MergedColumnParallelLinear=Layer,
        QKVParallelLinear=Layer,
        QuantizationConfig=object,
        RadixAttention=Layer,
        RMSNorm=Layer,
        RowParallelLinear=Layer,
        SiluAndMul=nn.Identity,
        StandardTopKOutput=lambda **kw: NS(**kw),
        VocabParallelEmbedding=Layer,
        get_moe_impl_class=lambda _: Experts,
        get_rope=lambda *_a, **_kw: None,
    )
    stub(
        "sglang_omni.vendor.sglang.models",
        apply_qk_norm=lambda *_: None,
        create_fused_set_kv_buffer_arg=lambda **_: None,
        enable_fused_set_kv_buffer=lambda _: False,
    )
    stub("sglang_omni.vendor.sglang.utils", make_layers=lambda *_: nn.ModuleList())
    path = (
        Path(__file__).resolve().parents[3]
        / "sglang_omni/models/llada2_uni/components/thinker.py"
    )
    spec = importlib.util.spec_from_file_location("_pr4_thinker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(**overrides):
    values = {
        "hidden_size": 4,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "n_group": 2,
        "topk_group": 1,
        "routed_scaling_factor": 1.7,
        "moe_intermediate_size": 4,
        "num_shared_experts": 1,
        "router_dtype": "fp32",
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "use_qk_norm": False,
        "use_qkv_bias": False,
        "rotary_dim": 2,
        "max_position_embeddings": 128,
        "rope_theta": 10000,
        "vocab_size": 16,
        "num_hidden_layers": 0,
        "rms_norm_eps": 1e-5,
    }
    return NS(**(values | overrides))


@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("shared", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_partial_experts_reduce_once_without_precision_promotion(
    thinker, monkeypatch, tp_size, shared, dtype
):
    calls = []
    monkeypatch.setattr(
        thinker, "get_tensor_model_parallel_world_size", lambda: tp_size
    )

    def reduce(x):
        calls.append(x.clone())
        return x * tp_size

    monkeypatch.setattr(thinker, "tensor_model_parallel_all_reduce", reduce)
    quant = object()
    block = thinker.LLaDA2MoeSparseMoeBlock(config(num_shared_experts=shared), 0, quant)
    block.gate.weight.data.copy_(torch.arange(32).view(8, 4) / 32)
    block.gate.expert_bias.copy_(torch.arange(8) / 100)
    assert block.experts.kwargs["quant_config"] is quant
    assert block.experts.kwargs["reduce_results"] is False
    if shared:
        assert block.shared_experts.down_proj.kwargs["reduce_results"] is False
        assert block.shared_experts.down_proj.kwargs["quant_config"] is quant
    inputs = torch.ones(3, 4, dtype=dtype)
    output = block(inputs)
    expected_partial = torch.full_like(inputs, 2 + shared)
    torch.testing.assert_close(output, expected_partial * tp_size, rtol=0, atol=0)
    assert output.dtype == dtype and len(calls) == (1 if tp_size > 1 else 0)
    if calls:
        assert calls[0].dtype == dtype
        torch.testing.assert_close(calls[0], expected_partial, rtol=0, atol=0)
    route = block.experts.routing
    assert route.router_logits.dtype == torch.float32
    scores = route.router_logits.sigmoid().gather(1, route.topk_ids)
    weights = scores / (scores.sum(-1, keepdim=True) + 1e-20) * 1.7
    torch.testing.assert_close(route.topk_weights, weights, rtol=0, atol=0)
    assert dict(block.gate.named_buffers())["expert_bias"].dtype == torch.float32


def test_quant_config_reaches_attention_and_embedding(thinker):
    quant = object()
    attention = thinker.LLaDA2MoeAttention(config(), 0, quant)
    assert attention.attn.kwargs["quant_config"] is quant
    assert attention.query_key_value.kwargs["quant_config"] is quant
    assert attention.dense.kwargs["quant_config"] is quant
    model = thinker.LLaDA2MoeTextModel(config(), quant)
    assert model.word_embeddings.kwargs["quant_config"] is quant


def test_router_backend_is_explicit_and_never_falls_back(thinker, monkeypatch):
    with pytest.raises(ValueError, match="llada2_router_topk_backend"):
        thinker.LLaDA2MoeSparseMoeBlock(config(llada2_router_topk_backend="auto"), 0)
    block = thinker.LLaDA2MoeSparseMoeBlock(
        config(llada2_router_topk_backend="triton"), 0
    )
    module = ModuleType("sglang_omni.models.llada2_uni.components.triton_topk")

    def fail(*_):
        raise RuntimeError("topk launch failed")

    module.grouped_topk_triton = fail
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(RuntimeError, match="topk launch failed"):
        block._group_limited_topk(torch.ones(1, 8))


@pytest.mark.parametrize("capture", [False, True])
def test_shared_expert_overlap_forks_and_joins_only_in_capture(
    thinker, monkeypatch, capture
):
    from contextlib import contextmanager

    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append((self.name, "wait", other.name))

    main, alternate = Stream("main"), Stream("alt")
    runner = ModuleType("sglang.srt.model_executor.runner")
    runner.get_is_capture_mode = lambda: capture
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)

    @contextmanager
    def on_stream(stream):
        assert stream is alternate
        events.append("enter-alt")
        yield
        events.append("leave-alt")

    monkeypatch.setattr(torch.cuda, "stream", on_stream)
    block = thinker.LLaDA2MoeSparseMoeBlock(config(), 0, alt_stream=alternate)
    block.gate.weight.data.fill_(0.1)
    block.shared_experts.register_forward_hook(lambda *_: events.append("shared"))
    block.experts.register_forward_hook(lambda *_: events.append("routed"))
    block.tp_size = 2

    def reduce(x):
        events.append("reduce")
        return x

    monkeypatch.setattr(thinker, "tensor_model_parallel_all_reduce", reduce)
    result = block(torch.ones(2, 4, dtype=torch.bfloat16))
    torch.testing.assert_close(result, torch.full_like(result, 3), rtol=0, atol=0)
    assert result.dtype == torch.bfloat16
    assert events == (
        [
            ("alt", "wait", "main"),
            "enter-alt",
            "shared",
            "leave-alt",
            "routed",
            ("main", "wait", "alt"),
            "reduce",
        ]
        if capture
        else ["routed", "shared", "reduce"]
    )


def test_overlap_mode_validation_and_shared_stream_plumbing(thinker, monkeypatch):
    with pytest.raises(ValueError, match="llada2_shared_expert_overlap"):
        thinker.LLaDA2MoeTextModel(config(llada2_shared_expert_overlap="auto"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        thinker.LLaDA2MoeTextModel(config(llada2_shared_expert_overlap="cuda_graph"))
    assert thinker.LLaDA2MoeTextModel(config()).alt_stream is None
    stream = object()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: stream)
    monkeypatch.setattr(
        thinker,
        "make_layers",
        lambda count, factory: nn.ModuleList([factory(i) for i in range(count)]),
    )
    model = thinker.LLaDA2MoeTextModel(
        config(
            llada2_shared_expert_overlap="cuda_graph",
            num_hidden_layers=2,
            first_k_dense_replace=0,
        )
    )
    assert all(layer.mlp.alt_stream is stream for layer in model.layers)
