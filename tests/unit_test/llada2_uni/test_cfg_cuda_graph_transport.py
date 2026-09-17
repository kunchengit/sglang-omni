# SPDX-License-Identifier: Apache-2.0
"""Real SGLang ForwardBatch/registry/eager transport; no import doubles.

Run in the target SGLang environment. This checks CPU metadata transport, not
FlashInfer kernel execution or CUDA graph capture.
"""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip(
    "sglang.srt.model_executor.forward_batch_info", exc_type=ImportError
)

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    GraphSlot,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner

from sglang_omni.models.llada2_uni.cfg_cuda_graph import (
    attach_cfg_graph_views,
    register_cfg_graph_slots,
)
from sglang_omni.models.llada2_uni.cfg_cuda_graph_metadata import (
    DllmCFGForwardBatch,
    as_cfg_forward_batch,
)
from tests.unit_test.llada2_uni.cfg_graph_test_utils import tiny_batch, tiny_runner


def test_real_runtime_backend_capability_registration(monkeypatch):
    from sglang.srt import server_args
    from sglang.srt.arg_groups import overrides
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

    from sglang_omni.models.llada2_uni.bootstrap import register_llada2_uni_cfg

    original = ATTENTION_BACKENDS.copy()
    register_llada2_uni_cfg()
    register_llada2_uni_cfg()
    assert server_args.ATTENTION_BACKEND_CHOICES.count("llada2_uni_cfg_flashinfer") == 1
    assert all(
        ATTENTION_BACKENDS[name] is factory for name, factory in original.items()
    )
    monkeypatch.setattr(
        overrides, "get_platform", lambda: NS(is_hip=False, is_npu=False)
    )
    args = NS(
        dllm_algorithm="LowConfidenceCFG",
        attention_backend="llada2_uni_cfg_flashinfer",
        cuda_graph_config=NS(decode=NS(backend="full")),
    )
    assert overrides._dllm_attention_backend(args) == {
        "attention_backend": "flashinfer"
    }


@pytest.mark.parametrize("pads", [(0,), (0, 6), (2, 6, 3)])
@pytest.mark.parametrize("path", ["replace", "registry", "eager", "eager_no_copy"])
def test_real_cfg_metadata_transport(monkeypatch, pads, path):
    size = len(pads)
    batch = DllmCFGForwardBatch(
        forward_mode=ForwardMode.DLLM_EXTEND,
        batch_size=size,
        input_ids=torch.arange(size * 4),
        req_pool_indices=torch.arange(size),
        seq_lens=torch.full((size,), 12, dtype=torch.int32),
        out_cache_loc=torch.arange(size * 4),
        seq_lens_sum=size * 12,
        positions=torch.arange(size * 4, dtype=torch.int64),
        extend_prefix_lens_cpu=[8] * size,
        extend_seq_lens_cpu=[4] * size,
        dllm_left_pad_lens_cpu=list(pads),
    )
    assert DllmCFGForwardBatch.init_new.__func__ is ForwardBatch.init_new.__func__
    assert DllmCFGForwardBatch.init_new.__self__ is DllmCFGForwardBatch
    batch = as_cfg_forward_batch(batch)
    registry = CudaGraphBufferRegistry(
        device=torch.device("cpu"), max_bs=4, max_num_tokens=16
    )
    registry.register_slot(
        GraphSlot(
            name="positions",
            shape_fn=lambda _, tokens: (tokens,),
            dtype=torch.int64,
            axis="tokens",
        )
    )
    register_cfg_graph_slots(registry, 4)
    if path == "replace":
        rebuilt = replace(batch)
    elif path == "registry":
        registry.fill_from(
            batch,
            raw_bs=size,
            padded_bs=size,
            raw_num_tokens=size * 4,
            padded_num_tokens=size * 4,
        )
        rebuilt = registry.extract_buffer(
            padded_bs=size, padded_num_tokens=size * 4, forward_batch_template=batch
        )
    else:
        monkeypatch.setenv(
            "SGLANG_EAGER_INPUT_NO_COPY", "1" if path == "eager_no_copy" else "0"
        )
        runner = object.__new__(EagerRunner)
        runner._eager_registry = registry
        rebuilt = runner.load_batch(batch)
    assert type(rebuilt) is DllmCFGForwardBatch
    assert list(rebuilt.dllm_left_pad_lens_cpu) == list(pads)
    assert torch.equal(rebuilt.positions, batch.positions)
    if path in ("registry", "eager"):
        assert rebuilt.dllm_left_pad_lens.tolist() == list(pads)
        assert (
            rebuilt.positions.data_ptr()
            == registry.get_slot("positions").buffer.data_ptr()
        )


def test_real_decode_load_attaches_and_resets_padded_metadata():
    received = []
    backend = NS()

    def receive(view):
        attach_cfg_graph_views(view, backend._cfg_graph_registry)
        received.append(view)

    backend.init_forward_metadata_out_graph = receive
    runner = tiny_runner(backend)
    pointer = runner.buffer_registry.get_slot("dllm_left_pad_lens").buffer.data_ptr()
    for prefix, pads in [((8, 8, 8), (2, 6, 3)), ((12, 12), (0, 5)), ((4,), (0,))]:
        batch = tiny_batch(prefix, pads)
        assert runner.can_run_graph(batch)
        runner.load_batch(batch)
        view = received[-1]
        assert type(view) is NS
        assert view.batch_size == 4 and view.num_padding == 4 - len(pads)
        assert view.dllm_left_pad_lens_cpu.tolist() == list(pads) + [0] * (
            4 - len(pads)
        )
        assert view.extend_prefix_lens_cpu.tolist() == list(prefix) + [0] * (
            4 - len(pads)
        )
        assert view.extend_seq_lens_cpu.tolist() == [4] * 4
        assert view.dllm_left_pad_lens.data_ptr() == pointer
        assert torch.equal(view.positions[: len(pads) * 4], batch.positions)
        batch.mark_forward_metadata_ready()
        batch.input_ids.add_(1)
        count = len(received)
        runner.load_batch(batch)
        assert (
            len(received) == count
        )  # Real preplanned path only copies tokens/positions.
        assert torch.equal(runner.buffers.input_ids[: len(pads) * 4], batch.input_ids)
