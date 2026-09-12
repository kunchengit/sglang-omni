# SPDX-License-Identifier: Apache-2.0
"""CPU bridge tests using real Torch and the target SGLang registry source.

Set SGLANG_SOURCE_ROOT to a 0.5.20 checkout. CUDA runners and FlashInfer kernels
are explicit doubles; the registry and replay-view builder are unmodified source.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "sglang_omni.models.llada2_uni."


@pytest.fixture
def bridge(monkeypatch):
    source = os.environ.get("SGLANG_SOURCE_ROOT")
    if source is None:
        pytest.skip("set SGLANG_SOURCE_ROOT for real 0.5.20 registry-source tests")
    source = Path(source) / "python/sglang/srt/model_executor"

    def stub(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    @dataclass
    class ForwardBatch:
        batch_size: int
        input_ids: torch.Tensor
        forward_mode: object = None
        positions: torch.Tensor | None = None
        seq_lens: torch.Tensor | None = None
        seq_lens_cpu: torch.Tensor | None = None
        seq_lens_sum: int = 0
        extend_prefix_lens: torch.Tensor | None = None
        extend_prefix_lens_cpu: object = None
        extend_seq_lens: torch.Tensor | None = None
        extend_seq_lens_cpu: object = None
        req_pool_indices: torch.Tensor | None = None
        spec_info: object = None
        out_cache_loc: object = None
        out_cache_loc_virtual: object = None

    stub("sglang.srt.model_executor.forward_batch_info", ForwardBatch=ForwardBatch)
    stub(
        "sglang.srt.model_executor.input_buffers",
        INDEX_SEMANTIC_BUFFERS=set(),
        share_input_buffer=lambda name, tensor: tensor,
    )
    registry = load(
        "sglang.srt.model_executor.cuda_graph_buffer_registry",
        source / "cuda_graph_buffer_registry.py",
    )
    # Execute only this dependency-light function, not the CUDA runner imports.
    parsed = ast.parse((source / "runner/decode_cuda_graph_runner.py").read_text())
    node = next(
        n
        for n in parsed.body
        if isinstance(n, ast.FunctionDef) and n.name == "build_replay_fb_view"
    )
    namespace = {"SimpleNamespace": NS}
    exec(  # noqa: S102 -- execute the explicitly selected local source function
        compile(ast.Module(body=[node], type_ignores=[]), "replay_view", "exec"),
        namespace,
    )

    class Runner:
        def can_run_graph(self, batch):
            return True

        def _build_replay_fb_view(self, **kwargs):
            return namespace["build_replay_fb_view"](**kwargs)

    stub(
        "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
        DecodeCudaGraphRunner=Runner,
    )
    stub("sglang.srt.arg_groups.model_override_base", resolved_view=lambda args: args)
    glue = NS(get=lambda: False)
    stub("sglang.srt.environ", envs=NS(SGLANG_ENABLE_METADATA_GLUE_GRAPH=glue))
    stub(
        "sglang.srt.model_executor.cuda_graph_config",
        Backend=NS(DISABLED="disabled", FULL="full"),
    )
    metadata = load(
        PACKAGE + "cfg_cuda_graph_metadata",
        ROOT / "sglang_omni/models/llada2_uni/cfg_cuda_graph_metadata.py",
    )
    load(
        PACKAGE + "cfg_cuda_graph_config",
        ROOT / "sglang_omni/models/llada2_uni/cfg_cuda_graph_config.py",
    )
    graph = load(
        PACKAGE + "cfg_cuda_graph",
        ROOT / "sglang_omni/models/llada2_uni/cfg_cuda_graph.py",
    )

    class FlashBackend:
        def init_forward_metadata(self, batch):
            self.delegated = True

        def init_forward_metadata_out_graph(self, batch, in_capture=False):
            self.delegated = True

    class Ragged:
        is_cuda_graph_enabled = False
        _custom_mask_buf = None
        _mask_indptr_buf = None

        def begin_forward(self, *args, **kwargs):
            self.plan = kwargs

    stub("flashinfer.prefill", BatchPrefillWithRaggedKVCacheWrapper=Ragged)
    stub(
        "sglang.srt.layers.attention.flashinfer_backend",
        FlashInferAttnBackend=FlashBackend,
        PrefillMetadata=lambda wrappers, **kwargs: NS(wrappers=wrappers, **kwargs),
        merge_state=lambda *args: None,
    )
    stub("sglang.srt.mem_cache.memory_pool", KVWriteLoc=object)
    stub(
        "sglang.srt.arg_groups.choices",
        ATTENTION_BACKEND_CHOICES=[],
        add_attention_backend_choices=lambda names: None,
    )
    stub("sglang.srt.layers.attention.attention_registry", ATTENTION_BACKENDS={})
    attention = load(
        PACKAGE + "cfg_attention_backend",
        ROOT / "sglang_omni/models/llada2_uni/cfg_attention_backend.py",
    )
    return NS(
        meta=metadata,
        graph=graph,
        registry=registry,
        attention=attention,
        ragged=Ragged,
        glue=glue,
        runner=Runner,
    )


def batch_for(bridge, pads=(0, 6, 3), prefix=8, query=4):
    size = len(pads)
    return bridge.meta.DllmCFGForwardBatch(
        batch_size=size,
        input_ids=torch.arange(size * query),
        forward_mode=NS(is_dllm_extend=lambda: True),
        positions=torch.arange(size * query, dtype=torch.int64),
        seq_lens=torch.full((size,), prefix + query, dtype=torch.int32),
        seq_lens_cpu=torch.full((size,), prefix + query, dtype=torch.int32),
        seq_lens_sum=size * (prefix + query),
        extend_prefix_lens=torch.full((size,), prefix, dtype=torch.int32),
        extend_prefix_lens_cpu=[prefix] * size,
        extend_seq_lens=torch.full((size,), query, dtype=torch.int32),
        extend_seq_lens_cpu=[query] * size,
        dllm_left_pad_lens=torch.tensor(pads, dtype=torch.int32),
        dllm_left_pad_lens_cpu=list(pads),
        req_pool_indices=torch.arange(size),
    )


def registry_for(bridge):
    registry = bridge.registry.CudaGraphBufferRegistry(
        device=torch.device("cpu"), max_bs=4, max_num_tokens=16
    )
    registry.register_slot(
        bridge.registry.GraphSlot(
            name="positions",
            shape_fn=lambda _, tokens: (tokens,),
            dtype=torch.int64,
            axis="tokens",
            padding_policy=bridge.registry.PaddingPolicy.ZERO,
        )
    )
    bridge.graph.register_cfg_graph_slots(registry, 4)
    return registry


def fill(registry, batch):
    registry.fill_from(
        batch,
        raw_bs=batch.batch_size,
        padded_bs=4,
        raw_num_tokens=batch.input_ids.numel(),
        padded_num_tokens=16,
    )


@pytest.mark.parametrize(
    "pads,prefix,local,paged",
    [
        ((0, 6), 4, (0, 2), (4, 0)),
        ((0, 6), 8, (0, 0), (8, 2)),
        ((2, 6, 3), 4, (0, 2, 0), (2, 0, 1)),
        ((0,), 0, (0,), (0,)),
        ((0, 10), 0, (0, 4), (0, 0)),
    ],
)
def test_cpu_geometry(bridge, monkeypatch, pads, prefix, local, paged):
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("scalar sync"))
    geometry = bridge.meta.cfg_attention_geometry(batch_for(bridge, pads, prefix))
    assert geometry.local_pad == local
    assert geometry.paged_lens == paged


def test_missing_gpu_host_mirror_is_an_error(bridge):
    batch = batch_for(bridge)
    batch.dllm_left_pad_lens_cpu = None
    batch.dllm_left_pad_lens = torch.empty(3, device="meta")
    with pytest.raises(RuntimeError, match="host-known dllm_left_pad_lens_cpu"):
        bridge.meta.cfg_attention_geometry(batch)


def test_request_host_padding_metadata(bridge):
    batch = batch_for(bridge)
    batch.dllm_left_pad_lens_cpu = None
    batch.reqs = [
        NS(_dllm_left_pad_len=2),
        NS(_dllm_left_pad_len=6, _is_uncond=True),
        NS(_dllm_left_pad_len=3, _is_uncond=True, _is_uncond_img=True),
    ]
    promoted = bridge.meta.as_cfg_forward_batch(batch)
    assert promoted.dllm_left_pad_lens_cpu == [2, 6, 3]
    assert promoted.reqs is batch.reqs
    assert promoted.positions is batch.positions
    assert batch.dllm_left_pad_lens_cpu is None


def test_registry_replace_and_replay_reset(bridge):
    registry = registry_for(bridge)
    pointers = {
        name: registry.get_slot(name).buffer.data_ptr()
        for name in registry.slot_names()
    }
    for pads in [(2, 6, 3), (0, 5), (0,), (0, 2, 4)]:
        batch = batch_for(bridge, pads)
        fill(registry, batch)
        rebuilt = registry.extract_buffer(
            padded_bs=4, padded_num_tokens=16, forward_batch_template=batch
        )
        assert type(rebuilt) is bridge.meta.DllmCFGForwardBatch
        assert replace(rebuilt).dllm_left_pad_lens is rebuilt.dllm_left_pad_lens
        assert rebuilt.dllm_left_pad_lens.tolist() == list(pads) + [0] * (4 - len(pads))
        assert (
            rebuilt.dllm_left_pad_lens_cpu.tolist()
            == rebuilt.dllm_left_pad_lens.tolist()
        )
        assert rebuilt.extend_prefix_lens.tolist() == [8] * len(pads) + [0] * (
            4 - len(pads)
        )
        assert rebuilt.extend_seq_lens.tolist() == [4] * 4
        assert torch.equal(rebuilt.positions[: len(pads) * 4], batch.positions)
    assert pointers == {
        name: registry.get_slot(name).buffer.data_ptr()
        for name in registry.slot_names()
    }


def test_real_replay_view_transports_cfg_fields(bridge):
    batch = batch_for(bridge, (0, 6))
    registry = registry_for(bridge)
    fill(registry, batch)
    runner = object.__new__(bridge.graph.LLaDA2CFGDecodeCudaGraphRunner)
    runner.buffer_registry = registry
    buffers = NS(
        input_ids=torch.arange(16),
        positions=registry.get_slot("positions").buffer,
        req_pool_indices=torch.tensor([0, 1, 0, 0]),
        seq_lens=torch.tensor([12, 12, 4, 4]),
        seq_lens_cpu=torch.tensor([12, 12, 4, 4]),
        mamba_track_indices=None,
    )
    view = runner._build_replay_fb_view(
        forward_batch=batch,
        buffers=buffers,
        bs=4,
        raw_bs=2,
        num_tokens=16,
        seq_len_fill_value=4,
        capture_forward_mode=batch.forward_mode,
        is_encoder_decoder=False,
    )
    assert type(view) is bridge.meta.DllmCFGForwardBatch
    assert view.batch_size == 4 and view.num_padding == 2
    assert view.seq_lens_sum == 32
    assert view.dllm_left_pad_lens_cpu.tolist() == [0, 6, 0, 0]
    assert view.positions.data_ptr() == buffers.positions.data_ptr()
    assert bridge.meta.cfg_attention_geometry(view).paged_lens == (8, 2, 0, 0)


@pytest.mark.parametrize(
    "pads,prefix,query,eligible",
    [
        ((0, 6), 4, 4, False),
        ((0, 6), 8, 4, True),
        ((2, 6, 3), 8, 4, True),
        ((0, 0), 8, 2, False),
    ],
)
def test_graph_eligibility_without_runner_mutation(
    bridge, pads, prefix, query, eligible
):
    runner = object.__new__(bridge.graph.LLaDA2CFGDecodeCudaGraphRunner)
    runner.captured_req_width = 4
    before = vars(runner).copy()
    assert runner.can_run_graph(batch_for(bridge, pads, prefix, query)) is eligible
    assert vars(runner) == before


def attention_for(bridge):
    backend = object.__new__(bridge.attention.LLaDA2CFGFlashInferAttnBackend)
    backend._cfg_prefill_wrapper_ragged = bridge.ragged()
    backend.prefill_wrappers_paged = [object()]
    backend.prefill_cuda_graph_metadata = {4: [object()]}
    backend.prefill_split_tile_size = None
    backend._cfg_graph_cached_pad = torch.zeros(4, dtype=torch.int32)
    backend._cfg_graph_paged_lens = torch.zeros(4, dtype=torch.int32)
    backend.kv_read_tables = object()
    backend.kv_view = object()
    backend.kv_index_translator = NS(
        index_table_for_batch=lambda _batch: backend.kv_view,
        build_index_table=lambda **_kwargs: backend.kv_view,
    )
    calls = []
    backend.indices_updater_prefill = NS(
        call_begin_forward=lambda *args, **kwargs: calls.append((args, kwargs)),
        kv_indptr=[None],
        qo_indptr=[None],
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=2,
        q_data_type=torch.float32,
        data_type=torch.float32,
        prefill_wrapper_ragged=bridge.ragged(),
    )
    return backend, calls


def test_attention_eager_to_graph_plans_and_zero_pad_transition(bridge, monkeypatch):
    backend, calls = attention_for(bridge)
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("scalar sync"))
    backend.init_forward_metadata(batch_for(bridge, (0, 6), prefix=4))
    assert backend._cfg_local_left_pad_active
    args, kwargs = calls[-1]
    assert args[4] == 4
    assert kwargs["kv_view"] is backend.kv_view
    registry = registry_for(bridge)
    pointers = (
        backend._cfg_graph_cached_pad.data_ptr(),
        backend._cfg_graph_paged_lens.data_ptr(),
    )
    for pads in [(2, 6, 3), (0, 5), (0,)]:
        batch = batch_for(bridge, pads)
        fill(registry, batch)
        static = registry.extract_buffer(
            padded_bs=4, padded_num_tokens=16, forward_batch_template=batch
        )
        static.seq_lens = torch.tensor([12] * len(pads) + [4] * (4 - len(pads)))
        backend.init_forward_metadata_out_graph(static)
        assert not backend._cfg_local_left_pad_active
        args, kwargs = calls[-1]
        assert args[1] is backend.prefill_cuda_graph_metadata[4][0]
        assert args[4] == sum(8 - p for p in pads)
        assert args[7].tolist() == list(pads) + [0] * (4 - len(pads))
        assert kwargs["kv_view"] is backend.kv_view
    assert pointers == (
        backend._cfg_graph_cached_pad.data_ptr(),
        backend._cfg_graph_paged_lens.data_ptr(),
    )


def test_attention_rejects_query_local_graph_plan(bridge):
    backend, _ = attention_for(bridge)
    with pytest.raises(ValueError, match="query-local padding"):
        backend.init_forward_metadata_out_graph(batch_for(bridge, (0, 6), prefix=4))


def test_graph_configuration_never_silently_disables_features(bridge):
    args = NS(
        dllm_algorithm="LowConfidenceCFG",
        attention_backend="llada2_uni_cfg_flashinfer",
        cuda_graph_config=NS(decode=NS(backend="full"), prefill=NS(backend="disabled")),
    )
    bridge.graph.validate_cfg_cuda_graph_config(args)
    bridge.glue.get = lambda: True
    with pytest.raises(ValueError, match="METADATA_GLUE"):
        bridge.graph.validate_cfg_cuda_graph_config(args)
    assert args.cuda_graph_config.decode.backend == "full"
    bridge.glue.get = lambda: False
    args.attention_backend = "flashinfer"
    with pytest.raises(ValueError, match="capability registration"):
        bridge.graph.validate_cfg_cuda_graph_config(args)
    assert args.attention_backend == "flashinfer"


def test_missing_upstream_hook_fails_before_capture(bridge, monkeypatch):
    monkeypatch.delattr(bridge.runner, "_build_replay_fb_view")
    with pytest.raises(RuntimeError, match="extension hook"):
        bridge.graph.LLaDA2CFGDecodeCudaGraphRunner(NS())


def test_capture_declares_static_metadata_before_model_capture(bridge, monkeypatch):
    runner = object.__new__(bridge.graph.LLaDA2CFGDecodeCudaGraphRunner)
    runner.buffer_registry = bridge.registry.CudaGraphBufferRegistry(
        device=torch.device("cpu"), max_bs=4, max_num_tokens=16
    )
    runner.captured_req_width = runner.seq_len_fill_value = 4
    runner.enable_two_batch_overlap = runner.enable_pdmux = False
    runner.require_gathered_buffer = False
    runner.capture_forward_mode = NS(is_dllm_extend=lambda: True)
    captured = []
    monkeypatch.setattr(
        bridge.runner,
        "capture",
        lambda self: captured.append(set(self.buffer_registry.slot_names())),
        raising=False,
    )
    runner.capture()
    assert captured == [set(bridge.graph.CFG_GRAPH_FIELDS)]

    def prepare(_self, size, *_args):
        dummy = batch_for(bridge, (0,) * size, prefix=0)
        dummy.hisparse_coordinator = None
        return dummy, "backend", None

    monkeypatch.setattr(
        bridge.runner,
        "capture_prepare",
        prepare,
        raising=False,
    )
    for size in (2, 3):
        cfg, backend, _ = runner.capture_prepare(size)
        assert backend == "backend"
        assert type(cfg) is bridge.meta.DllmCFGForwardBatch
        assert cfg.dllm_left_pad_lens_cpu.tolist() == [0] * size
        assert cfg.extend_prefix_lens_cpu.tolist() == [0] * size
        assert cfg.extend_seq_lens_cpu.tolist() == [4] * size


@pytest.mark.parametrize("prefix,query,pad", [(-1, 4, 0), (0, 0, 0), (0, 4, -1)])
def test_invalid_host_geometry_is_rejected(bridge, prefix, query, pad):
    with pytest.raises(ValueError, match="nonnegative pad/prefix and a nonempty query"):
        bridge.meta.cfg_attention_geometry(batch_for(bridge, (pad,), prefix, query))


def test_load_promotes_without_mutating_batch(bridge, monkeypatch):
    batch = batch_for(bridge)
    seen = []
    monkeypatch.setattr(
        bridge.runner,
        "load_batch",
        lambda self, promoted, proxy: seen.append(promoted),
        raising=False,
    )
    runner = object.__new__(bridge.graph.LLaDA2CFGDecodeCudaGraphRunner)
    runner.load_batch(batch)
    assert seen[0] is not batch
    assert type(seen[0]) is bridge.meta.DllmCFGForwardBatch
    assert seen[0].input_ids is batch.input_ids
    assert seen[0].positions is batch.positions


def alignment_function():
    path = ROOT / "sglang_omni/models/llada2_uni/request_builders.py"
    parsed = ast.parse(path.read_text())
    node = next(
        n
        for n in parsed.body
        if isinstance(n, ast.FunctionDef) and n.name == "_align_cfg_branch_group"
    )
    namespace = {}
    exec(  # noqa: S102 -- compile the explicitly selected local helper
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[node.name]


@pytest.mark.parametrize(
    "branches,pads,expected_pads",
    [
        (
            {"conditional": [1, 2, 3], "uncond": [4]},
            {},
            {"conditional": 0, "uncond": 2},
        ),
        (
            {"conditional": [1], "uncond": [2, 3, 4]},
            {},
            {"conditional": 2, "uncond": 0},
        ),
        (
            {"conditional": [1], "uncond": [2, 3], "uncond_img": [99, 4, 5]},
            {"uncond_img": 1},
            {"conditional": 2, "uncond": 1, "uncond_img": 1},
        ),
    ],
)
def test_longest_branch_alignment_source(branches, pads, expected_pads):
    original = {key: list(value) for key, value in branches.items()}
    aligned, lengths = alignment_function()(
        tokenizer=NS(mask_token_id=99), branches=branches, existing_left_pad_lens=pads
    )
    assert lengths == expected_pads
    assert len({len(ids) for ids in aligned.values()}) == 1
    assert branches == original


def test_alignment_requires_mask_only_when_padding():
    fn = alignment_function()
    assert fn(
        tokenizer=NS(),
        branches={"conditional": [1], "uncond": [2]},
        existing_left_pad_lens={},
    ) == ({"conditional": [1], "uncond": [2]}, {"conditional": 0, "uncond": 0})
    with pytest.raises(ValueError, match="mask_token_id"):
        fn(
            tokenizer=NS(),
            branches={"conditional": [1], "uncond": [2, 3]},
            existing_left_pad_lens={},
        )


def test_preprocessor_defers_longer_companion_to_group_alignment():
    path = ROOT / "sglang_omni/models/llada2_uni/components/preprocessor.py"
    tree = ast.parse(path.read_text())
    helper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "align_cfg_unconditional_input_ids"
    )
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LLaDA2Preprocessor"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_set_cfg_branch"
    ]
    namespace = {}
    exec(  # noqa: S102 -- compile the selected local preprocessing methods
        compile(ast.Module(body=[helper, cls], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    processor = namespace[cls.name]()
    processor._tokenizer = NS(mask_token_id=99)
    for companion, expected_pad in [([2], 1), ([2, 3], 0), ([2, 3, 4], 0)]:
        state = {}
        processor._set_cfg_branch(state, [0, 1], companion)
        assert state["uncond_left_pad_len"] == expected_pad
        branches, pads = alignment_function()(
            tokenizer=processor._tokenizer,
            branches={"conditional": [0, 1], "uncond": state["uncond_input_ids"]},
            existing_left_pad_lens={"uncond": expected_pad},
        )
        assert len(branches["conditional"]) == len(branches["uncond"])
        assert pads["conditional"] == max(len(companion) - 2, 0)


def test_model_runner_selection_is_scoped(bridge):
    path = ROOT / "sglang_omni/model_runner/sglang_model_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SGLModelRunner"
    )
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_decode_cuda_graph_runner_cls"
    ]
    original = object()

    class ModelRunner:
        def _decode_cuda_graph_runner_cls(self):
            return original

    namespace = {"ModelRunner": ModelRunner}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    for algorithm in (None, "LowConfidence", "LowConfidenceCFG"):
        runner = namespace["SGLModelRunner"]()
        runner.server_args = NS(dllm_algorithm=algorithm)
        assert runner._decode_cuda_graph_runner_cls() is (
            bridge.graph.LLaDA2CFGDecodeCudaGraphRunner
            if algorithm == "LowConfidenceCFG"
            else original
        )
