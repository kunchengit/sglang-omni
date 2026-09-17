# SPDX-License-Identifier: Apache-2.0
"""CPU CFG contracts with explicit doubles for CUDA-only SGLang/FlashInfer APIs.

These exercise real scheduler/algorithm/backend Python code and Torch tensors;
they do not claim FlashInfer kernel or end-to-end model validation.
"""

from __future__ import annotations

import importlib.util
import sys
from array import array
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


class ReqDouble:
    def __init__(self, rid, origin_input_ids=None, **kwargs):
        self.rid = rid
        self.constructor_input_ids = origin_input_ids
        self.origin_input_ids = array("q", origin_input_ids or [1, 2])
        self.output_ids = array("q")
        self.full_untruncated_fill_ids = self.origin_input_ids + array("q", [9] * 4)
        self.extend_range = NS(start=0, end=4, length=4)
        self.kv = NS(holds_kv=True, cache_protected_len=0, kv_allocated_len=4)
        self.last_node = rid
        self.lock_receipt = object()
        self.dllm_phase = "decode"
        self.dllm_block_offset = 0
        self.dllm_incomplete_ids = array("q")
        self.dllm_algo_state = None
        self.finished_reason = None
        self.finish_after = 100
        self.tokenizer = None
        self.__dict__.update(kwargs)

    @property
    def output_ids_through_stop(self):
        return self.output_ids[: self.finish_after]

    def finished(self):
        return self.finished_reason is not None

    def is_dllm_prefill(self):
        return self.dllm_phase == "prefill"

    def init_next_round_input(self, *_):
        self.dllm_block_offset += 4
        self.kv.cache_protected_len = 20

    def update_finish_state(self, **_):
        if len(self.output_ids) >= self.finish_after:
            self.finished_reason = NS(to_json=lambda: {"type": "length"})


@pytest.fixture
def modules(monkeypatch):
    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    class AlgorithmBase:
        def __init__(self, config):
            self.block_size = config.block_size
            self.mask_id = config.mask_id
            self.fdfo = config.first_done_first_out_mode

    @dataclass
    class ForwardBatchDouble:
        batch_size: int
        input_ids: torch.Tensor

        @classmethod
        def init_new(cls, batch, runner, **kwargs):
            return cls(batch_size=batch.batch_size, input_ids=batch.input_ids)

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def normalize(self, _):
            self.normalized = True

        def verify(self, _):
            assert self.normalized

    class RaggedWrapper:
        is_cuda_graph_enabled = False

        def __init__(self, *_args, **_kwargs):
            self._custom_mask_buf = None
            self._mask_indptr_buf = None

        def begin_forward(self, *args, **kwargs):
            self.plan = (args, kwargs)

    class FlashBackend:
        def init_forward_metadata(self, _):
            self.delegated = True

    config = NS(page_size=1, max_prefill_tokens=128, chunked_prefill_size=4)
    released = []

    def release(req, cache):
        assert req.kv.holds_kv
        released.append(req.rid)
        req.kv.holds_kv = False

    stub("sglang.srt.dllm.algorithm", algo_name_to_cls={})
    stub(
        "sglang.srt.dllm.algorithm.base",
        DllmAlgorithm=AlgorithmBase,
        DllmRunOutput=tuple,
    )
    stub("sglang.srt.dllm.config", DllmConfig=object)
    stub("sglang.srt.layers.logits_processor", LogitsProcessorOutput=object)
    stub(
        "sglang.srt.model_executor.forward_batch_info", ForwardBatch=ForwardBatchDouble
    )
    stub("sglang.srt.model_executor.model_runner", ModelRunner=object)
    stub("sglang.srt.managers.schedule_batch", Req=ReqDouble, ScheduleBatch=object)
    stub(
        "sglang.srt.managers.schedule_policy",
        PrefillAdder=object,
        AddReqResult=NS(CONTINUE=0, NO_TOKEN=1, OTHER=2),
    )
    stub("sglang.srt.mem_cache.common", release_kv_cache=release)
    stub(
        "sglang.srt.runtime_context",
        get_schedule=lambda: config,
    )
    stub("sglang.srt.speculative.spec_info", SpeculativeAlgorithm=NS(NONE=None))
    stub("sglang.srt.sampling.sampling_params", SamplingParams=SamplingParams)
    stub(
        "sglang_omni.model_runner.base", resolve_deferred_prefill_inputs=lambda *_: None
    )
    load("sglang_omni.scheduling.messages", "sglang_omni/scheduling/messages.py")
    choices = ["flashinfer", "llada2_cfg_flashinfer"]
    registry = {name: object() for name in choices}
    stub(
        "sglang.srt.server_args",
        ATTENTION_BACKEND_CHOICES=choices,
        add_attention_backend_choices=choices.extend,
    )
    stub("sglang.srt.layers.attention.attention_registry", ATTENTION_BACKENDS=registry)
    stub("sglang.srt.mem_cache.memory_pool", KVWriteLoc=lambda *args: ("write", *args))
    stub("flashinfer.prefill", BatchPrefillWithRaggedKVCacheWrapper=RaggedWrapper)
    stub(
        "sglang.srt.layers.attention.flashinfer_backend",
        FlashInferAttnBackend=FlashBackend,
        PrefillMetadata=lambda wrappers, **kwargs: NS(
            prefill_wrappers=wrappers, swa_out_cache_loc=None, **kwargs
        ),
        merge_state=lambda *_: None,
    )
    stub(
        "sglang.srt.arg_groups.model_override_base",
        resolved_view=lambda x: x,
        attention_backends_of=lambda x: (x.attention_backend, x.attention_backend),
    )
    stub(
        "sglang.srt.model_executor.cuda_graph_config",
        Backend=NS(DISABLED="disabled", FULL="full"),
    )
    stub(
        "sglang.srt.environ",
        envs=NS(SGLANG_ENABLE_METADATA_GLUE_GRAPH=NS(get=lambda: False)),
    )
    stub(
        "sglang_omni.vendor.sglang.server_args",
        override_server_args=lambda *_a, **_k: pytest.fail(
            "CFG validation must not override server args"
        ),
    )
    algo = load(
        "sglang_omni.models.llada2_uni.algorithm.low_confidence_cfg",
        "sglang_omni/models/llada2_uni/algorithm/low_confidence_cfg.py",
    )
    attn = load(
        "sglang_omni.models.llada2_uni.cfg_attention_backend",
        "sglang_omni/models/llada2_uni/cfg_attention_backend.py",
    )
    scheduler = load("_cfg_scheduler", "sglang_omni/scheduling/dllm_scheduler.py")
    load(
        "sglang_omni.models.llada2_uni.cfg_cuda_graph_config",
        "sglang_omni/models/llada2_uni/cfg_cuda_graph_config.py",
    )
    bootstrap = load("_cfg_bootstrap", "sglang_omni/models/llada2_uni/bootstrap.py")
    return NS(
        algo=algo,
        attn=attn,
        scheduler=scheduler,
        bootstrap=bootstrap,
        config=config,
        released=released,
        registry=registry,
        choices=choices,
        wrapper=RaggedWrapper,
    )


def scheduler_for(modules):
    return modules.scheduler.DllmScheduler(
        tp_worker=NS(tp_rank=0),
        tree_cache=NS(dec_lock_ref=lambda *_: None),
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        server_args=None,
        model_config=None,
        dllm_config=NS(
            block_size=4, max_running_requests=3, first_done_first_out_mode=False
        ),
        request_builder=lambda req: NS(req=req),
        result_adapter=lambda data: data,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_decode_reference_precision(modules, dtype):
    config = NS(
        block_size=4, mask_id=9, first_done_first_out_mode=False, algorithm_config={}
    )
    algorithm = modules.algo.LowConfidenceCFG(config)
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, -1.0]], dtype=dtype)
    ids, confidence = algorithm._argmax_confidence(logits)
    expected = (
        logits.softmax(-1).gather(-1, logits.argmax(-1, keepdim=True)).squeeze(-1)
    )
    torch.testing.assert_close(confidence, expected, rtol=0, atol=0)
    assert confidence.dtype == dtype and ids.tolist() == [2, 0]


def group(size=3, prompt=None):
    reqs = [ReqDouble("cond", prompt)]
    for i in range(1, size):
        reqs.append(
            ReqDouble(f"cond-u{i}", prompt, _is_uncond=True, _is_uncond_img=i == 2)
        )
    for req in reqs:
        req._cfg_group_rid = "cond"
    return reqs


def track(scheduler, reqs):
    scheduler._cond_to_unconds = {reqs[0].rid: [r.rid for r in reqs[1:]]}
    scheduler._uncond_to_cond = {r.rid: reqs[0].rid for r in reqs[1:]}
    scheduler._uncond_rids = {r.rid for r in reqs[1:]}


@pytest.mark.parametrize("size", [1, 2, 3])
@pytest.mark.parametrize("rescale", [0.0, 0.7])
def test_guidance_and_five_field_contract(modules, size, rescale):
    config = NS(
        block_size=4,
        mask_id=9,
        first_done_first_out_mode=False,
        algorithm_config={
            "threshold": 1.0,
            "image_token_offset": 3,
        },
    )
    algorithm = modules.algo.LowConfidenceCFG(config)
    reqs = group(size)
    if size == 1:
        del reqs[0]._cfg_group_rid
    reqs[0]._task_kind = "edit" if size == 3 else "t2i"
    reqs[0]._dllm_steps = 3
    reqs[0]._cfg_scale = 2.0
    reqs[0]._cfg_image_scale = 1.5
    reqs[0]._cfg_rescale = rescale
    ids = torch.tensor([1, 9, 9, 9] * size)
    if size > 1:
        ids[4] = 9  # A prompt pad, not an output mask.
    logits = torch.tensor(
        [
            [100.0, 1.0, 1.0, 2.0, 1.0, 0.0],
            [100.0, 1.0, 1.0, 3.0, 0.0, 1.0],
            [100.0, 1.0, 1.0, 5.0, 2.0, 0.0],
        ]
    )[:size]
    expected = logits[0].clone()
    if size >= 2:
        expected = logits[1] + 2 * (logits[0] - logits[1])
    if size == 3:
        expected += 1.5 * (logits[1] - logits[2])
    if size >= 2 and rescale:
        normalized = expected * (logits[0].std() / (expected.std() + 1e-6))
        expected = rescale * normalized + (1 - rescale) * expected
    expected[:3] = -torch.inf
    calls = []

    def forward(batch, **_):
        calls.append(batch.input_ids.clone())
        return NS(
            logits_output=NS(full_logits=logits.repeat_interleave(4, dim=0).clone()),
            can_run_graph=False,
        )

    batch = NS(input_ids=ids, batch_size=size, reqs=reqs)
    result = algorithm.run(NS(forward=forward), batch, algo_states=None)
    assert len(result) == 5 and result[2:] == (None, None, False)
    assert len(result[1]) == size
    assert all(row.tolist() == [expected.argmax().item()] * 3 for row in result[1])
    assert all(not (row == 9).any() for row in result[1])
    assert len(calls) >= 2
    if size > 1:
        assert ids[4].item() == 9


def test_cfg_prefill_does_not_denoise_padding(modules):
    algorithm = modules.algo.LowConfidenceCFG(
        NS(
            block_size=4,
            mask_id=9,
            first_done_first_out_mode=False,
            algorithm_config={},
        )
    )
    reqs = group(2)
    for req in reqs:
        req.dllm_phase = "prefill"
    ids = torch.tensor([1, 2, 3, 4, 9, 9, 3, 4])
    original = ids.clone()
    calls = []

    def forward(*_, **__):
        calls.append(1)
        return NS(logits_output=None, can_run_graph=False)

    result = algorithm.run(
        NS(forward=forward), NS(reqs=reqs, batch_size=2, input_ids=ids)
    )
    assert result == (None, [], None, None, False)
    assert torch.equal(ids, original) and len(calls) == 1


def test_malformed_cfg_and_fdfo_are_rejected(modules):
    config = NS(
        block_size=4, mask_id=9, first_done_first_out_mode=True, algorithm_config={}
    )
    with pytest.raises(ValueError, match="FDFO"):
        modules.algo.LowConfidenceCFG(config)
    config.first_done_first_out_mode = False
    algo = modules.algo.LowConfidenceCFG(config)
    reqs = group(2)
    reqs[1]._cfg_group_rid = "different"
    with pytest.raises(RuntimeError, match="Malformed CFG"):
        algo.run(None, NS(reqs=reqs, batch_size=2))


@pytest.mark.parametrize("name", ["_cfg_scale", "_cfg_rescale", "_cfg_image_scale"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_cfg_values_are_rejected(modules, name, value):
    algorithm = modules.algo.LowConfidenceCFG(
        NS(
            block_size=4,
            mask_id=9,
            first_done_first_out_mode=False,
            algorithm_config={},
        )
    )
    reqs = group(3)
    setattr(reqs[0], name, value)

    with pytest.raises(ValueError, match=f"{name} must be finite"):
        algorithm.run(None, NS(reqs=reqs, batch_size=3))


def test_companion_constructor_uses_arrays_and_dllm_config(modules):
    scheduler = scheduler_for(modules)
    cond = ReqDouble(
        "cond",
        sampling_params=NS(max_new_tokens=8),
        vocab_size=20,
        eos_token_ids={0},
        dllm_config=scheduler.dllm_config,
    )
    scheduler._create_uncond_companion(cond, [9, 2], 1, "-uncond", False)
    companion = scheduler._waiting_queue[0]
    assert isinstance(companion.constructor_input_ids, array)
    assert isinstance(companion.origin_input_ids, array)
    assert companion.dllm_config is cond.dllm_config
    assert companion._cfg_group_rid == cond.rid
    assert companion.sampling_params.normalized
    with pytest.raises(ValueError, match="physically aligned"):
        scheduler._create_uncond_companion(cond, [1], 0, "-bad", False)


@pytest.mark.parametrize(
    "offset,pad,expected",
    [(0, 2, [0, 0, 0, 1]), (4, 2, [2, 3, 4, 5]), (4, 6, [0, 0, 0, 1])],
)
def test_cfg_positions(modules, offset, pad, expected):
    scheduler = scheduler_for(modules)
    reqs = group(2)
    reqs[1]._dllm_left_pad_len = pad
    batch = NS(
        positions=torch.arange(offset, offset + 4).repeat(2),
        extend_seq_lens_cpu=[4, 4],
        seq_lens=torch.tensor([offset + 4] * 2),
        forward_mode=NS(is_extend=lambda: True),
    )
    scheduler._apply_cfg_padding_metadata(batch, NS(reqs=reqs))
    assert batch.positions[4:].tolist() == expected
    assert batch.dllm_left_pad_lens.tolist() == [0, pad]


@pytest.mark.parametrize("staging", [False, True])
@pytest.mark.parametrize("page_size", [1, 2, 4])
def test_atomic_admission_restores_nested_kv_and_retries(
    modules, monkeypatch, staging, page_size
):
    modules.config.page_size = page_size
    scheduler = scheduler_for(modules)
    reqs = group()
    track(scheduler, reqs)
    scheduler._staging_queue = reqs.copy() if staging else []
    scheduler._waiting_queue = [] if staging else reqs.copy()
    released_locks = []
    scheduler.tree_cache.dec_lock_ref = lambda node, receipt: released_locks.append(
        node
    )
    limit = [2]

    class Adder:
        def __init__(self, *_args, **kwargs):
            assert kwargs["prefill_max_requests"] == 3
            self.can_run_list = []

        def add(self, req, **_):
            self.can_run_list.append(req)
            return 1 if len(self.can_run_list) == limit[0] else 0

        add_one_req = add
        add_dllm_staging_req = add

    monkeypatch.setattr(modules.scheduler, "PrefillAdder", Adder)
    monkeypatch.setattr(
        modules.scheduler,
        "ScheduleBatch",
        NS(init_new=lambda **kw: NS(reqs=kw["reqs"], prepare_for_extend=lambda: None)),
    )
    assert scheduler._schedule_next_batch() is None
    assert all(
        req.dllm_block_offset == 0 and req.kv.cache_protected_len == 0 for req in reqs
    )
    assert released_locks == ([] if staging else ["cond", "cond-u1"])
    limit[0] = 3
    reqs[0].dllm_phase = "prefill"
    batch = scheduler._schedule_next_batch()
    assert batch.reqs == reqs
    assert all(req.dllm_phase == "prefill" for req in reqs)


@pytest.mark.parametrize("abort_index", [0, 1, 2])
def test_abort_any_branch_releases_whole_group_once(modules, abort_index):
    scheduler = scheduler_for(modules)
    reqs = group()
    track(scheduler, reqs)
    scheduler._staging_queue = reqs.copy()
    scheduler.abort(reqs[abort_index].rid)
    scheduler._drain_and_purge()
    assert sorted(modules.released) == sorted(r.rid for r in reqs)
    assert not scheduler._staging_queue and not scheduler._uncond_rids
    assert not scheduler._cond_to_unconds and not scheduler._uncond_to_cond


def test_completion_updates_all_fill_ids_and_emits_only_cond(modules):
    scheduler = scheduler_for(modules)
    reqs = group()
    track(scheduler, reqs)
    scheduler._staging_queue = reqs.copy()
    scheduler._rid_to_req_data["cond"] = NS(req=reqs[0])
    reqs[0].finish_after = 2
    batch = NS(reqs=reqs, filter_batch=lambda **_: None)
    scheduler._apply_results(
        batch,
        NS(
            next_token_ids=[[6, 7]] * 3,
            accept_length_per_req_cpu=None,
            dllm_algo_state=None,
        ),
    )
    assert all(list(r.full_untruncated_fill_ids[2:4]) == [6, 7] for r in reqs)
    scheduler._post_step(batch)
    message = scheduler.outbox.get_nowait()
    assert message.request_id == "cond" and list(message.data.output_ids) == [6, 7]
    assert scheduler.outbox.empty()
    assert len(modules.released) == 3 and not scheduler._staging_queue
    assert not scheduler._uncond_rids and not scheduler._orphaned_uncond_rids


def test_staged_cfg_retains_rows_for_abort(modules):
    scheduler = scheduler_for(modules)
    reqs = group()
    track(scheduler, reqs)
    scheduler._staging_queue = reqs.copy()
    scheduler.tree_cache.cache_unfinished_req = lambda *_a, **_k: None
    scheduler.req_to_token_pool = NS(
        free=lambda _: pytest.fail("CFG row released early")
    )
    scheduler._post_step(NS(reqs=reqs, filter_batch=lambda **_: None))
    assert all(req.kv.holds_kv for req in reqs)
    scheduler.abort("cond")
    scheduler._drain_and_purge()
    assert len(modules.released) == 3


def test_attention_local_and_cached_pad_masks(modules):
    backend = object.__new__(modules.attn.LLaDA2CFGFlashInferAttnBackend)
    backend._cfg_prefill_wrapper_ragged = modules.wrapper()
    backend.num_wrappers = 1
    backend.prefill_wrappers_paged = [object()]
    backend.prefill_split_tile_size = None
    calls = []
    kv_view = object()
    backend.kv_index_translator = NS(
        index_table_for_batch=lambda _batch: kv_view,
    )
    backend.indices_updater_prefill = NS(
        call_begin_forward=lambda *a, **kw: calls.append((a, kw)),
        kv_indptr=[None],
        qo_indptr=[None],
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=2,
        q_data_type=torch.float32,
        data_type=torch.float32,
        prefill_wrapper_ragged=modules.wrapper(),
    )
    batch = NS(
        dllm_left_pad_lens=torch.tensor([0, 6]),
        forward_mode=NS(is_dllm_extend=lambda: True),
        seq_lens=torch.tensor([8, 8]),
        extend_prefix_lens=torch.tensor([4, 4]),
        req_pool_indices=torch.tensor([0, 1]),
    )
    backend.init_forward_metadata(batch)
    args, kwargs = calls[-1]
    assert args[3].tolist() == [4, 0]  # Real cached keys only.
    assert args[7].tolist() == [0, 4]  # Skip cached pads.
    assert kwargs["kv_view"] is kv_view
    mask = backend._cfg_prefill_wrapper_ragged.plan[1]["custom_mask"].reshape(2, 4, 4)
    assert mask[0].all()
    assert not mask[1, 2:, :2].any()
    assert mask[1, 0, 0] and mask[1, 1, 1]
    assert mask[1, :, 2:].all()
    batch.seq_lens += 4
    batch.extend_prefix_lens += 4
    backend.init_forward_metadata(batch)
    assert not backend._cfg_local_left_pad_active
    assert calls[-1][0][3].tolist() == [8, 2]
    batch.dllm_left_pad_lens.zero_()
    backend.init_forward_metadata(batch)
    assert backend.delegated


@pytest.mark.parametrize("page_size", [1, 2, 4])
def test_registration_is_model_specific_and_graph_boundary_is_explicit(
    modules, page_size
):
    original = modules.registry.copy()
    modules.bootstrap.register_llada2_uni_cfg()
    modules.bootstrap.register_llada2_uni_cfg()
    for name, factory in original.items():
        assert modules.registry[name] is factory
    assert modules.choices.count("llada2_uni_cfg_flashinfer") == 1
    cfg = NS(
        dllm_algorithm="LowConfidenceCFG",
        dllm_fdfo=False,
        page_size=page_size,
        attention_backend="llada2_uni_cfg_flashinfer",
        cuda_graph_config=NS(
            decode=NS(backend="disabled"), prefill=NS(backend="disabled")
        ),
    )
    modules.bootstrap._validate_cfg(cfg)
    cfg.cuda_graph_config.decode.backend = "full"
    modules.bootstrap._validate_cfg(cfg)
    assert cfg.cuda_graph_config.decode.backend == "full"
    cfg.cuda_graph_config.prefill.backend = "full"
    with pytest.raises(ValueError, match="does not support prefill"):
        modules.bootstrap._validate_cfg(cfg)
    cfg.dllm_algorithm = "LowConfidence"
    modules.bootstrap._validate_cfg(cfg)  # Text variant unchanged.


@pytest.mark.parametrize("cached", [False, True])
def test_attention_uses_backend_pool_and_kv_write_loc(modules, monkeypatch, cached):
    backend = object.__new__(modules.attn.LLaDA2CFGFlashInferAttnBackend)
    backend._cfg_local_left_pad_active = True
    backend._cfg_has_cached_prefix = cached
    backend.forward_metadata = NS(swa_out_cache_loc=None)
    current = torch.ones(4, 1, 2)
    prefix = torch.full_like(current, 2)
    backend._cfg_prefill_wrapper_ragged = NS(
        forward=lambda *a, **kw: current,
        forward_return_lse=lambda *a, **kw: (current, torch.zeros(4, 1)),
    )
    backend.prefill_wrappers_paged = [
        NS(forward_return_lse=lambda *a, **kw: (prefix, torch.zeros(4, 1)))
    ]
    monkeypatch.setattr(
        modules.attn, "merge_state", lambda a, sa, b, sb: ((a + b) / 2, sa)
    )
    writes = []
    backend.token_to_kv_pool = NS(
        get_kv_buffer=lambda _: object(), set_kv_buffer=lambda *a: writes.append(a)
    )
    backend._kv_write_scales = lambda _: (None, None)
    layer = NS(
        tp_q_head_num=1,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=2,
        scaling=1.0,
        logit_cap=0.0,
        layer_id=0,
        k_scale_float=None,
        v_scale_float=None,
        is_cross_attention=False,
    )
    # ForwardBatch is not the owner of the KV pool.
    batch = NS(out_cache_loc=torch.arange(4))
    result = backend.forward_extend(current, current, current, layer, batch)
    assert torch.equal(
        result, (current if not cached else (current + prefix) / 2).view(4, 2)
    )
    assert len(writes) == 1
    assert writes[0][1][0] == "write" and writes[0][1][2] is None
    assert torch.equal(writes[0][1][1], batch.out_cache_loc)


@pytest.mark.parametrize("page_size,required", [(1, 19), (2, 19), (4, 21)])
def test_capacity_uses_allocator_page_charge_and_full_group(
    modules, page_size, required
):
    scheduler = scheduler_for(modules)
    modules.config.page_size = page_size
    reqs = group(prompt=[1] * 6)
    modules.config.max_prefill_tokens = required - 1
    with pytest.raises(RuntimeError, match=f"at least {required}"):
        scheduler._validate_request_group_capacity(reqs)
    modules.config.max_prefill_tokens = required
    scheduler._validate_request_group_capacity(reqs)


def test_mismatched_branch_span_never_reaches_forward(modules, monkeypatch):
    scheduler = scheduler_for(modules)
    reqs = group()
    track(scheduler, reqs)
    scheduler._staging_queue = reqs.copy()

    class Adder:
        def __init__(self, *_a, **_kw):
            self.can_run_list = []

        def add_dllm_staging_req(self, req):
            self.can_run_list.append(req)
            req.extend_range = NS(
                start=0,
                end=2 if req is reqs[-1] else 4,
                length=2 if req is reqs[-1] else 4,
            )
            return 0

    monkeypatch.setattr(modules.scheduler, "PrefillAdder", Adder)
    assert scheduler._schedule_next_batch() is None
    assert all(req.extend_range.length == 4 for req in reqs)
    assert all(req.dllm_block_offset == 0 for req in reqs)


def test_schedule_to_forward_batch_keeps_deferred_input_order(modules, monkeypatch):
    scheduler = scheduler_for(modules)
    reqs = group(2)
    batch = NS(reqs=reqs)
    forward_batch = NS()
    calls = []
    scheduler._running = True
    scheduler._drain_and_purge = lambda: None
    scheduler._schedule_next_batch = lambda: batch
    scheduler._apply_results = lambda *_: calls.append("apply")

    def stop(_):
        scheduler._running = False

    scheduler._post_step = stop

    def initialize(value, runner, *, return_hidden_states_before_norm):
        assert value is batch and not return_hidden_states_before_norm
        assert calls == ["resolve"]
        calls.append("init")
        return forward_batch

    def forward(value, *, batch):
        assert value is forward_batch and value.reqs is reqs
        calls.append("forward")
        return object()

    scheduler.tp_worker = NS(
        model_runner=NS(device="cpu"), forward_batch_generation=forward
    )
    monkeypatch.setattr(modules.scheduler, "DllmForwardBatch", NS(init_new=initialize))
    monkeypatch.setattr(
        modules.scheduler,
        "resolve_deferred_prefill_inputs",
        lambda *_: calls.append("resolve"),
    )
    scheduler._event_loop()
    assert calls == ["resolve", "init", "forward", "apply"]


def test_dllm_subclass_declares_eager_metadata(modules):
    batch = modules.scheduler.DllmForwardBatch.init_new(
        NS(batch_size=2, input_ids=torch.arange(8)),
        None,
        return_hidden_states_before_norm=False,
    )
    batch.reqs = group(2)
    batch.dllm_left_pad_lens = torch.tensor([0, 2])
    rebuilt = replace(batch)
    assert type(rebuilt) is modules.scheduler.DllmForwardBatch
    assert rebuilt.reqs is batch.reqs
    assert rebuilt.dllm_left_pad_lens is batch.dllm_left_pad_lens


def test_plain_request_is_not_combined_with_waiting_cfg_group(modules):
    scheduler = scheduler_for(modules)
    reqs = group()
    track(scheduler, reqs)
    plain = ReqDouble("plain")
    assert scheduler._get_request_group([plain, *reqs]) == [plain]
    assert scheduler._get_request_group([*reqs, plain]) == reqs


def test_sync_and_fdfo_result_paths_remain_available_without_cfg(modules):
    scheduler = scheduler_for(modules)
    req = ReqDouble("plain")
    scheduler._apply_results(
        NS(reqs=[req]),
        NS(next_token_ids=[6, 7], accept_length_per_req_cpu=None, dllm_algo_state=None),
    )
    assert list(req.output_ids) == [6, 7]
    assert list(req.full_untruncated_fill_ids[2:4]) == [6, 7]
    scheduler.dllm_config.first_done_first_out_mode = True
    state = {"round": 1}
    scheduler._apply_results(
        NS(reqs=[req]),
        NS(
            next_token_ids=[[9, 9, 6, 7]],
            accept_length_per_req_cpu=[0],
            dllm_algo_state=[state],
        ),
    )
    assert list(req.output_ids) == [6, 7]
    assert (
        list(req.dllm_incomplete_ids) == [9, 9, 6, 7] and req.dllm_algo_state is state
    )
