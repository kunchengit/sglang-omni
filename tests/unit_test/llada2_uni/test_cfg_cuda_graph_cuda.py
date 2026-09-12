# SPDX-License-Identifier: Apache-2.0
"""Opt-in tiny FlashInfer graph test; no thinker/model weights are loaded.

SGLANG_TEST_CFG_CUDA=1 makes missing runtime/CUDA/FlashInfer a failure, not a skip.
Select one available GPU with CUDA_VISIBLE_DEVICES; workspace is 32 MiB.
"""

import os
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

if os.environ.get("SGLANG_TEST_CFG_CUDA") != "1":
    pytest.skip(
        "set SGLANG_TEST_CFG_CUDA=1 for real FlashInfer capture/replay",
        allow_module_level=True,
    )
if not torch.cuda.is_available():
    raise RuntimeError("SGLANG_TEST_CFG_CUDA=1 requires a CUDA device")

from flashinfer.prefill import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.kv_index_translator import KVIndexTranslator

from sglang_omni.models.llada2_uni.cfg_attention_backend import (
    LLaDA2CFGFlashInferAttnBackend,
)
from tests.unit_test.llada2_uni.cfg_graph_test_utils import tiny_batch, tiny_runner


def tiny_attention(page_size):
    device = "cuda"
    backend = object.__new__(LLaDA2CFGFlashInferAttnBackend)
    backend.workspace_buffer = torch.empty(
        32 * 1024 * 1024, dtype=torch.uint8, device=device
    )
    backend.num_wrappers = 1
    backend.max_context_len = 32
    backend.skip_prefill = backend.use_sliding_window_kv_pool = False
    backend.prefill_backend = "fa2"
    backend.prefill_split_tile_size = None
    backend.prefill_uses_dequant_workspace = False
    backend.is_dllm_model = True
    backend.is_multimodal = backend.enable_mis = backend.enable_deterministic = False
    backend.use_paged = False
    backend.dq_page_table = None
    backend.page_size = page_size
    backend.kv_indptr = [torch.zeros(5, dtype=torch.int32, device=device)]
    backend.qo_indptr = [torch.zeros(5, dtype=torch.int32, device=device)]
    backend.kv_last_page_len = torch.ones(4, dtype=torch.int32, device=device)
    backend.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
        backend.workspace_buffer, "NHD", backend="fa2"
    )
    backend._cfg_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
        backend.workspace_buffer, "NHD", backend="fa2"
    )
    backend.prefill_wrappers_paged = [
        BatchPrefillWithPagedKVCacheWrapper(
            backend.workspace_buffer, "NHD", backend="fa2"
        )
    ]
    backend.prefill_cuda_graph_metadata = {}
    table = torch.arange(128, dtype=torch.int32, device=device).reshape(4, 32)
    cache = tuple(
        torch.randn(128, 2, 64, dtype=torch.float16, device=device) for _ in range(2)
    )
    pool = NS(get_kv_buffer=lambda layer_id: cache)
    backend.token_to_kv_pool = pool
    backend.req_to_token_pool = NS(req_to_token=table)
    backend.kv_index_translator = KVIndexTranslator(
        req_to_token=table,
        token_to_kv_pool_allocator=object(),
        token_to_kv_pool=pool,
        page_size=page_size,
        device=device,
    )
    updater = object.__new__(FlashInferIndicesUpdaterPrefill)
    updater.attn_backend = backend
    updater.num_qo_heads = updater.num_kv_heads = 2
    updater.head_dim = 64
    updater.q_data_type = updater.data_type = torch.float16
    updater.kv_indptr, updater.qo_indptr = backend.kv_indptr, backend.qo_indptr
    updater.kv_last_page_len = backend.kv_last_page_len
    updater.prefill_wrapper_ragged = backend.prefill_wrapper_ragged
    updater.update = updater.update_single_wrapper
    backend.indices_updater_prefill = updater
    backend.init_cuda_graph_state(4, 16)
    layer = NS(
        tp_q_head_num=2,
        tp_k_head_num=2,
        tp_v_head_num=2,
        head_dim=64,
        scaling=64**-0.5,
        logit_cap=0.0,
        layer_id=0,
        is_cross_attention=False,
        attn_type=AttentionType.ENCODER_ONLY,
        sliding_window_size=-1,
        k_scale_float=None,
        v_scale_float=None,
    )
    return backend, layer, cache, table


def dense_reference(q, k, v, cache, table, prefix, pads):
    outputs = []
    for i, (n, pad) in enumerate(zip(prefix, pads)):
        rows = table[i + 1, min(pad, n) : n].long()
        keys = torch.cat((cache[0][rows], k[i * 4 : (i + 1) * 4]))
        values = torch.cat((cache[1][rows], v[i * 4 : (i + 1) * 4]))
        valid_prefix = max(n - pad, 0)
        local_pad = min(max(pad - n, 0), 4)
        mask = torch.ones(4, keys.shape[0], dtype=torch.bool, device=q.device)
        mask[:, valid_prefix : valid_prefix + local_pad] = False
        diag = torch.arange(local_pad, device=q.device)
        mask[diag, valid_prefix + diag] = True
        result = F.scaled_dot_product_attention(
            q[i * 4 : (i + 1) * 4].transpose(0, 1).float(),
            keys.transpose(0, 1).float(),
            values.transpose(0, 1).float(),
            attn_mask=mask,
        )
        outputs.append(result.transpose(0, 1).reshape(4, -1))
    return torch.cat(outputs).half()


@pytest.mark.parametrize("page_size", [1, 4])
def test_tiny_cfg_graph_replay_matches_eager(page_size):
    torch.manual_seed(0)
    backend, layer, cache, table = tiny_attention(page_size)
    runner = tiny_runner(backend, device="cuda")
    batch = tiny_batch(device="cuda")
    reg = runner.buffer_registry
    reg.fill_from(batch, raw_bs=3, padded_bs=4, raw_num_tokens=12, padded_num_tokens=16)
    static = reg.extract_buffer(
        padded_bs=4, padded_num_tokens=16, forward_batch_template=batch
    )
    static.seq_lens_sum = batch.seq_lens_sum + 4
    backend.init_forward_metadata_out_graph(static, in_capture=True)
    q, k, v = [
        torch.randn(16, 2, 64, device="cuda", dtype=torch.float16) for _ in range(3)
    ]
    # This target inherits a no-op; it must not invoke stock planning in capture.
    assert (
        backend.init_forward_metadata_in_graph.__func__
        is AttentionBackend.init_forward_metadata_in_graph
    )
    planned = backend.forward_metadata
    backend.init_forward_metadata_in_graph(static)
    assert backend.forward_metadata is planned
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            backend.forward_extend(q, k, v, layer, static, save_kv_cache=False)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.init_forward_metadata_in_graph(static)
        output = backend.forward_extend(q, k, v, layer, static, save_kv_cache=False)
    assert backend.forward_metadata is planned
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output[:12],
        dense_reference(q, k, v, cache, table, (8, 8, 8), (0, 6, 3)),
        atol=3e-3,
        rtol=3e-3,
    )
    pointers = {name: reg.get_slot(name).buffer.data_ptr() for name in reg.slot_names()}
    cases = [
        ((8, 8, 8), (0, 6, 3)),
        ((12, 8, 16), (2, 5, 0)),
        ((4, 12), (0, 7)),
        ((8,), (0,)),
        ((8, 8, 8), (8, 0, 5)),
    ]
    for prefix, pads in cases:
        batch = tiny_batch(prefix, pads, device="cuda")
        count = batch.input_ids.numel()
        for tensor in (q, k, v):
            tensor.copy_(torch.randn_like(tensor))
        assert runner.can_run_graph(batch)
        backend.init_forward_metadata(batch)
        eager = backend.forward_extend(
            q[:count], k[:count], v[:count], layer, batch, save_kv_cache=False
        ).clone()
        reference = dense_reference(q, k, v, cache, table, prefix, pads)
        torch.testing.assert_close(eager, reference, atol=3e-3, rtol=3e-3)
        runner.load_batch(
            batch
        )  # Actual registry -> overridden replay view -> CFG plan.
        replay_plan = backend.forward_metadata
        backend.init_forward_metadata_in_graph(static)
        assert backend.forward_metadata is replay_plan
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(output[:count]).all()
        torch.testing.assert_close(output[:count], eager, atol=3e-3, rtol=3e-3)
        assert runner.bs == 4 and runner.raw_bs == len(prefix)
        assert torch.equal(runner.buffers.positions[:count], batch.positions)
    assert pointers == {
        name: reg.get_slot(name).buffer.data_ptr() for name in reg.slot_names()
    }
    # Query-local pad is an explicit eager-only shape, including conditional pad.
    local = tiny_batch((4, 4, 4), (6, 0, 5), device="cuda")
    assert not runner.can_run_graph(local)
    with pytest.raises(ValueError, match="query-local padding"):
        backend.init_forward_metadata_out_graph(local)
    backend.init_forward_metadata(local)
    assert backend._cfg_local_left_pad_active
    eager = backend.forward_extend(
        q[:12], k[:12], v[:12], layer, local, save_kv_cache=False
    )
    reference = dense_reference(q, k, v, cache, table, (4, 4, 4), (6, 0, 5))
    torch.testing.assert_close(eager, reference, atol=3e-3, rtol=3e-3)
