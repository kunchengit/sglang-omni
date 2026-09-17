# SPDX-License-Identifier: Apache-2.0
"""Tiny real-runtime graph transport fixtures; no model weights or import doubles."""

from types import SimpleNamespace as NS

import torch
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode

from sglang_omni.models.llada2_uni.cfg_cuda_graph import (
    CFG_GRAPH_FIELDS,
    LLaDA2CFGDecodeCudaGraphRunner,
    register_cfg_graph_slots,
)
from sglang_omni.models.llada2_uni.cfg_cuda_graph_metadata import DllmCFGForwardBatch


def tiny_batch(prefix=(8, 8, 8), pads=(0, 6, 3), *, device="cpu"):
    size = len(prefix)
    return DllmCFGForwardBatch(
        forward_mode=ForwardMode.DLLM_EXTEND,
        capture_hidden_mode=CaptureHiddenMode.NULL,
        batch_size=size,
        input_ids=torch.arange(size * 4, device=device),
        req_pool_indices=torch.arange(1, size + 1, device=device),
        seq_lens=torch.tensor(
            [p + 4 for p in prefix], dtype=torch.int32, device=device
        ),
        seq_lens_cpu=torch.tensor([p + 4 for p in prefix], dtype=torch.int32),
        out_cache_loc=torch.zeros(size * 4, dtype=torch.int64, device=device),
        seq_lens_sum=sum(prefix) + size * 4,
        positions=torch.tensor(
            [max(p + q - pad, 0) for p, pad in zip(prefix, pads) for q in range(4)],
            dtype=torch.int64,
            device=device,
        ),
        extend_prefix_lens=torch.tensor(prefix, dtype=torch.int32, device=device),
        extend_prefix_lens_cpu=list(prefix),
        extend_seq_lens=torch.full((size,), 4, dtype=torch.int32, device=device),
        extend_seq_lens_cpu=[4] * size,
        dllm_left_pad_lens=torch.tensor(pads, dtype=torch.int32, device=device),
        dllm_left_pad_lens_cpu=list(pads),
    )


def tiny_runner(attention_backend, *, device="cpu"):
    """Use real load_batch/eligibility/hook methods without constructing a model."""
    registry = build_decode_registry(
        device=torch.device(device),
        max_bs=4,
        max_num_token=16,
        seq_len_fill_value=4,
        cache_loc_dtype=torch.int64,
        register_global_num_tokens=False,
        share_pool=False,
    )
    original = set(registry.slot_names())
    positions = registry.get_slot("positions").buffer
    assert not original.intersection(CFG_GRAPH_FIELDS)
    register_cfg_graph_slots(registry, 4)
    assert set(registry.slot_names()) == original.union(CFG_GRAPH_FIELDS)
    assert registry.get_slot("positions").buffer is positions
    runner = object.__new__(LLaDA2CFGDecodeCudaGraphRunner)
    runner.buffer_registry = registry
    runner.buffers = NS(
        **{name: registry.get_slot(name).buffer for name in registry.slot_names()}
    )
    runner.buffers.mamba_track_indices = None
    runner.attn_backend = attention_backend
    runner.ragged_verify_mode = runner.require_mlp_tp_gather = False
    runner.require_mlp_sync = runner.enable_two_batch_overlap = False
    runner.enable_pdmux = runner.is_encoder_decoder = runner.disable_padding = False
    runner.captured_req_width = runner.seq_len_fill_value = runner.max_bs = 4
    runner.capture_bs = [4]
    runner.capture_forward_mode = ForwardMode.DLLM_EXTEND
    runner.capture_hidden_mode = CaptureHiddenMode.NULL
    runner._metadata_glue = None
    runner.deepep_adapter = NS(replay=lambda: None)
    runner.model_runner = NS(
        spec_algorithm=NS(is_dflash_family=lambda: False, is_ngram=lambda: False),
        hisparse_coordinator=None,
        lora_manager=None,
    )
    return runner
