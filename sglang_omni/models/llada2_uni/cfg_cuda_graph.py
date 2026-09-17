# SPDX-License-Identifier: Apache-2.0
"""Model-specific DLLM CFG bridge for the SGLang decode graph runner.

Integration uses ModelRunner._decode_cuda_graph_runner_cls and backend-owned
registry views. No runner or backend is monkeypatched. Only fixed-width blocks
without query-local padding are graph eligible.
"""

from __future__ import annotations

from dataclasses import fields

import torch
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    GraphSlot,
    PaddingPolicy,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)

from sglang_omni.models.llada2_uni.cfg_cuda_graph_config import (
    validate_cfg_cuda_graph_config,
)
from sglang_omni.models.llada2_uni.cfg_cuda_graph_metadata import (
    DllmCFGForwardBatch,
    as_cfg_forward_batch,
    cfg_attention_geometry,
    cpu_lengths,
)

CFG_GRAPH_FIELDS = (
    "extend_prefix_lens",
    "extend_prefix_lens_cpu",
    "extend_seq_lens",
    "extend_seq_lens_cpu",
    "dllm_left_pad_lens",
    "dllm_left_pad_lens_cpu",
)


def register_cfg_graph_slots(registry, block_size: int) -> None:
    """Mirror CPU-authoritative layout into typed slots, including dummy tails."""
    for name in CFG_GRAPH_FIELDS:
        cpu_name = name if name.endswith("_cpu") else name + "_cpu"
        pad_value = block_size if name.startswith("extend_seq_lens") else 0

        def source(batch, ctx, cpu_name=cpu_name):
            return torch.tensor(
                cpu_lengths(getattr(batch, cpu_name), cpu_name, ctx.raw_bs),
                dtype=torch.int32,
                device="cpu",
            )

        registry.register_slot(
            GraphSlot(
                name=name,
                shape_fn=lambda max_bs, _: (max_bs,),
                dtype=torch.int32,
                axis="bs",
                device="cpu" if name.endswith("_cpu") else None,
                padding_policy=PaddingPolicy.FILL_SENTINEL,
                pad_value=pad_value,
                source_fn=source,
            )
        )


def cfg_graph_views(registry, bs: int) -> dict:
    return {name: registry.get_slot(name).buffer[:bs] for name in CFG_GRAPH_FIELDS}


def attach_cfg_graph_views(forward_batch, registry):
    """Attach model-owned registry fields to SGLang's replay metadata view."""
    for name, value in cfg_graph_views(registry, forward_batch.batch_size).items():
        setattr(forward_batch, name, value)
    return forward_batch


class LLaDA2CFGDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Scoped capture/replay transport with ordinary runner eligibility gates."""

    def __init__(self, model_runner, **kwargs):
        validate_cfg_cuda_graph_config(model_runner.server_args)
        super().__init__(model_runner, **kwargs)

    def capture(self):
        # Upstream builds the registry before calling this overridable method.
        if (
            self.enable_two_batch_overlap
            or self.enable_pdmux
            or self.require_gathered_buffer
        ):
            raise ValueError(
                "DLLM CFG CUDA graphs do not support split or gathered batch metadata"
            )
        if not self.capture_forward_mode.is_dllm_extend():
            raise ValueError("DLLM CFG CUDA graphs require DLLM_EXTEND capture")
        if not self.buffer_registry.has_slot("dllm_left_pad_lens"):
            register_cfg_graph_slots(self.buffer_registry, self.captured_req_width)
        self.attn_backend._cfg_graph_registry = self.buffer_registry
        return super().capture()

    def capture_prepare(self, size, stream_idx=None, num_tokens=None):
        batch, backend, proxy = super().capture_prepare(size, stream_idx, num_tokens)
        for name in CFG_GRAPH_FIELDS:
            value = (
                self.captured_req_width
                if name.startswith("extend_seq_lens")
                else self.seq_len_fill_value - self.captured_req_width
                if name.startswith("extend_prefix_lens")
                else 0
            )
            self.buffer_registry.get_slot(name).buffer[:size].fill_(value)
        values = {f.name: getattr(batch, f.name) for f in fields(batch)}
        values.update(cfg_graph_views(self.buffer_registry, size))
        cfg_batch = DllmCFGForwardBatch(**values)
        # Upstream assigns this non-dataclass capture attribute explicitly.
        cfg_batch.hisparse_coordinator = batch.hisparse_coordinator
        return cfg_batch, backend, proxy

    def can_run_graph(self, forward_batch):
        if not forward_batch.forward_mode.is_dllm_extend():
            return False
        geometry = cfg_attention_geometry(forward_batch)
        if any(geometry.local_pad) or any(
            q != self.captured_req_width for q in geometry.query
        ):
            # First edit block and short prefill blocks need dynamic eager masks.
            return False
        return super().can_run_graph(forward_batch)

    def load_batch(self, forward_batch, pp_proxy_tensors=None):
        # Promotion is local to this call. The algorithm retains its original reqs.
        self.attn_backend._cfg_graph_registry = self.buffer_registry
        return super().load_batch(as_cfg_forward_batch(forward_batch), pp_proxy_tensors)
