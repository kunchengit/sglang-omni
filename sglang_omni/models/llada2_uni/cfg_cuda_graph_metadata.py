# SPDX-License-Identifier: Apache-2.0
"""Host-known CFG geometry and declared ForwardBatch graph transport fields."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from numbers import Integral

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class DllmCFGForwardBatch(ForwardBatch):
    """Omni-owned fields survive both eager and graph registry replacements."""

    reqs: list = field(default_factory=list)
    dllm_left_pad_lens: torch.Tensor | None = None
    dllm_left_pad_lens_cpu: list[int] | torch.Tensor | None = None
    actual_forward_mode: object = None
    num_padding: int = 0


def cpu_lengths(value, name: str, size: int) -> tuple[int, ...]:
    """Accept host metadata only; never turn a missing mirror into a GPU sync."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise RuntimeError(f"DLLM CFG requires host-known {name}")
        value = value.tolist()
    if value is None or len(value) != size:
        raise ValueError(f"DLLM CFG {name} must contain {size} entries")
    if any(not isinstance(x, Integral) for x in value):
        raise ValueError(f"DLLM CFG {name} must contain integers")
    return tuple(int(x) for x in value)


@dataclass(frozen=True)
class CFGAttentionGeometry:
    prefix: tuple[int, ...]
    query: tuple[int, ...]
    pad: tuple[int, ...]

    @property
    def cached_pad(self) -> tuple[int, ...]:
        return tuple(min(p, n) for p, n in zip(self.pad, self.prefix))

    @property
    def local_pad(self) -> tuple[int, ...]:
        return tuple(
            min(max(p - n, 0), q) for p, n, q in zip(self.pad, self.prefix, self.query)
        )

    @property
    def paged_lens(self) -> tuple[int, ...]:
        return tuple(max(n - p, 0) for n, p in zip(self.prefix, self.pad))


def cfg_attention_geometry(batch) -> CFGAttentionGeometry:
    size = getattr(batch, "batch_size", len(batch.seq_lens))
    prefix = getattr(batch, "extend_prefix_lens_cpu", None)
    if prefix is None:
        prefix = getattr(batch, "extend_prefix_lens", None)
    prefix = cpu_lengths(prefix, "extend_prefix_lens_cpu", size)
    query = getattr(batch, "extend_seq_lens_cpu", None)
    if query is None:
        query = getattr(batch, "extend_seq_lens", None)
    if query is None:
        seq = getattr(batch, "seq_lens_cpu", None)
        if seq is None:
            seq = getattr(batch, "seq_lens", None)
        seq = cpu_lengths(seq, "seq_lens_cpu", size)
        query = tuple(s - p for s, p in zip(seq, prefix))
    query = cpu_lengths(query, "extend_seq_lens_cpu", size)
    pad = getattr(batch, "dllm_left_pad_lens_cpu", None)
    if pad is None:
        reqs = getattr(batch, "reqs", None)
        if reqs is not None and len(reqs) == size:
            pad = [getattr(req, "_dllm_left_pad_len", 0) for req in reqs]
        else:
            pad = getattr(batch, "dllm_left_pad_lens", None)
            if pad is None:
                pad = [0] * size
    pad = cpu_lengths(pad, "dllm_left_pad_lens_cpu", size)
    if any(p < 0 or n < 0 or q <= 0 for p, n, q in zip(pad, prefix, query)):
        raise ValueError(
            "DLLM CFG requires nonnegative pad/prefix and a nonempty query"
        )
    return CFGAttentionGeometry(prefix, query, pad)


def as_cfg_forward_batch(batch) -> DllmCFGForwardBatch:
    """Promote a declared batch, without mutating the scheduler's original batch."""
    geometry = cfg_attention_geometry(batch)
    values = {f.name: getattr(batch, f.name) for f in fields(batch)}
    values.update(
        extend_prefix_lens_cpu=list(geometry.prefix),
        extend_seq_lens_cpu=list(geometry.query),
        dllm_left_pad_lens_cpu=list(geometry.pad),
    )
    return DllmCFGForwardBatch(**values)
