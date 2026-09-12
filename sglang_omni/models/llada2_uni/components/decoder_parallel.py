# SPDX-License-Identifier: Apache-2.0
"""Decoder parallel contract over model-owned SGLang diffusion groups.

The stage launcher owns processes, shared rendezvous and process timeouts.
decoder_runtime owns group/config initialization and teardown; teardown runs
only after compute exits.
This module only validates and uses that runtime; it does not own teardown.
"""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DecoderParallelConfig:
    backend: str = "diffusers"
    stage_role: str = "single"
    sp_rank: int = 0
    sp_size: int = 1
    ulysses_degree: int = 1
    ring_degree: int = 1
    attention_backend: str = "torch_sdpa"

    def __post_init__(self):
        if self.backend not in {"diffusers", "sglang"}:
            raise ValueError(f"Unsupported image decoder backend: {self.backend!r}")
        if any(
            type(n) is not int or n < 1
            for n in (self.sp_size, self.ulysses_degree, self.ring_degree)
        ):
            raise ValueError("Decoder parallel degrees must be positive integers")
        if type(self.sp_rank) is not int or not 0 <= self.sp_rank < self.sp_size:
            raise ValueError("Decoder sp_rank must be in [0, sp_size)")
        if self.sp_size != self.ulysses_degree * self.ring_degree:
            raise ValueError("Decoder sp_size must equal ulysses_degree * ring_degree")
        expected_role = (
            "single"
            if self.sp_size == 1
            else ("leader" if self.sp_rank == 0 else "follower")
        )
        if self.stage_role != expected_role:
            raise ValueError(
                f"Decoder rank {self.sp_rank}/{self.sp_size} requires stage_role={expected_role!r}"
            )
        if not isinstance(self.attention_backend, str) or self.attention_backend in {
            "",
            "auto",
        }:
            raise ValueError("Decoder attention_backend must be explicit")
        if self.backend == "diffusers" and (
            self.sp_size != 1 or self.attention_backend != "torch_sdpa"
        ):
            raise ValueError(
                "diffusers decoder supports only SP1 and its default torch_sdpa attention"
            )

    @property
    def is_leader(self):
        return self.sp_rank == 0


class SGLangDecoderRuntime:
    """Validate and use a dedicated TP1/SP stage runtime; never fall back.

    Requires initialized SGLang diffusion groups, server args with
    kv_gather_degree=1 (explicit Ulysses/ring), and matching compute dtype/device.
    All ranks must enter decode in the same order. Recoverable preparation
    failures are synchronized; failures inside backend collectives still rely
    on the launcher's timeout and whole-group termination policy.
    """

    def __init__(self, config: DecoderParallelConfig, device, dtype):
        if config.backend != "sglang":
            raise ValueError("SGLang decoder runtime requires backend='sglang'")
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.group = None
        self.validate()

    def validate(self):
        from sglang.multimodal_gen.runtime.distributed import parallel_state as ps
        from sglang.multimodal_gen.runtime.server_args import get_global_server_args
        from sglang.multimodal_gen.utils import get_compute_dtype

        if not dist.is_initialized() or not ps.model_parallel_is_initialized():
            raise RuntimeError(
                "Native decoder requires caller-initialized SGLang diffusion groups"
            )
        c = self.config
        actual = (
            ps.get_world_size(),
            ps.get_tp_world_size(),
            ps.get_sp_world_size(),
            ps.get_sp_parallel_rank(),
            ps.get_ulysses_parallel_world_size(),
            ps.get_ring_parallel_world_size(),
        )
        expected = (c.sp_size, 1, c.sp_size, c.sp_rank, c.ulysses_degree, c.ring_degree)
        if actual != expected:
            raise RuntimeError(
                f"Decoder runtime topology mismatch: {actual}, expected {expected}"
            )
        args = get_global_server_args()
        if args.kv_gather_degree != 1 or args.sp_split_auto:
            raise RuntimeError(
                "Decoder requires explicit Ulysses/ring settings and kv_gather_degree=1"
            )
        if get_compute_dtype() != self.dtype:
            raise RuntimeError(
                "Decoder runtime compute dtype does not match model dtype"
            )
        if self.device.type == "cuda":
            current = torch.cuda.current_device()
            if self.device.index is None:
                self.device = torch.device("cuda", current)
            elif self.device.index != current:
                raise RuntimeError(
                    "Decoder CUDA device does not match current runtime device"
                )
        group = ps.get_sp_group().device_group
        if self.group is not None and self.group is not group:
            raise RuntimeError("Decoder runtime group changed after construction")
        self.group = group

    def _all_gather_object(self, value):
        if self.config.sp_size == 1:
            return [value]
        values = [None] * self.config.sp_size
        dist.all_gather_object(values, value, group=self.group)
        return values

    @contextmanager
    def preparation(self, phase):
        error = None
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - notify peers before leaving preparation
            error = exc
        failures = self._all_gather_object(
            None if error is None else f"{type(error).__name__}: {error}"
        )
        if any(failure is not None for failure in failures):
            raise RuntimeError(
                f"Decoder {phase} failed across ranks: {failures}"
            ) from error

    def request_seed(self, metadata, seed):
        self.validate()
        requests = self._all_gather_object((metadata, seed))
        if any(request != requests[0] for request in requests):
            raise ValueError("Decoder ranks received inconsistent request settings")
        if self.config.sp_size == 1:
            return seed
        shared = [
            seed
            if seed is not None
            else (secrets.randbits(63) if self.config.is_leader else None)
        ]
        dist.broadcast_object_list(
            shared, src=dist.get_global_rank(self.group, 0), group=self.group
        )
        return shared[0]

    def broadcast_features(self, features):
        if self.config.sp_size > 1:
            dist.broadcast(
                features, src=dist.get_global_rank(self.group, 0), group=self.group
            )
        return features
