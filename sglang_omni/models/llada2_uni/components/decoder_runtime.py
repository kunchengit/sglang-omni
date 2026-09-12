# SPDX-License-Identifier: Apache-2.0
"""Native decoder startup/close for a dedicated process, without a server."""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist

from .decoder_parallel import DecoderParallelConfig, SGLangDecoderRuntime


class DecoderRuntimeHandle:
    """Own startup state in an otherwise uninitialized decoder worker.

    Wrap decoder construction and every scheduler compute with
    ``compute_context()``. Close after scheduler.start() exits, before generic
    distributed teardown. Close may run on another thread; it never touches
    precision TLS. No groups may be borrowed/replaced in this dedicated process.
    There is no implicit destructor running collectives.
    """

    def __init__(self, device, dtype, ps, args_module, context, precision):
        self.device, self.dtype = device, dtype
        self._ps, self._args, self._context, self._precision = (
            ps,
            args_module,
            context,
            precision,
        )
        self._world_started = self._model_started = self._published = False
        self._closed = False
        self.runtime = None

    @contextmanager
    def compute_context(self):
        """Set device/precision on the actual computing thread, then restore it."""
        if self._closed:
            raise RuntimeError("Decoder runtime is closed")
        device_context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with device_context:
            state = self._precision._mixed_precision_state
            missing = object()
            previous = getattr(state, "state", missing)
            self._precision.set_mixed_precision_policy(
                param_dtype=self.dtype, reduce_dtype=torch.float32
            )
            try:
                yield
            finally:
                if previous is missing:
                    del state.state
                else:
                    state.state = previous

    def close(self):
        if self._closed:
            return
        self._closed = True
        # Attempt all owned teardown. Errors remain visible through exception
        # chaining; partial initialization uses these same upstream APIs.
        try:
            if self._model_started:
                self._ps.destroy_model_parallel()
        finally:
            try:
                if self._world_started:
                    self._ps.destroy_distributed_environment()
            finally:
                if self._published:
                    try:
                        self._args.set_global_server_args(None)
                    finally:
                        self._context.reset_context()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Decoder runtime is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if exc is not None:
                raise exc from cleanup
            raise


def initialize_decoder_runtime(
    model_path: str,
    *,
    sp_rank: int,
    sp_size: int,
    stage_role: str | None = None,
    gpu_id: int | None = 0,
    nccl_port: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    attention_backend: str = "torch_sdpa",
    dist_timeout: int = 180,
) -> DecoderRuntimeHandle:
    """Consume core factory kwargs and initialize real diffusion/SRT state.

    gpu_id is process-visible (normally 0 with one GPU per worker); None is
    explicit CPU/Gloo. SP workers use shared nccl_port and MASTER_ADDR. Native
    SP1 without a port allocates a fresh loopback rendezvous, never an inherited
    MASTER_PORT. No weights, scheduler, server or subprocesses are created.
    """
    config = DecoderParallelConfig(
        backend="sglang",
        stage_role=stage_role
        if stage_role is not None
        else ("single" if sp_size == 1 else ("leader" if sp_rank == 0 else "follower")),
        sp_rank=sp_rank,
        sp_size=sp_size,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        attention_backend=attention_backend,
    )
    if gpu_id is not None and (type(gpu_id) is not int or gpu_id < 0):
        raise ValueError("Decoder gpu_id must be a nonnegative CUDA ordinal or None")
    if type(dist_timeout) is not int or dist_timeout < 1:
        raise ValueError("Decoder dist_timeout must be positive seconds")
    precisions = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
    if dtype not in precisions:
        raise ValueError("Unsupported decoder runtime dtype")
    if nccl_port is not None and (
        type(nccl_port) is not int or not 1 <= nccl_port <= 65535
    ):
        raise ValueError("Decoder nccl_port must be an integer in [1, 65535]")
    if sp_size > 1 and nccl_port is None:
        raise ValueError("Native SP workers require the core-provided shared nccl_port")
    if dist.is_initialized():
        raise RuntimeError(
            "Decoder initialization requires a dedicated process with no existing groups"
        )
    if nccl_port is None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            nccl_port = sock.getsockname()[1]
        host = "127.0.0.1"
    else:
        host = os.environ.get("MASTER_ADDR", "127.0.0.1")
    if not host or any(char in host for char in "/?#@"):
        raise ValueError("Decoder MASTER_ADDR must be a hostname or IP address")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    address = f"{host}:{nccl_port}"

    from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
        ZImagePipelineConfig,
    )
    from sglang.multimodal_gen.runtime.distributed import parallel_state as ps
    from sglang.multimodal_gen.runtime.server_args import server_args as args_module
    from sglang.multimodal_gen import utils as precision
    from sglang.srt import runtime_context as context
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    if (
        ps.world_group_is_initialized()
        or ps.model_parallel_is_initialized()
        or args_module._global_server_args is not None
        or context.get_context()._server_args is not None
    ):
        raise RuntimeError(
            "Decoder initialization refuses existing SGLang groups/config"
        )
    device = torch.device("cpu" if gpu_id is None else f"cuda:{gpu_id}")
    owner = DecoderRuntimeHandle(device, dtype, ps, args_module, context, precision)
    try:
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        args = args_module.ServerArgs(
            model_path=model_path,
            backend=args_module.Backend.SGLANG,
            pipeline_config=ZImagePipelineConfig(dit_precision=precisions[dtype]),
            performance_mode="manual",
            num_gpus=sp_size,
            tp_size=1,
            sp_degree=sp_size,
            dp_size=1,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            kv_gather_degree=1,
            sp_split_auto=False,
            enable_cfg_parallel=False,
            cfg_parallel_degree=1,
            attention_backend=attention_backend,
            use_fsdp_inference=False,
            dit_cpu_offload=False,
            dit_layerwise_offload=False,
            vae_cpu_offload=False,
            text_encoder_cpu_offload=False,
            image_encoder_cpu_offload=False,
            enable_torch_compile=False,
            enable_breakable_cuda_graph=False,
            nccl_port=nccl_port,
            dist_init_addr=address,
            dist_timeout=dist_timeout,
        )
        owner._published = True
        args_module.set_global_server_args(args)
        owner._world_started = True
        ps.init_distributed_environment(
            world_size=sp_size,
            rank=sp_rank,
            distributed_init_method=f"tcp://{address}",
            local_rank=gpu_id or 0,
            backend="gloo" if gpu_id is None else "nccl",
            device_id=None if gpu_id is None else device,
            timeout=dist_timeout,
        )
        owner._model_started = True
        ps.initialize_model_parallel(
            data_parallel_size=1,
            classifier_free_guidance_degree=1,
            sequence_parallel_degree=sp_size,
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            vae_parallel_size=0,
            backend="gloo" if gpu_id is None else "nccl",
        )
        # Publish complete config bags after groups, as the upstream worker does.
        context.publish(
            SrtServerArgs(model_path="dummy", tp_size=1), role="diffusion_gpu_worker"
        )
        with owner.compute_context():
            owner.runtime = SGLangDecoderRuntime(config, device, dtype)
    except BaseException as error:
        try:
            owner.close()
        except BaseException as cleanup:
            raise error from cleanup
        raise
    return owner
