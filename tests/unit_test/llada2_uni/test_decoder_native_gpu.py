# SPDX-License-Identifier: Apache-2.0
"""Opt-in tiny real CUDA/native-loader integration; no model downloads or mocks.

Run with LLADA_DECODER_GPU_TEST=1 on the target SGLang installation. Only
physical GPUs 0/1 are exposed to spawned workers. Ordinary unit runs skip it.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp


def _rendezvous():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_workers(sp, directory, cfg, dtype):
    workers = mp.spawn(
        _native_worker,
        args=(sp, _rendezvous() if sp > 1 else None, directory, cfg, dtype),
        nprocs=sp,
        join=False,
    )
    try:
        deadline = time.monotonic() + 240
        while not workers.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Tiny native SP{sp} integration exceeded 240 seconds"
                )
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.terminate()
        for process in workers.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()


def _native_worker(rank, sp_size, address, directory, cfg, dtype):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(sp_size)
    if address is not None:
        os.environ["MASTER_PORT"] = str(address)
    from sglang_omni.models.llada2_uni.components.decoder_model import (
        ZImageTransformer2DModelWrapper,
    )
    from sglang_omni.models.llada2_uni.components.decoder_runtime import (
        initialize_decoder_runtime,
    )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Tiny native integration requires one visible CUDA GPU per worker"
        )
    path = Path(directory)
    with initialize_decoder_runtime(
        directory,
        sp_rank=rank,
        sp_size=sp_size,
        gpu_id=0,
        stage_role="single"
        if sp_size == 1
        else ("leader" if rank == 0 else "follower"),
        nccl_port=address,
        dtype=dtype,
        ulysses_degree=sp_size,
        ring_degree=1,
        attention_backend="torch_sdpa",
        dist_timeout=120,
    ) as handle:
        from sglang.srt.runtime_context import get_exec, get_model, get_parallel

        assert get_exec() is not None and get_model() is not None
        assert get_parallel().tp_size == 1
        main_thread = threading.get_ident()

        def compute_and_close():
            from sglang.multimodal_gen.utils import get_compute_dtype

            assert threading.get_ident() != main_thread
            previous = get_compute_dtype()
            try:
                with handle.compute_context(), torch.inference_mode():
                    assert get_compute_dtype() == dtype
                    inputs = torch.load(
                        path / "inputs.pt",
                        weights_only=True,
                        map_location=handle.device,
                    )
                    model = ZImageTransformer2DModelWrapper(
                        directory,
                        cfg,
                        handle.device,
                        dtype,
                        backend="sglang",
                        parallel_runtime=handle.runtime,
                    )
                    assert all(
                        not p.is_meta and p.dtype == dtype for p in model.parameters()
                    )
                    output = torch.stack(
                        model(
                            list(inputs["x"].to(dtype).unbind(0)),
                            inputs["t"],
                            list(inputs["cap"].to(dtype).unbind(0)),
                            return_dict=False,
                        )[0]
                    )
                    assert (
                        output.shape == inputs["x"].shape
                        and torch.isfinite(output).all()
                    )
                    torch.save(output.cpu(), path / f"native-sp{sp_size}-rank{rank}.pt")
                    del model, inputs, output
            finally:
                try:
                    assert get_compute_dtype() == previous
                finally:
                    handle.close()

        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(compute_and_close).result()
    assert not torch.distributed.is_initialized()
    handle.close()  # Idempotent; must not tear down anything else.


@pytest.mark.skipif(
    os.environ.get("LLADA_DECODER_GPU_TEST") != "1",
    reason="opt-in real native GPU test",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_tiny_native_strict_load_sp1_sp2(tmp_path, monkeypatch, dtype):
    from diffusers.models.transformers.transformer_z_image import (
        ZImageTransformer2DModel,
    )
    from safetensors.torch import save_file

    from sglang_omni.models.llada2_uni.components.decoder_model import _decoder_config

    # Set before spawn: child CUDA ordinals 0/1 cannot reach any other GPU.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    cfg = _decoder_config(
        {
            # 8/3 * dim must satisfy the native fused activation vector width.
            "dim": 768,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "n_heads": 6,
            "n_kv_heads": 6,
            "cap_feat_dim": 16,
            "axes_dims": (32, 48, 48),
            "axes_lens": (128, 64, 64),
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        reference = ZImageTransformer2DModel(**cfg).eval()
        torch.nn.init.normal_(reference.x_pad_token, std=0.01)
        torch.nn.init.normal_(reference.cap_pad_token, std=0.01)
        inputs = {
            "x": torch.randn(2, 16, 1, 16, 16),
            "cap": torch.randn(2, 32, 16),
            "t": torch.tensor([0.125, 0.875]),
        }
    save_file(
        {
            name.replace("cap_embedder.", "semantic_embedder."): value.contiguous()
            for name, value in reference.state_dict().items()
        },
        str(tmp_path / "model.safetensors"),
    )
    torch.save(inputs, tmp_path / "inputs.pt")
    with torch.inference_mode():
        expected = torch.stack(
            reference(
                x=list(inputs["x"].unbind(0)),
                t=inputs["t"],
                cap_feats=list(inputs["cap"].unbind(0)),
                return_dict=False,
            )[0]
        )
    del reference
    for sp in (1, 2):
        _run_workers(sp, str(tmp_path), cfg, dtype)
    single = torch.load(tmp_path / "native-sp1-rank0.pt", weights_only=True)
    print(
        f"native/diffusers max_abs_error={(single.float() - expected).abs().max():.6g}"
    )
    torch.testing.assert_close(single.float(), expected, rtol=2e-2, atol=2e-2)
    for rank in range(2):
        parallel = torch.load(tmp_path / f"native-sp2-rank{rank}.pt", weights_only=True)
        print(
            f"{dtype} SP2 rank{rank} max_abs_error={(parallel.float() - single.float()).abs().max():.6g}"
        )
        torch.testing.assert_close(parallel, single, rtol=2e-2, atol=2e-2)
