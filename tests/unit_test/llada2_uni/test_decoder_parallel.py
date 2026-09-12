# SPDX-License-Identifier: Apache-2.0
"""CPU contracts, NOT distributed/GPU validation.

SGLANG_SOURCE_DIR selects the target checkout. Source-backed tests execute
its spatial-policy methods unchanged; the native model, loader and process
groups are test doubles. No server or process group is started by this suite.
"""

from __future__ import annotations

import ast
import enum
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from sglang_omni.models.llada2_uni.components import image_decoder as image_module
from sglang_omni.models.llada2_uni.components.decoder_model import (
    ZImageTransformer2DModelWrapper,
)
from sglang_omni.models.llada2_uni.components.decoder_parallel import (
    DecoderParallelConfig,
    SGLangDecoderRuntime,
)


def _module(monkeypatch, name, **attrs):
    parts = name.split(".")
    for index in range(1, len(parts) + 1):
        path = ".".join(parts[:index])
        if path not in sys.modules:
            module = ModuleType(path)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, path, module)
            if index > 1:
                monkeypatch.setattr(
                    sys.modules[".".join(parts[: index - 1])],
                    parts[index - 1],
                    module,
                    raising=False,
                )
    for key, value in attrs.items():
        monkeypatch.setattr(sys.modules[name], key, value, raising=False)


@pytest.fixture
def native_env(monkeypatch):
    import torch.distributed as dist

    state = SimpleNamespace(
        sp=1,
        rank=0,
        u=1,
        ring=1,
        tp=1,
        initialized=True,
        dtype=torch.float32,
        group=object(),
        args=SimpleNamespace(kv_gather_degree=1, sp_split_auto=False),
        broadcasts=[],
        contexts=[],
        failures=None,
        requests=None,
        full=None,
    )
    ps = "sglang.multimodal_gen.runtime.distributed.parallel_state"
    _module(
        monkeypatch,
        ps,
        model_parallel_is_initialized=lambda: state.initialized,
        get_world_size=lambda: state.sp,
        get_tp_world_size=lambda: state.tp,
        get_sp_world_size=lambda: state.sp,
        get_sp_parallel_rank=lambda: state.rank,
        get_ulysses_parallel_world_size=lambda: state.u,
        get_ring_parallel_world_size=lambda: state.ring,
        get_sp_group=lambda: SimpleNamespace(device_group=state.group),
    )
    _module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.server_args",
        get_global_server_args=lambda: state.args,
    )
    _module(
        monkeypatch,
        "sglang.multimodal_gen.utils",
        get_compute_dtype=lambda: state.dtype,
    )
    monkeypatch.setattr(dist, "is_initialized", lambda: state.initialized)
    monkeypatch.setattr(dist, "get_global_rank", lambda group, rank: 19 + rank)

    def gather_objects(output, value, group):
        assert group is state.group
        output[:] = state.failures or state.requests or [value] * state.sp

    def broadcast_objects(values, src, group):
        assert group is state.group and src == 19
        state.broadcasts.append(values[0])
        if state.rank > 0 and values[0] is None:
            values[0] = 37

    def broadcast_tensor(tensor, src, group):
        assert group is state.group and src == 19
        state.broadcasts.append(tuple(tensor.shape))
        if state.rank > 0:
            tensor.fill_(1)

    @contextmanager
    def forward_context(**kwargs):
        state.contexts.append(kwargs)
        yield

    monkeypatch.setattr(dist, "all_gather_object", gather_objects)
    monkeypatch.setattr(dist, "broadcast_object_list", broadcast_objects)
    monkeypatch.setattr(dist, "broadcast", broadcast_tensor)
    _module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.managers.forward_context",
        set_forward_context=forward_context,
    )
    return state


def _config(state, sp=1, rank=0, u=None, ring=1, attention="torch_sdpa"):
    state.sp, state.rank, state.u, state.ring = sp, rank, u or sp, ring
    return DecoderParallelConfig(
        backend="sglang",
        sp_size=sp,
        sp_rank=rank,
        ulysses_degree=state.u,
        ring_degree=ring,
        attention_backend=attention,
        stage_role="single" if sp == 1 else ("leader" if rank == 0 else "follower"),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sp_size": 0},
        {"sp_size": 2},
        {"sp_rank": -1},
        {"sp_rank": 1},
        {"stage_role": "leader"},
        {"backend": "auto"},
        {"attention_backend": "auto"},
        {"backend": "diffusers", "attention_backend": "fa"},
        {
            "backend": "diffusers",
            "sp_size": 2,
            "ulysses_degree": 2,
            "stage_role": "leader",
        },
        {
            "backend": "sglang",
            "sp_size": 2,
            "ulysses_degree": 2,
            "stage_role": "follower",
        },
    ],
)
def test_invalid_parallel_contract(kwargs):
    with pytest.raises(ValueError):
        DecoderParallelConfig(**kwargs)


@pytest.mark.parametrize("sp,u,ring", [(1, 1, 1), (2, 2, 1), (4, 2, 2)])
def test_runtime_topology_contract(native_env, sp, u, ring):
    for rank in range(sp):
        cfg = _config(native_env, sp, rank, u, ring)
        runtime = SGLangDecoderRuntime(cfg, "cpu", torch.float32)
        assert runtime.group is native_env.group
        assert cfg.is_leader == (rank == 0)


@pytest.mark.parametrize(
    "failure", ["uninitialized", "tp", "rank", "dtype", "auto", "kv_gather", "group"]
)
def test_runtime_mismatch_fails_closed(native_env, failure):
    cfg = _config(native_env, 2)
    runtime = SGLangDecoderRuntime(cfg, "cpu", torch.float32)
    if failure == "uninitialized":
        native_env.initialized = False
    elif failure == "tp":
        native_env.tp = 2
    elif failure == "rank":
        native_env.rank = 1
    elif failure == "dtype":
        native_env.dtype = torch.bfloat16
    elif failure == "auto":
        native_env.args.sp_split_auto = True
    elif failure == "kv_gather":
        native_env.args.kv_gather_degree = 2
    else:
        native_env.group = object()
    with pytest.raises(RuntimeError):
        runtime.validate()


def test_native_constructor_does_not_initialize_or_fall_back(native_env, tmp_path):
    native_env.initialized = False
    with pytest.raises(RuntimeError, match="caller-initialized"):
        image_module.LLaDA2ImageDecoder(
            str(tmp_path), device="cpu", dtype=torch.float32, backend="sglang"
        )
    with pytest.raises(RuntimeError, match="caller-initialized"):
        ZImageTransformer2DModelWrapper(
            tmp_path, {}, "cpu", torch.float32, backend="sglang"
        )


def test_request_metadata_seed_and_group_scoped_broadcast(native_env):
    runtime = SGLangDecoderRuntime(_config(native_env, 2), "cpu", torch.float32)
    assert runtime.request_seed((32, 32, "normal", 50, 2), 11) == 11
    seed = runtime.request_seed((32, 32, "normal", 50, 2), None)
    assert isinstance(seed, int) and 0 <= seed < 2**63
    features = torch.ones(4, 16)
    assert runtime.broadcast_features(features) is features
    native_env.requests = [("first", 11), ("second", 11)]
    with pytest.raises(ValueError, match="inconsistent"):
        runtime.request_seed((), 11)


@pytest.mark.parametrize("remote", [False, True])
def test_preparation_failure_is_shared(native_env, remote):
    runtime = SGLangDecoderRuntime(_config(native_env, 2), "cpu", torch.float32)
    if remote:
        native_env.failures = [None, "ValueError: remote checkpoint"]
    with (
        pytest.raises(RuntimeError, match="weight loading failed across ranks"),
        runtime.preparation("weight loading"),
    ):
        if not remote:
            raise ValueError("local checkpoint")


@pytest.fixture
def source_spatial(native_env):
    root = os.environ.get("SGLANG_SOURCE_DIR")
    if not root:
        pytest.skip("set SGLANG_SOURCE_DIR for source-backed spatial contracts")
    base = Path(root) / "python/sglang/multimodal_gen/configs/pipeline_configs"
    tree = ast.parse((base / "zimage.py").read_text())
    original = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ZImagePipelineConfig"
    )
    methods = {
        "_ceil_to_multiple",
        "_split_evenly",
        "_build_zimage_sp_plan",
        "_get_zimage_sp_plan",
        "shard_latents_for_sp",
        "gather_latents_for_sp",
        "gather_noise_pred_for_sp",
        "_caption_rope_length",
        "get_freqs_cis",
        "prepare_pos_cond_kwargs",
    }
    body = [
        node
        for node in original.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    assert len(body) == len(methods)
    base_tree = ast.parse((base / "base.py").read_text())
    body.append(
        next(
            node
            for node in ast.walk(base_tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "require_text_seq_lens"
        )
    )
    cls = ast.ClassDef(
        name="SourceSpatialConfig", bases=[], keywords=[], body=body, decorator_list=[]
    )
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {
        "torch": torch,
        "math": math,
        "dist": torch.distributed,
        "get_sp_world_size": lambda: native_env.sp,
        "get_sp_parallel_rank": lambda: native_env.rank,
        "get_sp_group": lambda: SimpleNamespace(device_group=native_env.group),
    }
    # Execute only the explicitly selected methods from the caller's checkout.
    exec(compile(module, str(base / "zimage.py"), "exec"), scope)  # noqa: S102
    policy = scope["SourceSpatialConfig"]
    policy.SEQ_LEN_MULTIPLE, policy.PATCH_SIZE, policy.F_PATCH_SIZE = 32, 2, 1

    def init(self, dit_config):
        self.dit_config = dit_config
        self.vae_config = SimpleNamespace(
            arch_config=SimpleNamespace(spatial_compression_ratio=1)
        )
        self.vae_config.post_init = lambda: setattr(
            self.vae_config.arch_config, "spatial_compression_ratio", 8
        )

    policy.__init__ = init
    return policy


@pytest.fixture
def native_loader(native_env, source_spatial, monkeypatch):
    captured = SimpleNamespace(
        constructor=None, load=None, forward=[], selection=None, fail=False
    )

    class Backend(enum.Enum):
        TORCH_SDPA = 1
        FA = 2

    class NativeModel(torch.nn.Module):
        def __init__(self, config, hf_config):
            super().__init__()
            self.param_names_mapping = {"native": "packing"}
            captured.constructor = (config, hf_config)
            self.weight = torch.nn.Parameter(torch.empty(1))

        def rotary_emb(self, ids):
            return ids.float(), ids.float() + 1000

        def forward(self, **kwargs):
            captured.forward.append(kwargs)
            return -kwargs["hidden_states"]

    @contextmanager
    def selection(backend, **kwargs):
        captured.selection = backend, kwargs
        yield

    @contextmanager
    def default_dtype(dtype):
        old = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(old)

    def load(
        *,
        model,
        full_sd_iterator,
        checkpoint_load_device,
        param_dtype,
        strict,
        cpu_offload,
        param_names_mapping,
    ):
        captured.load = (
            dict(full_sd_iterator),
            checkpoint_load_device,
            param_dtype,
            strict,
            cpu_offload,
            param_names_mapping,
        )
        if captured.fail:
            raise RuntimeError("native checkpoint failure")
        model.to_empty(device=checkpoint_load_device)

    def weights(paths):
        captured.paths = paths
        yield "semantic_embedder.0.weight", torch.ones(1)

    _module(
        monkeypatch,
        "sglang.multimodal_gen.configs.models.dits.zimage",
        ZImageArchConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ZImageDitConfig=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    _module(
        monkeypatch,
        "sglang.multimodal_gen.configs.pipeline_configs.zimage",
        ZImagePipelineConfig=source_spatial,
    )
    _module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.platforms",
        AttentionBackendEnum=Backend,
    )
    _module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.layers.attention.selector",
        component_attn_backend_context_manager=selection,
    )
    _module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.loader.fsdp_load",
        load_model_from_full_model_state_dict=load,
    )
    _module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.loader.utils",
        get_param_names_mapping=lambda mapping: mapping,
        set_default_torch_dtype=default_dtype,
    )
    _module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.loader.weight_utils",
        safetensors_weights_iterator=weights,
    )
    _module(
        monkeypatch,
        "sglang.multimodal_gen.runtime.models.dits.zimage",
        ZImageTransformer2DModel=NativeModel,
    )

    def gather(outputs, local, group):
        assert group is native_env.group
        rank = native_env.rank
        try:
            for i, out in enumerate(outputs):
                native_env.rank = i
                policy = source_spatial(dit_config=None)
                shard, _ = policy.shard_latents_for_sp(
                    SimpleNamespace(raw_latent_shape=native_env.full.shape),
                    native_env.full,
                )
                out.zero_()
                out[tuple(slice(0, dim) for dim in shard.shape)].copy_(-shard)
        finally:
            native_env.rank = rank

    monkeypatch.setattr(torch.distributed, "all_gather", gather)
    return captured


def _native_wrapper(env, tmp_path, *, sp=1, rank=0, u=None, ring=1, cfg=None):
    runtime = SGLangDecoderRuntime(
        _config(env, sp, rank, u, ring), "cpu", torch.float32
    )
    return ZImageTransformer2DModelWrapper(
        tmp_path,
        cfg or {},
        "cpu",
        torch.float32,
        backend="sglang",
        parallel_runtime=runtime,
    )


def test_new_native_constructor_and_loader_contract(
    native_env, native_loader, tmp_path
):
    model = _native_wrapper(native_env, tmp_path)
    config, hf = native_loader.constructor
    assert config.arch_config.num_layers == hf["n_layers"] == 30
    assert config.arch_config.num_attention_heads == hf["n_heads"] == 30
    assert config.arch_config.cap_feat_dim == 4096
    assert config.arch_config.axes_lens == (32768, 1024, 1024)
    state, device, dtype, strict, offload, mapping = native_loader.load
    assert list(state) == ["cap_embedder.0.weight"]
    assert device == torch.device("cpu")
    assert dtype == torch.float32 and strict and not offload
    assert mapping is model.model.param_names_mapping
    assert native_loader.selection[1]["require_backend_selection"] is True
    assert not model.model.training and not model.model.weight.is_meta
    assert not model.model.weight.requires_grad


def test_native_load_failure_has_no_diffusers_fallback(
    native_env, native_loader, tmp_path
):
    native_loader.fail = True
    with pytest.raises(RuntimeError, match="native checkpoint failure"):
        _native_wrapper(native_env, tmp_path)


@pytest.mark.parametrize("device", ["cuda", "cuda:2"])
def test_native_normalizes_loader_device(
    native_env, native_loader, monkeypatch, tmp_path, device
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    observed = []
    monkeypatch.setattr(
        ZImageTransformer2DModelWrapper,
        "_load_sglang_model",
        lambda self, path, device, dtype: (
            observed.append(device) or torch.nn.Identity()
        ),
    )
    ZImageTransformer2DModelWrapper(
        tmp_path, {}, device, torch.float32, backend="sglang"
    )
    assert observed == [torch.device("cuda:2")]


def test_native_rejects_wrong_cuda_device(native_env, monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    with pytest.raises(RuntimeError, match="current runtime device"):
        SGLangDecoderRuntime(_config(native_env), "cuda:1", torch.float32)


def test_unknown_native_attention_fails_before_loading(
    native_env, native_loader, tmp_path
):
    runtime = SGLangDecoderRuntime(
        _config(native_env, attention="not_an_attention_backend"), "cpu", torch.float32
    )
    with pytest.raises(ValueError, match="Unknown decoder attention backend"):
        ZImageTransformer2DModelWrapper(
            tmp_path,
            {},
            "cpu",
            torch.float32,
            backend="sglang",
            parallel_runtime=runtime,
        )
    assert native_loader.constructor is native_loader.load is None


@pytest.mark.parametrize(
    "sp,rank,u,ring",
    [
        (1, 0, 1, 1),
        (2, 0, 2, 1),
        (2, 1, 2, 1),
        (4, 0, 2, 2),
        (4, 1, 2, 2),
        (4, 2, 2, 2),
        (4, 3, 2, 2),
    ],
)
@pytest.mark.parametrize("hw", [(16, 32), (32, 16)])
def test_source_spatial_shard_rope_public_forward_gather(
    native_env, native_loader, tmp_path, sp, rank, u, ring, hw
):
    model = _native_wrapper(native_env, tmp_path, sp=sp, rank=rank, u=u, ring=ring)
    native_env.full = torch.arange(2 * 16 * hw[0] * hw[1], dtype=torch.float32).reshape(
        2, 16, 1, *hw
    )
    x = list(native_env.full.unbind(0))
    cap = [torch.ones(17, 4096), torch.zeros(17, 4096)]
    outputs = model(x, torch.tensor([0.25, 0.75]), cap, return_dict=False)[0]
    torch.testing.assert_close(torch.stack(outputs), native_env.full, rtol=0, atol=0)
    call = native_loader.forward[-1]
    assert call["hidden_states"].numel() == native_env.full.numel() // sp
    assert call["encoder_hidden_states"][0] is cap[0]
    torch.testing.assert_close(call["timestep"], torch.tensor([750.0, 250.0]))
    torch.testing.assert_close(call["caption_valid_lens"], torch.tensor([17, 17]))
    cap_pos, img_pos = call["freqs_cis"][0][0], call["freqs_cis"][1][0]
    assert cap_pos.shape == (32, 3) and cap_pos[0].tolist() == [1, 0, 0]
    offset = rank * (max(hw) // 2 // sp)
    assert img_pos[0].tolist() == (
        [33, 0, offset] if hw[1] > hw[0] else [33, offset, 0]
    )
    assert call["image_seq_len_target"] == (None if sp == 1 else 128 // sp)
    assert native_env.contexts[-1] == {
        "current_timestep": 0,
        "attn_metadata": None,
        "forward_batch": None,
    }


def test_native_rebuilds_geometry_cache(native_env, native_loader, tmp_path):
    model = _native_wrapper(native_env, tmp_path, sp=2, rank=1)
    cap = [torch.zeros(17, 4096)]
    for h, w in [(16, 32), (32, 16), (16, 64)]:
        native_env.full = torch.ones(1, 16, 1, h, w)
        out = model(list(native_env.full.unbind(0)), 0.5, cap).sample
        assert out[0].shape == (16, 1, h, w)
        assert model._native_cache[1].raw_latent_shape == (1, 16, 1, h, w)
        assert model._native_cache[1].prompt_embeds is None


@pytest.mark.parametrize("sp,rank", [(2, 0), (2, 1), (4, 0), (4, 1), (4, 2), (4, 3)])
@pytest.mark.parametrize("transpose", [False, True])
def test_source_uneven_spatial_shards_gather_without_extra_learned_padding(
    native_env, native_loader, tmp_path, sp, rank, transpose
):
    model = _native_wrapper(native_env, tmp_path, sp=sp, rank=rank, u=2, ring=sp // 2)
    hw = (14, 34 if sp == 2 else 70)
    if transpose:
        hw = hw[::-1]
    native_env.full = torch.arange(16 * hw[0] * hw[1], dtype=torch.float32).reshape(
        1, 16, 1, *hw
    )
    output = model(list(native_env.full.unbind(0)), 0.5, [torch.ones(17, 4096)]).sample
    torch.testing.assert_close(torch.stack(output), native_env.full, rtol=0, atol=0)
    assert native_loader.forward[-1]["image_seq_len_target"] == 64
    plan = model._native_cache[1]._zimage_sp_plan
    assert len(set(plan["shard_sizes_tok"])) > 1


@pytest.mark.parametrize("failure", ["ragged", "frames", "padding", "empty"])
def test_native_unsupported_shapes_fail_before_forward(
    native_env, native_loader, tmp_path, failure
):
    model = _native_wrapper(native_env, tmp_path, sp=4, rank=3, u=2, ring=2)
    x = [torch.ones(16, 1, 16, 32)]
    cap = [torch.ones(17, 4096)]
    if failure == "ragged":
        x.append(torch.ones(16, 1, 32, 16))
        cap.append(cap[0])
    elif failure == "frames":
        x = [torch.ones(16, 2, 16, 32)]
    elif failure == "padding":
        x = [torch.ones(16, 1, 60, 80)]
    else:
        x = [torch.ones(16, 1, 2, 2)]
    with pytest.raises(RuntimeError, match="native input preparation failed"):
        model(x, 0.5, cap)
    assert native_loader.forward == []


def test_invalid_head_split_rejected(native_env, native_loader, tmp_path):
    with pytest.raises(ValueError, match="divisible by Ulysses"):
        _native_wrapper(native_env, tmp_path, sp=4, u=4)


def test_follower_does_not_load_sigvq_vae_or_emit_bytes(
    native_env, monkeypatch, tmp_path
):
    cfg = _config(native_env, 2, rank=1)
    decoder = image_module.LLaDA2ImageDecoder(
        str(tmp_path),
        device="cpu",
        dtype=torch.float32,
        backend="sglang",
        stage_role=cfg.stage_role,
        sp_rank=cfg.sp_rank,
        sp_size=2,
        ulysses_degree=2,
    )
    monkeypatch.setattr(
        decoder, "_ensure_sigvq", lambda: pytest.fail("follower loaded SigVQ")
    )
    monkeypatch.setattr(
        decoder, "_ensure_vae", lambda: pytest.fail("follower loaded VAE")
    )
    monkeypatch.setattr(decoder, "_ensure_diff_model", lambda mode: None)
    decoder._diff_config = {}
    decoder._diff_model = lambda **kwargs: ([torch.ones_like(x) for x in kwargs["x"]],)
    assert decoder.decode_to_bytes([1], 1, 1, num_steps=2, seed=37) is None
    assert (4, 4096) in native_env.broadcasts


@pytest.mark.parametrize("mode", ["normal", "decoder-turbo"])
@pytest.mark.parametrize("seed", [None, 11])
def test_leader_follower_sampling_uses_identical_noise(
    native_env, monkeypatch, tmp_path, mode, seed
):
    from sglang_omni.models.llada2_uni.components import decoder_parallel

    monkeypatch.setattr(decoder_parallel.secrets, "randbits", lambda bits: 37)
    trajectories = []
    for rank in range(2):
        cfg = _config(native_env, 2, rank=rank)
        decoder = image_module.LLaDA2ImageDecoder(
            str(tmp_path),
            "cpu",
            torch.float32,
            backend="sglang",
            stage_role=cfg.stage_role,
            sp_rank=rank,
            sp_size=2,
            ulysses_degree=2,
        )
        calls = []

        def model(calls=calls, **kwargs):
            calls.append(torch.stack(kwargs["x"]).clone())
            return ([x * 0.01 for x in kwargs["x"]],)

        decoder._sigvq = lambda ids: torch.ones(1, ids.shape[1], 4096)
        decoder._vae = SimpleNamespace(
            config=SimpleNamespace(scaling_factor=2, shift_factor=0.5),
            decode=lambda x, return_dict: (x[:, :3].tanh(),),
        )
        decoder._diff_model = model
        decoder._diff_config = {}
        decoder._diff_model_mode = mode
        result = decoder.decode([1], 1, 1, decode_mode=mode, num_steps=4, seed=seed)
        assert (result is None) == (rank == 1)
        trajectories.append(calls)
    assert len(trajectories[0]) == len(trajectories[1]) > 1
    for leader, follower in zip(*trajectories):
        torch.testing.assert_close(leader, follower, rtol=0, atol=0)
