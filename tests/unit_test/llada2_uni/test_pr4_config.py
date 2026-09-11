# SPDX-License-Identifier: Apache-2.0
"""Use the existing typed placement/engine schema for PR4, without a TP variant."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from sglang_omni.models.llada2_uni.config import LLaDA2UniOmniPipelineConfig


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_existing_typed_tp_and_memory_configuration(tp_size):
    from sglang_omni.config.runtime import resolve_stage_factory_arg_defaults

    config = LLaDA2UniOmniPipelineConfig(model_path="unused")
    data = config.model_dump()
    thinker = next(s for s in data["stages"] if s["name"] == "thinker")
    thinker.update(
        tp_size=tp_size,
        gpu=list(range(tp_size)),
        process="thinker",
        gpu_memory_fraction=0.7,
    )
    thinker["engine"]["json_model_override_args"] = (
        '{"llada2_router_topk_backend":"triton"}'
    )
    thinker["factory"]["dllm_algorithm_config"] = "/tmp/pr4-decode.yaml"
    rebuilt = LLaDA2UniOmniPipelineConfig.model_validate(data)
    stage = next(s for s in rebuilt.stages if s.name == "thinker")
    defaults = resolve_stage_factory_arg_defaults(stage, rebuilt)
    assert stage.tp_size == tp_size and stage.gpu == list(range(tp_size))
    assert stage.engine.mem_fraction_static == 0.75
    assert stage.engine.quantization is None
    assert defaults["total_gpu_memory_fraction"] == 0.7


def test_stage_factory_forwards_rank_topology_and_budget(monkeypatch):
    captured = {}

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    def build(path, **kwargs):
        captured["server"] = kwargs
        return NS(**kwargs, mem_fraction_static=0.75)

    def schedule(args, gpu, **kwargs):
        captured["scheduler"] = (args, gpu, kwargs)
        return "scheduler"

    stub(
        "sglang_omni.models.llada2_uni.bootstrap",
        create_dllm_thinker_scheduler=schedule,
    )
    stub(
        "sglang_omni.scheduling.sglang_backend",
        build_sglang_server_args=build,
        pin_resolved_device_type=lambda args, device: args.update(device=device),
    )
    stub(
        "sglang_omni.utils.device",
        resolve_concrete_device=lambda *_: NS(type="cuda", index=3),
    )
    stub("sglang.srt.arg_groups.model_override_base", resolved_view=lambda x: x)
    path = (
        Path(__file__).resolve().parents[3] / "sglang_omni/models/llada2_uni/stages.py"
    )
    spec = importlib.util.spec_from_file_location("_pr4_stages", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = module.create_sglang_dllm_thinker_executor_from_config
    assert (
        factory(
            "unused",
            tp_size=4,
            tp_rank=2,
            nccl_port=29000,
            total_gpu_memory_fraction=0.7,
        )
        == "scheduler"
    )
    assert captured["server"]["tp_size"] == 4
    assert captured["server"]["device"] == "cuda"
    assert captured["server"]["disable_cuda_graph"] is True
    assert captured["scheduler"][1:] == (
        3,
        {
            "tp_rank": 2,
            "nccl_port": 29000,
            "total_gpu_memory_fraction": 0.7,
        },
    )
    with pytest.raises(ValueError, match="topology"):
        factory("unused", tp_size=4, server_args_overrides={"tp_size": 1})
    with pytest.raises(ValueError, match="tp_rank"):
        factory("unused", tp_size=2, tp_rank=2)
