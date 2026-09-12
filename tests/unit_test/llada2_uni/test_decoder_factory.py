# SPDX-License-Identifier: Apache-2.0
"""Exercise decoder placement and factory lifecycle without checkpoint loading."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from PIL import Image

from sglang_omni.models.llada2_uni.config import (
    LLaDA2ImageDecoderStageConfig,
    LLaDA2UniOmniPipelineConfig,
)


@pytest.mark.parametrize("size", [1, 2])
def test_decoder_config_roundtrip(size):
    original = LLaDA2UniOmniPipelineConfig(model_path="unused")
    data = original.model_dump()
    decoder = next(s for s in data["stages"] if s["name"] == "image_decode")
    assert decoder["process"] != "pipeline"
    decoder.update(sp_size=size, gpu=list(range(size)))
    decoder["factory"].update(backend="sglang", ulysses_degree=size)
    rebuilt = LLaDA2UniOmniPipelineConfig.model_validate(data)
    stage = next(s for s in rebuilt.stages if s.name == "image_decode")
    assert isinstance(stage, LLaDA2ImageDecoderStageConfig)
    assert stage.sp_size == size and stage.tp_size == 1


@pytest.mark.parametrize("role", ["single", "leader", "follower"])
def test_decoder_factory_context_and_terminal_output(monkeypatch, role):
    from sglang_omni.models.llada2_uni import merge, stages
    from sglang_omni.models.llada2_uni.components import decoder_runtime, image_decoder
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    events = []
    settings = {}

    class Runtime:
        @contextmanager
        def compute_context(self):
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")

        def close(self):
            events.append("close")

    def initialize(path, **kwargs):
        settings.update(kwargs)
        return Runtime()

    class Decoder:
        def __init__(self, **kwargs):
            assert events[-1] == "enter"
            assert kwargs["backend"] == "sglang"

        def decode(self, tokens, h, w, **kwargs):
            assert events[-1] == "enter"
            assert kwargs == {"decode_mode": "decoder-turbo", "num_steps": 8}
            assert tokens == [3, 4] and (h, w) == (1, 2)
            return None if role == "follower" else Image.new("RGB", (8, 8))

    monkeypatch.setattr(decoder_runtime, "initialize_decoder_runtime", initialize)
    monkeypatch.setattr(image_decoder, "LLaDA2ImageDecoder", Decoder)
    monkeypatch.setattr(
        merge,
        "extract_image_vq_tokens",
        lambda state: ([3, 4], 1, 2, {"decode_mode": "decoder-turbo"}),
    )
    size = 1 if role == "single" else 2
    scheduler = stages.create_image_decode_executor(
        "unused",
        device="cpu",
        backend="sglang",
        sp_size=size,
        sp_rank=int(role == "follower"),
        stage_role=role,
        ulysses_degree=size,
        nccl_port=29001,
    )
    payload = SimpleNamespace(data={})
    output = scheduler._fn(payload)
    assert settings["sp_size"] == size and settings["nccl_port"] == 29001
    if role == "follower":
        assert output is None
    else:
        assert output is payload
        assert payload.data["format"] == "png" and payload.data["image"]
    assert events == ["enter", "exit", "enter", "exit"]

    def finish(self):
        events.append("compute-finished")

    monkeypatch.setattr(SimpleScheduler, "start", finish)
    scheduler.start()
    assert events[-2:] == ["compute-finished", "close"]
