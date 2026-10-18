# SPDX-License-Identifier: Apache-2.0
"""Interleaved transitions, frame CFG, and EOI stopping."""

import asyncio
from types import SimpleNamespace

import pytest

from sglang_omni.models.llada2_uni.components.preprocessor import (
    IMAGE_TOKEN_OFFSET,
    LLaDA2Preprocessor,
)
from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload


class Tokenizer:
    mask_token_id = 99
    eos_token_id = 2
    additional_stop_token_ids = [666]

    def convert_tokens_to_ids(self, token):
        return {"<|image|>": 10, "<|/image|>": 11, "<boi>": 12}[token]

    def convert_ids_to_tokens(self, token):
        return f"<|reserved_token_{token - 20}|>"

    def encode(self, text, add_special_tokens=False):
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(token) for token in ids if token != self.eos_token_id)


CONFIG = {
    "mode": "interleaved",
    "max_frames": 3,
    "cfg_text_scale": 4.0,
    "cfg_image_scale": 1.0,
    "seed": 42,
    "decode_mode": "decoder-turbo",
    "decoder_steps": 8,
}


def initial_payload():
    preprocessor = object.__new__(LLaDA2Preprocessor)
    preprocessor.tokenizer = Tokenizer()
    preprocessor.max_seq_len = 8192
    request = OmniRequest(
        inputs=[{"role": "user", "content": "Illustrate a three-frame story"}],
        params={"stream": False},
        metadata={"image_generation": CONFIG},
    )
    return asyncio.run(
        preprocessor(StagePayload(request_id="story", request=request, data={}))
    )


def advance(state, ids):
    state.thinker_out = {"output_ids": ids, "is_final": True}
    advance_interleaved_state(
        state, Tokenizer(), completed_phase=state.stream_state["interleaved"]["phase"]
    )


def test_image_request_uses_current_cfg_and_eoi_only_stopping():
    pytest.importorskip("sglang")
    from sglang_omni.models.llada2_uni.request_builders import (
        build_dllm_thinker_request,
    )

    state = LLaDA2UniPipelineState.from_dict(initial_payload().data)
    advance(state, [71, 10, 22, 23, 12])
    req = build_dllm_thinker_request(
        state,
        params={"stop": ["stop"], "stop_token_ids": [666]},
        tokenizer=Tokenizer(),
        vocab_size=IMAGE_TOKEN_OFFSET + 16,
        dllm_config=SimpleNamespace(
            block_size=4, mask_id=99, requires_separate_context_encoding=False
        ),
        request_id="story",
    ).req
    assert req.tokenizer is None and req.eos_token_ids == {11}
    assert req.sampling_params.stop_strs == []
    assert req.sampling_params.stop_token_ids == {11}
    assert req.sampling_params.max_new_tokens == 1500
    assert len(req.origin_input_ids) == len(
        req._uncond_input_ids
    )  # noqa: leading-underscore  # DLLM protocol
    assert req._cfg_scale == 4.0 and req._allowed_stop_token_ids == (
        11,
    )  # noqa: leading-underscore  # DLLM protocol
    assert req._dllm_steps == 32  # noqa: leading-underscore  # DLLM protocol

    advance(state, [IMAGE_TOKEN_OFFSET, IMAGE_TOKEN_OFFSET + 1])
    req = build_dllm_thinker_request(
        state,
        params={},
        tokenizer=Tokenizer(),
        vocab_size=IMAGE_TOKEN_OFFSET + 16,
        dllm_config=SimpleNamespace(
            block_size=4, mask_id=99, requires_separate_context_encoding=False
        ),
        request_id="story",
    ).req
    assert req.sampling_params.max_new_tokens == 1498


@pytest.mark.parametrize(
    "scale, mode", [(0.0, "simple"), (1.0, "none"), (4.0, "simple")]
)
def test_frame_cfg_preserves_unconditional_and_history_branches(scale, mode):
    state = LLaDA2UniPipelineState.from_dict(initial_payload().data)
    config = state.request_metadata["image_generation"]
    config.update(cfg_scale=0.0, cfg_text_scale=scale, cfg_image_scale=0.0)
    advance(state, [71, 10, 22, 23, 12])
    assert state.stream_state["interleaved"]["cfg_plan"]["mode"] == mode
    advance(state, [IMAGE_TOKEN_OFFSET + i for i in range(6)] + [11])
    advance(state, [72, 10, 22, 23, 12])
    plan = state.stream_state["interleaved"]["cfg_plan"]
    assert plan["mode"] == ("editing" if scale > 0 else "simple")
    assert len(plan["branches"]) == (2 if scale > 0 else 1)
