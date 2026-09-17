from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.llada2_uni.components.preprocessor import IMAGE_TOKEN_OFFSET
from sglang_omni.models.llada2_uni.config import (
    DECODE_STAGE,
    IMAGE_DECODE_STAGE,
    THINKER_STAGE,
    LLaDA2UniInterleavedPipelineConfig,
)
from sglang_omni.models.llada2_uni.interleaved import (
    InterleavedGenerationConfig,
    build_cfg_plan,
    parse_image_header,
)
from sglang_omni.models.llada2_uni.merge import build_interleaved_content
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import (
    _advance_interleaved_state,
    _align_cfg_branch_group,
    make_dllm_thinker_scheduler_adapters,
)
from sglang_omni.models.llada2_uni.routing import interleaved_decoder_next, thinker_next
from sglang_omni.proto import OmniRequest, StagePayload


class _Tokenizer:
    mask_token_id = 99
    eos_token_id = 2
    SOI = 101
    BOI = 102
    EOI = 103
    HEIGHT = 104
    WIDTH = 105
    UNCOND = 106

    _token_to_id = {
        "<|image|>": SOI,
        "<boi>": BOI,
        "<|/image|>": EOI,
        "<uncondition>": UNCOND,
    }

    def convert_tokens_to_ids(self, token):
        return self._token_to_id.get(token, -1)

    def convert_ids_to_tokens(self, token_id):
        return {
            self.HEIGHT: "<|reserved_token_2|>",
            self.WIDTH: "<|reserved_token_2|>",
        }.get(token_id, f"token-{token_id}")

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<uncondition>":
            return [self.UNCOND]
        return [900]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "|".join(str(token_id) for token_id in token_ids)


def _metadata(**overrides):
    config = {
        "max_frames": 2,
        "max_image_tokens": 16,
        "cfg_scale": 4.0,
        "cfg_text_scale": 7.5,
        "cfg_image_scale": 1.5,
    }
    config.update(overrides)
    return {"interleaved_generation": config}


def _payload(state, request_id="request"):
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="prompt"),
        data=state.to_dict(),
    )


def test_config_and_image_header_validation():
    config = InterleavedGenerationConfig.from_metadata(
        _metadata(max_frames=3, dllm_steps=8)
    )
    assert config.max_frames == 3
    assert config.decoder_steps == 8
    stream_state = config.to_stream_state(prompt_length=10, max_seq_len=8192)
    assert stream_state["interleaved_image_dllm_steps"] == 8
    assert "interleaved_dllm_steps" not in stream_state

    tokenizer = _Tokenizer()
    header = parse_image_header(
        [77, tokenizer.SOI, tokenizer.HEIGHT, tokenizer.WIDTH, tokenizer.BOI],
        tokenizer,
    )
    assert header.soi_position == 1
    assert header.grid_h == 2
    assert header.grid_w == 2
    assert header.image_token_count == 4

    with pytest.raises(ValueError, match="did not end"):
        parse_image_header([tokenizer.SOI, tokenizer.HEIGHT], tokenizer)
    with pytest.raises(ValueError, match="max_frames"):
        InterleavedGenerationConfig.from_metadata(_metadata(max_frames=0))


def test_scheduler_adapters_roundtrip_two_frames_and_cfg_phases():
    tokenizer = _Tokenizer()
    metadata = _metadata(dllm_steps=8)
    config = InterleavedGenerationConfig.from_metadata(metadata)
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[10, 11]])},
        task_kind="interleaved",
        request_metadata=metadata,
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
    )
    build, adapt = make_dllm_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=IMAGE_TOKEN_OFFSET + 8192,
        dllm_config=SimpleNamespace(block_size=32, mask_id=99),
    )
    payload = _payload(state)
    for frame in (1, 2):
        text = build(payload)
        assert text.req._interleaved_phase == "text"
        assert not hasattr(text.req, "_dllm_steps")
        assert not hasattr(text.req, "_uncond_input_ids")
        assert text.req.eos_token_ids == {tokenizer.BOI, tokenizer.eos_token_id}
        text.output_ids = [
            70 + frame,
            tokenizer.SOI,
            tokenizer.HEIGHT,
            tokenizer.WIDTH,
            tokenizer.BOI,
        ]
        payload = adapt(text)
        image = build(payload)
        assert image.req._interleaved_phase == "image"
        assert image.req._dllm_steps == 8
        assert image.req.eos_token_ids == {tokenizer.EOI}
        assert image.req.sampling_params.max_new_tokens == 1500
        assert len(image.req.origin_input_ids) == len(image.req._uncond_input_ids)
        assert hasattr(image.req, "_uncond_img_input_ids") == (frame == 2)
        if frame == 2:
            assert len(image.req._uncond_img_input_ids) == len(
                image.req.origin_input_ids
            )
        image.output_ids = list(range(IMAGE_TOKEN_OFFSET, IMAGE_TOKEN_OFFSET + 4)) + [
            tokenizer.EOI
        ]
        payload = adapt(image)
        result = LLaDA2UniPipelineState.from_dict(payload.data)
        assert result.thinker_out["output_ids"] == image.output_ids[:-1]
        assert result.stream_state["interleaved_frame_index"] == frame
        assert result.stream_state["interleaved_emit_frame"]
        assert result.stream_state.get("interleaved_done", False) == (frame == 2)


def test_cfg_plan_matches_reference_first_and_later_frame_conditions():
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    suffix = [tokenizer.SOI, tokenizer.HEIGHT, tokenizer.WIDTH, tokenizer.BOI]

    first_ids = [10, 11] + suffix
    first = build_cfg_plan(
        full_ids=first_ids,
        header=parse_image_header(first_ids, tokenizer),
        frame_index=0,
        tokenizer=tokenizer,
        config=config,
    )
    assert first.mode == "simple"
    assert first.cfg_scale == 4.0
    assert first.branches["uncond"] == [900] + suffix

    later_ids = [20, tokenizer.EOI, 30, 31] + suffix
    later = build_cfg_plan(
        full_ids=later_ids,
        header=parse_image_header(later_ids, tokenizer),
        frame_index=1,
        tokenizer=tokenizer,
        config=config,
    )
    assert later.mode == "editing"
    assert later.branches["no_text"] == [20, tokenizer.EOI, tokenizer.UNCOND] + suffix
    assert later.branches["no_image"] == [900, 30, 31] + suffix


def test_image_budget_survives_api_and_partial_phase_reentry():
    from sglang_omni.serve.protocol import InterleavedGenerationParams

    tokenizer = _Tokenizer()
    metadata = {
        "interleaved_generation": InterleavedGenerationParams(
            image_max_new_tokens=1600
        ).model_dump()
    }
    config = InterleavedGenerationConfig.from_metadata(metadata)
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[10, 11]])},
        task_kind="interleaved",
        request_metadata=metadata,
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
    )
    build, adapt = make_dllm_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=IMAGE_TOKEN_OFFSET + 8192,
        dllm_config=SimpleNamespace(block_size=32, mask_id=99),
    )
    text = build(_payload(state))
    text.output_ids = [tokenizer.SOI, tokenizer.HEIGHT, tokenizer.WIDTH, tokenizer.BOI]
    image = build(adapt(text))
    assert image.req.sampling_params.max_new_tokens == 1600
    image.output_ids = [IMAGE_TOKEN_OFFSET, IMAGE_TOKEN_OFFSET + 1]
    continuation = build(adapt(image))
    assert continuation.req.sampling_params.max_new_tokens == 1598


def test_image_grid_must_fit_generation_budget_including_eoi():
    tokenizer = _Tokenizer()
    metadata = _metadata(image_max_new_tokens=4)
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[10]])},
        task_kind="interleaved",
        request_metadata=metadata,
        stream_state=InterleavedGenerationConfig.from_metadata(
            metadata
        ).to_stream_state(prompt_length=1, max_seq_len=8192),
        thinker_out={
            "output_ids": [
                tokenizer.SOI,
                tokenizer.HEIGHT,
                tokenizer.WIDTH,
                tokenizer.BOI,
            ]
        },
    )
    with pytest.raises(ValueError, match="budget cannot fit"):
        _advance_interleaved_state(state, tokenizer, completed_phase="text")


@pytest.mark.parametrize("scale, mode", [(0.0, "simple"), (1.0, "none")])
def test_cfg_plan_matches_hf_simple_cfg_activation(scale, mode) -> None:
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(
        _metadata(cfg_scale=scale, cfg_text_scale=0.0, cfg_image_scale=0.0)
    )
    full_ids = [10, tokenizer.SOI, tokenizer.HEIGHT, tokenizer.WIDTH, tokenizer.BOI]

    plan = build_cfg_plan(
        full_ids=full_ids,
        header=parse_image_header(full_ids, tokenizer),
        frame_index=0,
        tokenizer=tokenizer,
        config=config,
    )

    assert plan.mode == mode
    assert plan.cfg_scale == scale
    assert bool(plan.branches) == (scale != 1.0)


def test_cfg_plan_keeps_three_branches_when_later_frame_disables_image_guidance() -> (
    None
):
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(
        _metadata(cfg_scale=0.0, cfg_text_scale=4.0, cfg_image_scale=0.0)
    )
    suffix = [tokenizer.SOI, tokenizer.HEIGHT, tokenizer.WIDTH, tokenizer.BOI]
    full_ids = [20, tokenizer.EOI, 30, 31] + suffix

    plan = build_cfg_plan(
        full_ids=full_ids,
        header=parse_image_header(full_ids, tokenizer),
        frame_index=1,
        tokenizer=tokenizer,
        config=config,
    )

    assert plan.mode == "editing"
    assert plan.branches["no_text"] == [20, tokenizer.EOI, tokenizer.UNCOND] + suffix
    assert plan.branches["no_image"] == [900, 30, 31] + suffix
    assert plan.cfg_text_scale == 4.0
    assert plan.cfg_image_scale == 0.0


def test_cfg_attachment_aligns_dynamic_and_pre_aligned_branches() -> None:
    tokenizer = _Tokenizer()
    tokenizer.mask_token_id = 99
    branches, pads = _align_cfg_branch_group(
        tokenizer=tokenizer,
        branches={"cond": [10, 11, 12], "uncond": [7, 8], "no_image": [99, 6, 5]},
        existing_left_pad_lens={"no_image": 1},
    )
    assert branches == {
        "cond": [10, 11, 12],
        "uncond": [99, 7, 8],
        "no_image": [99, 6, 5],
    }
    assert pads == {"cond": 0, "uncond": 1, "no_image": 1}


def test_interleaved_text_length_finishes_the_current_segment() -> None:
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2]])},
        thinker_out={"output_ids": [70, 71], "is_final": True},
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
        request_metadata=_metadata(),
        task_kind="interleaved",
    )

    _advance_interleaved_state(
        state,
        tokenizer,
        completed_phase="text",
        finish_reason="length",
    )

    assert state.stream_state["interleaved_phase"] == "done"
    assert state.stream_state["interleaved_finish_reason"] == "length"
    assert state.stream_state["interleaved_trailing_text"] == "70|71"
    assert state.prompt["input_ids"].flatten().tolist() == [1, 2, 70, 71]
    assert state.thinker_out["output_ids"] == [70, 71]


def test_interleaved_state_machine_fans_out_one_frame_then_reenters_text():
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2]])},
        thinker_out={
            "output_ids": [
                77,
                tokenizer.SOI,
                tokenizer.HEIGHT,
                tokenizer.WIDTH,
                tokenizer.BOI,
            ],
            "is_final": True,
        },
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
        request_metadata=_metadata(),
        task_kind="interleaved",
    )

    _advance_interleaved_state(state, tokenizer, completed_phase="text")
    assert state.stream_state["interleaved_phase"] == "image"
    assert state.stream_state["interleaved_cfg_plan"]["mode"] == "simple"
    assert state.stream_state["interleaved_current_frame"]["image_token_count"] == 4

    vq_tokens = [IMAGE_TOKEN_OFFSET + index for index in range(4)]
    state.thinker_out = {"output_ids": vq_tokens[:2], "is_final": True}
    _advance_interleaved_state(state, tokenizer, completed_phase="image")
    assert state.stream_state["interleaved_phase"] == "image"
    assert (
        state.stream_state["interleaved_current_frame"]["remaining_image_tokens"] == 2
    )
    assert thinker_next("request", _payload(state)) == THINKER_STAGE

    state.thinker_out = {
        "output_ids": vq_tokens[2:] + [tokenizer.EOI],
        "is_final": True,
    }
    _advance_interleaved_state(state, tokenizer, completed_phase="image")
    assert state.thinker_out["output_ids"] == vq_tokens

    assert state.stream_state["interleaved_phase"] == "text"
    assert state.stream_state["interleaved_frame_index"] == 1
    assert state.stream_state["interleaved_emit_frame"] is True
    assert state.stream_state["interleaved_needs_reentry"] is True
    assert state.prompt["input_ids"].flatten().tolist()[-5:] == vq_tokens + [
        tokenizer.EOI
    ]

    payload = _payload(state)
    assert thinker_next(payload.request_id, payload) == IMAGE_DECODE_STAGE
    state.stream_state.pop("interleaved_emit_frame")
    assert (
        interleaved_decoder_next(payload.request_id, _payload(state)) == THINKER_STAGE
    )


def _state_at_image_phase() -> tuple[LLaDA2UniPipelineState, _Tokenizer]:
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2]])},
        thinker_out={
            "output_ids": [
                77,
                tokenizer.SOI,
                tokenizer.HEIGHT,
                tokenizer.WIDTH,
                tokenizer.BOI,
            ],
            "is_final": True,
        },
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
        request_metadata=_metadata(),
        task_kind="interleaved",
    )
    _advance_interleaved_state(state, tokenizer, completed_phase="text")
    return state, tokenizer


def test_interleaved_text_phase_rejects_image_vocab_tokens() -> None:
    tokenizer = _Tokenizer()
    config = InterleavedGenerationConfig.from_metadata(_metadata())
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[1, 2]])},
        thinker_out={
            "output_ids": [
                IMAGE_TOKEN_OFFSET + 7,
                tokenizer.SOI,
                tokenizer.HEIGHT,
                tokenizer.WIDTH,
                tokenizer.BOI,
            ],
            "is_final": True,
        },
        stream_state=config.to_stream_state(prompt_length=2, max_seq_len=8192),
        request_metadata=_metadata(),
        task_kind="interleaved",
    )

    with pytest.raises(ValueError, match="text phase emitted image token"):
        _advance_interleaved_state(state, tokenizer, completed_phase="text")


def test_interleaved_image_phase_requires_eoi_after_all_vq_tokens() -> None:
    state, tokenizer = _state_at_image_phase()
    vq_tokens = [IMAGE_TOKEN_OFFSET + index for index in range(4)]

    state.thinker_out = {"output_ids": vq_tokens[:2] + [tokenizer.EOI]}
    with pytest.raises(ValueError, match="EOI with 2 VQ tokens remaining"):
        _advance_interleaved_state(state, tokenizer, completed_phase="image")

    state, tokenizer = _state_at_image_phase()
    state.thinker_out = {"output_ids": vq_tokens}
    with pytest.raises(ValueError, match="without EOI"):
        _advance_interleaved_state(state, tokenizer, completed_phase="image")


def test_interleaved_image_phase_rejects_text_token_before_eoi() -> None:
    state, tokenizer = _state_at_image_phase()
    state.thinker_out = {"output_ids": [77, tokenizer.EOI]}

    with pytest.raises(ValueError, match="non-image token before EOI"):
        _advance_interleaved_state(state, tokenizer, completed_phase="image")


def test_serial_decoder_content_preserves_generation_order() -> None:
    state = LLaDA2UniPipelineState(
        task_kind="interleaved",
        stream_state={
            "interleaved_segments": [
                {"frame_index": 1, "text": "first"},
                {"frame_index": 2, "text": "second"},
            ],
            "interleaved_decoded_frames": [
                {"frame_index": 1, "data": "a", "format": "png"},
                {"frame_index": 2, "data": "b", "format": "png"},
            ],
            "interleaved_trailing_text": "done",
        },
    )

    assert build_interleaved_content(state) == [
        {"type": "text", "text": "first"},
        {
            "type": "image",
            "image": {"frame_index": 1, "data": "a", "format": "png"},
        },
        {"type": "text", "text": "second"},
        {
            "type": "image",
            "image": {"frame_index": 2, "data": "b", "format": "png"},
        },
        {"type": "text", "text": "done"},
    ]


def test_interleaved_pipeline_topology_is_serial():
    single = LLaDA2UniInterleavedPipelineConfig(model_path="model")
    stages = {stage.name: stage for stage in single.stages}
    assert stages[THINKER_STAGE].gpu == 0
    assert stages[IMAGE_DECODE_STAGE].gpu == 0
    assert stages[IMAGE_DECODE_STAGE].terminal is False
    assert stages[DECODE_STAGE].terminal is True
    assert stages[THINKER_STAGE].stream_to == []
    assert stages[IMAGE_DECODE_STAGE].stream_to == []


def test_interleaved_yaml_uses_current_config_resolver():
    from sglang_omni.config.manager import ConfigManager

    root = Path(__file__).resolve().parents[3]
    config = ConfigManager.from_file(
        str(root / "examples" / "configs" / "llada2_uni_interleaved.yaml")
    ).config
    stages = {stage.name: stage for stage in config.stages}
    assert stages[IMAGE_DECODE_STAGE].factory.backend == "diffusers"
    assert stages[THINKER_STAGE].engine.disable_cuda_graph is False
    assert stages[THINKER_STAGE].engine.overrides()["cuda_graph_bs"] == [1, 2, 3, 4]
    assert stages[IMAGE_DECODE_STAGE].gpu == 0
