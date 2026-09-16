# SPDX-License-Identifier: Apache-2.0
"""PR6 request validation and ordered non-streaming chat output."""

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sglang_omni.client import Client
from sglang_omni.client.types import CompletionResult
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import _build_chat_generate_request
from sglang_omni.serve.protocol import (
    ChatCompletionRequest,
    InterleavedGenerationParams,
)

CONTENT = [
    {"type": "text", "text": "first"},
    {"type": "image", "image": {"data": "a", "format": "png", "frame_index": 1}},
    {"type": "text", "text": "second"},
    {"type": "image", "image": {"data": "b", "format": "png", "frame_index": 2}},
]


class RecordingClient:
    def __init__(self, content=CONTENT):
        self.requests = []
        self.content = content

    async def completion(self, request, **kwargs):
        self.requests.append(request)
        return CompletionResult(request_id="r", text="aggregate", content=self.content)

    async def completion_stream(self, *args, **kwargs):
        pytest.fail("image streaming reached client")
        yield


def post(client, **fields):
    with TestClient(create_app(client, model_name="llada2-uni")) as api:
        return api.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Draw two frames"}],
                **fields,
            },
        )


@pytest.mark.parametrize("modalities", [None, ["text", "image"], ["image", "text"]])
def test_ordered_content_and_defaults_reach_pipeline(modalities):
    client = RecordingClient()
    response = post(
        client, interleaved_generation={"max_frames": 128}, modalities=modalities
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": CONTENT}
    assert client.requests[0].output_modalities == (modalities or ["text", "image"])
    omni = Client._build_omni_request(client.requests[0])
    assert omni.metadata["interleaved_generation"] == InterleavedGenerationParams(
        max_frames=128
    ).model_dump(exclude_none=True)
    assert omni.metadata["output_modalities"] == (modalities or ["text", "image"])


def test_interleaved_params_preserve_sampling_and_zero_overrides():
    params = dict(
        max_frames=2,
        text_max_new_tokens=16,
        dllm_steps=1,
        cfg_scale=0.0,
        cfg_text_scale=0.0,
        cfg_image_scale=0.0,
        cfg_rescale=0.0,
        decode_mode="normal",
        decoder_steps=1,
        seed=0,
        max_image_tokens=1024,
    )
    req = ChatCompletionRequest(
        messages=[
            {
                "role": "user",
                "content": ["Draw", {"type": "text", "text": "two frames"}],
            }
        ],
        interleaved_generation=params,
        temperature=0.0,
        seed=0,
        max_completion_tokens=64,
        stage_sampling={"thinker": {"max_new_tokens": 32, "top_p": 0.9}},
        stage_params={"thinker": {"return_logprob": True}},
        talker_top_k=0,
    )
    generate = _build_chat_generate_request(req)
    omni = Client._build_omni_request(generate)
    assert omni.metadata["interleaved_generation"] == {
        **params,
        "image_max_new_tokens": 1500,
    }
    assert omni.params["temperature"] == 0.0 and omni.params["seed"] == 0
    assert omni.params["max_new_tokens"] == 64
    assert omni.params["stage_sampling"]["thinker"]["max_new_tokens"] == 32
    assert omni.params["stage_params"] == req.stage_params
    assert generate.extra_params["talker_top_k"] == 0
    assert "temperature" in omni.metadata[EXPLICIT_GENERATION_PARAMS_KEY]
    assert generate.messages[0].content == req.messages[0].content


@pytest.mark.parametrize(
    "modalities",
    [
        [],
        ["text"],
        ["image"],
        ["audio"],
        ["text", "image", "audio"],
        ["text", "text", "image"],
    ],
)
def test_incompatible_interleaved_modalities_rejected_without_dispatch(modalities):
    client = RecordingClient()
    response = post(client, interleaved_generation={}, modalities=modalities)
    assert response.status_code == 400
    assert "requires text and image" in response.json()["detail"]
    assert not client.requests


@pytest.mark.parametrize(
    "fields,detail",
    [
        ({"image_generation": {}}, "mutually exclusive"),
        ({"stream": True}, "does not support streaming"),
        ({"images": ["image.png"]}, "text-only"),
        ({"audios": ["audio.wav"]}, "text-only"),
        ({"videos": ["video.mp4"]}, "text-only"),
    ],
)
def test_invalid_modes_rejected_before_dispatch(fields, detail):
    client = RecordingClient()
    response = post(client, interleaved_generation={}, **fields)
    assert response.status_code == 400 and detail in response.json()["detail"]
    assert not client.requests


@pytest.mark.parametrize(
    "content",
    [
        None,
        123,
        [{"type": "text", "text": 123}],
        [{"image_url": {"url": "image.png"}}],
        [{"type": "image_url", "image_url": "image.png"}],
        [{"type": "input_audio", "input_audio": {"data": "a"}}],
        [{"type": "video_url", "video_url": {"url": "a"}}],
        [{"type": "image", "image": "a"}],
    ],
)
def test_non_text_message_content_rejected(content):
    client = RecordingClient()
    response = post(
        client,
        interleaved_generation={},
        messages=[{"role": "user", "content": content}],
    )
    assert response.status_code == 400 and "text-only" in response.json()["detail"]
    assert not client.requests


@pytest.mark.parametrize(
    "field", ["cfg_scale", "cfg_text_scale", "cfg_image_scale", "cfg_rescale"]
)
@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", -0.1])
def test_invalid_cfg_rejected(field, value):
    with pytest.raises(ValidationError):
        InterleavedGenerationParams(**{field: value})
    client = RecordingClient()
    response = post(client, interleaved_generation={field: value})
    assert response.status_code == 422 and not client.requests


@pytest.mark.parametrize(
    "field",
    [
        "max_frames",
        "text_max_new_tokens",
        "dllm_steps",
        "decoder_steps",
        "max_image_tokens",
    ],
)
def test_positive_generation_limits(field):
    with pytest.raises(ValidationError):
        InterleavedGenerationParams(**{field: 0})


@pytest.mark.parametrize("content", [CONTENT, []])
@pytest.mark.parametrize("modalities", [[], ["text"], ["image"], ["text", "image"]])
def test_filtering_never_reintroduces_aggregate_text(content, modalities):
    client = RecordingClient(content)
    response = post(client, modalities=modalities)
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message == {
        "role": "assistant",
        "content": [p for p in content if p["type"] in modalities],
    }
    assert client.content == content


def test_text_only_chat_keeps_string_content():
    response = post(RecordingClient(None))
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "aggregate"
