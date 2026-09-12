# SPDX-License-Identifier: Apache-2.0
"""Ordered terminal content is authoritative and preserves other result fields."""

import asyncio
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from sglang_omni.client import Client, ClientError
from sglang_omni.client.types import GenerateChunk, GenerateRequest
from sglang_omni.proto import StreamMessage

CONTENT = [
    {"type": "text", "text": "first"},
    {"type": "image", "image": {"data": "a", "format": "png", "frame_index": 1}},
    {"type": "text", "text": "second"},
    {"type": "image", "image": {"data": "b", "format": "jpeg", "frame_index": 2}},
]


@pytest.mark.parametrize("content", [CONTENT, []])
def test_canonical_content_survives_completion_without_legacy_duplicates(content):
    result = {
        "content": content,
        "image": "legacy-must-not-duplicate",
        "text": "aggregate",
        "language": "English",
        "finish_reason": "length",
        "token_ids": [1, 2],
        "logprobs": [-0.1, -0.2],
        "output_token_logprobs": [[-0.1, 1]],
        "omni_rollout": {"version": 1},
        "weight_version": "v6",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        "engine_time_s": 0.1,
        "audio_data": [0.0, 0.1],
        "sample_rate": 24000,
    }

    class Coordinator:
        async def submit(self, *args):
            return result

    chunk = Client._default_result_builder("r", result)
    assert chunk.content == content
    assert chunk.token_ids == [1, 2] and chunk.logprobs == [-0.1, -0.2]
    completion = asyncio.run(
        Client(Coordinator()).completion(
            GenerateRequest(prompt="draw", stream=False), request_id="r"
        )
    )
    assert completion.content == content
    assert completion.text == "aggregate" and completion.language == "English"
    assert completion.audio is not None and completion.audio.transcript == "aggregate"
    assert completion.usage.total_tokens == 5 and completion.usage.engine_time_s == 0.1
    assert completion.output_token_logprobs == [[-0.1, 1]]
    assert (
        completion.omni_rollout == {"version": 1} and completion.weight_version == "v6"
    )
    assert completion.finish_reason == "length"
    assert not {"image", "images", "interleaved_content"}.intersection(
        asdict(completion)
    )
    assert result["content"] == content


@pytest.mark.parametrize("content", [CONTENT, []])
def test_merged_canonical_content_overrides_legacy_image(content):
    chunk = Client._default_result_builder(
        "r",
        {
            "decode": {"text": "summary", "language": "English"},
            "image_decode": {"image": "legacy"},
            "content": content,
        },
    )
    assert chunk.content == content and chunk.language == "English"


@pytest.mark.parametrize("content", ["text", [None], ["text"], {"type": "text"}])
def test_malformed_terminal_content_fails_explicitly(content):
    with pytest.raises(ClientError, match="ordered list"):
        Client._default_result_builder("r", {"content": content})


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize(
    "payload",
    [
        {"image": "legacy"},
        {"content": CONTENT},
        GenerateChunk(request_id="r", content=CONTENT),
    ],
)
def test_unexpected_image_stream_is_rejected_and_closed(terminal, payload):
    closed = []

    class Coordinator:
        async def stream(self, *args):
            try:
                if terminal:
                    yield SimpleNamespace(result=payload)
                else:
                    yield StreamMessage(
                        request_id="r", from_stage="decode", chunk=payload
                    )
            finally:
                closed.append(True)

    async def consume():
        return [
            c
            async for c in Client(Coordinator()).completion_stream(
                GenerateRequest(prompt="draw", stream=True), request_id="r"
            )
        ]

    with pytest.raises(ClientError, match="does not support streaming"):
        asyncio.run(consume())
    assert closed == [True]


@pytest.mark.parametrize(
    "payload", [{"modality": "text"}, GenerateChunk(request_id="r")]
)
def test_outer_image_modality_cannot_be_hidden_by_text_chunk(payload):
    msg = StreamMessage(
        request_id="r", from_stage="decode", chunk=payload, modality="image"
    )
    with pytest.raises(ClientError, match="does not support streaming"):
        Client._default_stream_builder("r", msg)


def test_legacy_image_format_is_preserved():
    chunk = Client._default_result_builder("r", {"image": "a", "format": "jpeg"})
    assert chunk.content == [
        {"type": "image", "image": {"data": "a", "format": "jpeg"}}
    ]
