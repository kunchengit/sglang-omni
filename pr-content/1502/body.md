## Motivation

Allow one LLaDA2-Uni request to alternate text and multiple generated images. Each completed image-token span can be decoded while the thinker continues, and the final response preserves segment order.

## Modifications

- Add the text/image state machine with generated image-header parsing, exact VQ-span extraction, frame limits, and context budgets.
- Preserve accumulated history and phase-specific CFG/budgets across thinker re-entry.
- Dispatch isolated frame payloads to a nonterminal decoder and collect results by request and frame identity before final completion.
- Include the shared self-route and multi-inflight stage lifecycle needed for that flow, including cancellation and late-preparation cleanup.
- Expose ordered text/image `segments` with inline PNG media, a concatenated text view, matching client handling, and `examples/configs/llada2_uni_interleaved.yaml`.

## Usage

Use `POST /v1/chat/completions` on a deployment configured with the interleaved pipeline:

```json
{
  "messages": [{"role": "user", "content": "Tell an illustrated story in three scenes"}],
  "modalities": ["text", "image"],
  "stream": false,
  "image_generation": {
    "mode": "interleaved",
    "max_frames": 3,
    "decoder_steps": 20,
    "seed": 42
  }
}
```

This path accepts text-only input and PNG output. If `modalities` is supplied, it must contain both text and image. Source images, client-specified dimensions, unknown interleaved controls, and streaming are rejected. Frame dimensions come from generated image headers, which determine the required VQ-token count and CFG header. A completed frame must contain that exact VQ span followed by EOI; `image_max_new_tokens` remains an independent sampling upper bound, also limited by the available context.

Supported controls are `max_frames`, `text_max_new_tokens`, `image_max_new_tokens`, `max_image_tokens`, `dllm_steps`, `cfg_scale`, `cfg_text_scale`, `cfg_image_scale`, `cfg_rescale`, `decoder_steps`, `seed`, `format`, and `decode_mode`. Decoder mode may be `normal` or `decoder-turbo`; `format` is `png`. `max_frames` defaults to 10; token/context limits apply independently.

The response exposes ordered `choices[0].message.segments`; `message.content` is the concatenated plain-text view. For example, a message containing text followed by an image has this shape:

```json
{
  "content": "First scene",
  "segments": [
    {
      "type": "segment",
      "session_id": "request-1",
      "segment_index": 0,
      "kind": "text",
      "data": "First scene"
    },
    {
      "type": "segment",
      "session_id": "request-1",
      "segment_index": 1,
      "kind": "image",
      "data": {
        "kind": "image",
        "mime_type": "image/png",
        "url": "data:image/png;base64,<PNG_BASE64>",
        "sha256": "<SHA256_OF_PNG_BYTES>",
        "size_bytes": 12345
      }
    }
  ]
}
```

Segment indices are contiguous from zero and share one request-execution `session_id`. Image data includes a PNG data URL, the decoded bytes' SHA-256 digest, and their byte length; the example uses placeholder media values. The separate standard omni pipeline retains `message.image` for single-image chat and `data[]` for native image endpoints. The interleaved deployment exposes this chat-completions extension, not native image endpoints or an external SSE stream.

## Scope and dependencies

Depends directly on #1500, inheriting #1499 and #2257. This PR includes the shared relay work previously proposed in #1487; the older PR is superseded and is not an additional dependency.

The relay/lifecycle changes are model-neutral infrastructure motivated by LLaDA2-Uni. Header interpretation, CFG construction, and the generation state machine remain model-specific. Asynchronous thinker/decoder overlap does not implement cross-request dynamic or continuous batching.

#1486 and #1501 are sibling optimization branches based on this PR, not prerequisites for the basic interleaved feature.

## Related Issues

Tracked in #2207; continues the work in #445.

## Validation

The interleaved pipeline added in this PR completed the same three-frame request with both decoder backends on the server:

| Decoder | Result |
| --- | --- |
| Diffusers | **3/3 PNG frames**, 1344x768; HTTP 200 |
| SGLang SP1 | **3/3 PNG frames**, 1344x768; HTTP 200 |

Both responses preserved ordered text/image segments and passed image-format, hash, and size checks. Generated images were retrieved and inspected. This validates generation and response handling, not prompt fidelity or cross-backend pixel equality.

Settings: BF16 LLaDA2.0-Uni, TP1 eager thinker, compilation disabled, identical prompt and seed 42, 32 dLLM steps, and 8 decoder-turbo steps. Focused LLaDA2-Uni regression: **54 passed, 1 skipped**.

## Contributors

- @kunchengit
- @Anmuliar
