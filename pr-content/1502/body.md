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

Candidate revision: `636ce7b632b220f478401803cf5894e941df9608`, based on `main` at `7dc8909e` and #1500. The regression suite and default-step protocol smoke below ran on `09e7c75e`, before a realtime-transcription-only rebase; the contribution, model runtime, image APIs, and dependencies are unchanged. The reduced-step observation predates that regression run.

- Related unit/regression suites: **1366 passed, 2 skipped**. The skips are engine-contract checks in configurations without an SGLang engine. Coverage includes self-routing, concurrent frame collection, cancellation, late-preparation cleanup, session admission, and image/API boundaries. Applicable formatting and static checks passed.
- A full-checkpoint request at the default 32 dLLM steps returned exactly two 1024×1024 PNG frames. The check verified contiguous segment indices, a shared session ID, segment/media types, PNG hashes and byte lengths, concatenated text, and token accounting. Requests completed without leftover pending work.
- The same sample at 16 dLLM steps failed strict frame validation because the image phase exhausted its sampling budget without EOI. Restoring the default 32 steps produced the required VQ span and EOI in each frame. Rejecting an invalid model-generated frame is intentional; the implementation does not insert EOI or truncate excess tokens. Lower-step output robustness is not established.
- Environment: Linux, NVIDIA H20-3e, SGLang 0.5.20, Diffusers 0.37.0, PyTorch 2.13.0+cu130, Transformers 5.12.1, and Triton 3.7.1. The multi-frame smoke used the default Diffusers decoder, seed 42 and 5 decoder steps.

These are lifecycle, protocol, and representative generation checks, not a multi-frame quality or performance benchmark. No throughput or latency improvement is claimed here.

The September 27 SGLang-decoder review returned two image frames through the interleaved response path. Visual inspection found vertical splits and repeated composition in the ship sample. Frame delivery succeeded, but this sample is not an image-quality pass; the normal-mode benchmark scores in #1499 must not be presented as interleaved quality validation.

## Contributors

- @kunchengit
- @Anmuliar
