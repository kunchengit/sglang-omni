## Motivation

Allow one LLaDA2-Uni request to alternate text and multiple generated images. Each completed image-token span can be decoded while the thinker continues, and the final response preserves segment order.

## Modifications

- Add the text/image state machine with generated image-header parsing, exact VQ-span extraction, frame limits, and context budgets.
- Preserve accumulated history and phase-specific CFG/budgets across thinker re-entry.
- Dispatch isolated frame payloads to a nonterminal decoder and collect results by request and frame identity before final completion.
- Include the shared self-route and multi-inflight stage lifecycle needed for that flow, including cancellation and late-preparation cleanup.
- Expose ordered text/image `segments` with inline PNG media, a concatenated text view, matching client handling, and `examples/configs/llada2_uni_interleaved.yaml`.

## Usage

The validated three-frame example uses this deployment configuration. Set `model_path` to your LLaDA2.0-Uni checkpoint and select either `sglang` or `diffusers` for the decoder backend:

```yaml
config_cls: LLaDA2UniInterleavedPipelineConfig
model_path: /path/to/LLaDA2.0-Uni
stages:
  thinker:
    gpu: 0
    gpu_memory_fraction: 0.65
    engine:
      mem_fraction_static: 0.55
      max_total_tokens: 24576
      max_running_requests: 4
      disable_cuda_graph: true
      enable_torch_compile: false
  image_decode:
    gpu: 0
    gpu_memory_fraction: 0.25
    factory:
      backend: sglang
      attention_backend: torch_sdpa
      decode_mode: decoder-turbo
      num_steps: 8
```

Launch with the configuration saved as `interleaved.yaml`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang_omni.cli serve \
  --config interleaved.yaml --host 127.0.0.1 --port 8000
```

Send this payload to `POST /v1/chat/completions`:

```json
{
  "model": "llada2-uni",
  "messages": [{"role": "user", "content": "Please generate a sequence of 3 frames showing an aerial view of a rusted shipwreck resting on a shallow coral reef, with the camera gradually moving closer and adjusting angle to reveal the full structure and surrounding ocean environment."}],
  "modalities": ["text", "image"],
  "stream": false,
  "temperature": 0.0,
  "seed": 42,
  "max_tokens": 8192,
  "image_generation": {
    "mode": "interleaved",
    "max_frames": 3,
    "text_max_new_tokens": 8192,
    "dllm_steps": 32,
    "decode_mode": "decoder-turbo",
    "decoder_steps": 8,
    "cfg_scale": 0.0,
    "cfg_text_scale": 4.0,
    "cfg_image_scale": 1.0,
    "cfg_rescale": 0.5,
    "seed": 42
  }
}
```

With the payload in `request.json`:

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @request.json > response.json
```

`dllm_steps` controls image-VQ generation; text generation keeps the scheduler's 32-step block schedule. Decoder steps are separate. The example explicitly sets the tested `cfg_rescale=0.5` instead of relying on the 0.7 default.

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

Tested the usage example through `/v1/chat/completions` on this PR's interleaved pipeline, once per decoder backend, with the same full LLaDA2.0-Uni checkpoint and seed. Runs used BF16, TP1/SP1, an eager thinker with compilation disabled, and `torch_sdpa` decoder attention. The request used the default 32 image-VQ steps, made explicit in the example above.

| Decoder | Result |
| --- | --- |
| Diffusers | **3/3 PNG frames**, 1344x768; HTTP 200; shipwreck sample passed visual review |
| SGLang SP1 | **3/3 PNG frames**, 1344x768; HTTP 200; shipwreck sample passed visual review |

Images were retrieved for side-by-side inspection. Both responses preserved ordered text/image segments and passed PNG, hash, and byte-length checks. All three SGLang images matched the historical shipwreck sample pixel-for-pixel; the two backends produced visually similar results. This is a sample-level visual check, not a full quality benchmark or a claim of cross-backend pixel equality.

Focused LLaDA2-Uni regression: **54 passed, 1 skipped**.

## Contributors

- @kunchengit
- @Anmuliar
