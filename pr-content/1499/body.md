## Motivation

Add non-thinking text-to-image generation and single-image editing to LLaDA2-Uni. The thinker produces semantic VQ tokens; a dedicated decoder turns them into a PNG through SigVQ conditioning, a diffusion transformer, and a VAE.

## Modifications

- Add image-generation preprocessing, exact image-token budgets, task routing, and the image-decoding terminal stage.
- Add native `/v1/images/generations` and `/v1/images/edits` endpoints on image-capable pipelines, while retaining the chat-completions image entry point.
- Support Diffusers and SGLang decoder backends. Diffusers is the default; `factory.backend="sglang"` selects the alternative explicitly.
- Support `normal` and `decoder-turbo` decoding, with per-request decoder steps and seed.
- Preprocess raw edit images on the server using an aspect-ratio-compatible crop grid, resize, and center crop. Understanding preprocessing remains separate.
- Add API, preprocessing, decoder, routing, and error-path tests, including invalid image controls and context-budget errors.

## Usage

All image responses are non-streaming and contain one PNG.

### Native image endpoints

Send JSON to `POST /v1/images/generations`:

```json
{
  "prompt": "A red apple on a wooden table",
  "size": "1024x1024",
  "guidance_scale": 4.0,
  "seed": 42,
  "response_format": "b64_json"
}
```

T2I accepts `size: "WIDTHxHEIGHT"` or a `width`/`height` pair; the default is 1024×1024. Dimensions must be positive multiples of 32. If both forms are supplied, they must agree. `guidance_scale` maps to thinker `cfg_scale`, and `num_inference_steps` maps to `decoder_steps`.

`POST /v1/images/edits` accepts multipart form data with exactly one source image upload or URL and a non-empty `prompt`. Edit dimensions follow the processed source grid, so `size`, `width`, and `height` are not accepted.

The native response uses `data[]`: `response_format="b64_json"` returns `data[0].b64_json`; `"url"` returns a PNG data URL in `data[0].url`. It is not a chat-completions envelope. Multiple-image `n`, masks, and non-PNG output are unsupported.

### Chat completions

Use `POST /v1/chat/completions` with `stream: false` and `modalities: ["image"]`. An `image_generation` object also selects the image-generation path; a source image selects editing rather than T2I.

```json
{
  "messages": [{"role": "user", "content": "A red apple on a wooden table"}],
  "modalities": ["image"],
  "stream": false,
  "image_generation": {
    "mode": "normal",
    "image_h": 1024,
    "image_w": 1024,
    "cfg_scale": 4.0,
    "seed": 42
  }
}
```

The chat entry point uses `image_generation.image_h`/`image_w`, not the native endpoint's `size`/`width`/`height` fields. It returns `choices[0].message.image = {"data": "<base64 PNG>", "format": "png"}`.

Shared controls are `decode_mode`, `decoder_steps`, `dllm_steps`, `seed`, and `cfg_rescale`. T2I uses `cfg_scale`; use `cfg_text_scale` and `cfg_image_scale` for editing. `resolution_multiplier` is a decoder factory setting, not a per-request field; the default multiplier of 2 maps the semantic grid to the requested T2I pixel size.

## Scope and dependencies

Depends directly on #2257 for thinker CFG and numerical correctness. The comparison with `main` includes that prerequisite until the stack is rebased.

This PR owns the single-image pipeline and decoder backends at SP1. Thinking generation is added in #1500, interleaved output in #1502, and decoder SP in #1501. The native `data[]`, single-image chat `message.image`, and interleaved chat `message.segments` responses are distinct contracts.

## Related Issues

Tracked in #2207; continues the work in #445.

## Validation

Candidate revision: `36059f66c45c9afbef69a0ddcfcd279a05ac75ad`, based on `main` at `7dc8909e` and #2257. The suites below ran on `e22e38ae`, before a realtime-transcription-only rebase; the contribution, model runtime, image APIs, and dependencies are unchanged.

- Image API, LLaDA2-Uni, and dLLM scheduler suites: **139 passed, 0 skipped**.
- The opt-in GPU test ran the SGLang SP1 decoder against Diffusers using a small synthetic checkpoint. This is backend numerical coverage, not a full-checkpoint image-quality evaluation.
- Regression coverage includes generation/edit request-budget errors returning 400, runtime failures returning 500, and text-only application startup without importing optional image schemas.
- Environment: Linux, NVIDIA H20-3e, SGLang 0.5.20, Diffusers 0.37.0, PyTorch 2.13.0+cu130, Transformers 5.12.1, and Triton 3.7.1. Applicable formatting and static checks passed.

No full-checkpoint quality parity, latency, or throughput improvement is claimed here.

## Contributors

- @kunchengit
- @btw616
- @wzy-ustc
- @Anmuliar
- @LiRongchuan
