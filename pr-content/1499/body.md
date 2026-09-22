## Motivation

Add non-thinking text-to-image generation and single-image editing to LLaDA2-Uni. The thinker generates image VQ tokens, then a dedicated image decoder converts them to a PNG through SigVQ conditioning, a diffusion transformer, and a VAE.

## Modifications

- Add generation/edit request preprocessing, task state, request construction, CFG branch construction, image-token generation, and decoder routing.
- Support both Diffusers and SGLang image-decoder backends through the decoder stage's `factory.backend` configuration. Diffusers remains the default; backend selection is explicit.
- Support normal and decoder-turbo decoding with per-request decoder steps and seed.
- Perform edit preprocessing on the server: choose an aspect-ratio-compatible crop grid around a 512x512 pixel budget, resize to cover that grid, then center crop before image encoding. Understanding requests keep their separate preprocessing path.
- Remove the evaluation-only `source_image_tokens` / `.pt` public input. Precomputed-token experiments belong in an external evaluation harness.
- Return one generated PNG through the existing chat-completions response and document the generation controls.

## Public API contract

Use `/v1/chat/completions` with `stream: false`. Include `"image"` in `modalities` to request image output. An `image_generation` object also selects the generation path; source-image presence distinguishes edit from T2I.

Example T2I request:

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

- T2I uses `image_h` and `image_w`, defaulting to 1024x1024; supplied dimensions must be positive multiples of 32.
- Edit requires exactly one source image and a non-empty instruction. Its generated grid follows the server-processed source image rather than the T2I dimension controls.
- Shared controls include `decode_mode` (`normal` or `decoder-turbo`), `decoder_steps`, `dllm_steps`, `seed`, and `cfg_rescale`.
- T2I uses `cfg_scale`; edit exposes `cfg_text_scale` and `cfg_image_scale`, with `cfg_scale` retained as the text-guidance alias when `cfg_text_scale` is absent.
- `resolution_multiplier` is a decoder factory setting, not a per-request image-generation field.

On this branch, the single-image response is `choices[0].message.image = {"data": "<base64 PNG>", "format": "png"}`. It is not the interleaved `message.images[]` contract introduced in #1502. No new HTTP endpoint is added.

Thinking mode, interleaved output, thinker TP, and decoder SP are outside this PR's incremental scope.

## Scope and dependencies

Stacked on `llada2/thinker-fix`, which supplies the CFG scheduling and padding-aware attention correctness prerequisite. Review this PR relative to that branch; its comparison with `main` currently includes the prerequisite.

This PR owns the LLaDA2-Uni image pipeline and both decoder integrations at SP1. Thinking generation follows in #1500. Any overlap with #878's image-response work must be reconciled before merge; the current branch does not claim to implement that PR's response format.

Part of #445; this does not close the full roadmap.

## Accuracy Test

- The affected image-decoder and serving API unit suites passed after the latest synchronization.
- Server-side resize/crop was compared with the reference preprocessing on 16 cases, with identical processed pixels.
- Real-checkpoint HTTP smoke tests on the updated native-image branch completed both T2I and raw-image edit requests and produced viewable PNGs (1024x1024 and 864x1152 respectively).

The smoke tests used BF16 LLaDA2.0-Uni on H20-3e with SGLang 0.5.19 and Torch 2.13.0+cu130. They establish request-to-image execution, not full benchmark quality. Historical precomputed-token edit scores are not evidence for the current raw-image path. Both decoder backends still need final-head coverage before merge.

## Benchmark & Profiling

No steady-state performance claim is made for this PR. Cold smoke-test latency is not a benchmark.

## Contributors

- @kunchengit
- @btw616
- @wzy-ustc
- @Anmuliar
- @LiRongchuan
