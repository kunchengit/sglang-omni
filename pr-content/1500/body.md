## Motivation

Add thinking-mode text-to-image generation on top of #1499. The thinker first produces text and an image-begin boundary, then generates the image-token grid in a second pass before image decoding.

## Modifications

- Accept `image_generation.mode="thinking"` for T2I; thinking-mode editing is unsupported.
- Run the first pass with a fixed 2048-token budget and `<boi>` as a stop token, without image-vocabulary constraints or CFG.
- Build the second-pass prompt from the original input and generated tokens through the first `<boi>`; validate the context budget and CFG inputs before committing the transition.
- Re-enter the thinker to generate the requested image-token grid, then use the decoder inherited from #1499.
- Retain the generated thinking text for text output and account for both generation passes as completion tokens, without reclassifying the thinking prefix as user input.
- Extend phase-transition, CFG, routing, state-transfer, usage, and non-thinking regression coverage.

## Usage

For a non-streaming chat response containing both the thinking text and PNG:

```json
{
  "messages": [{"role": "user", "content": "Design and illustrate a small coastal village"}],
  "modalities": ["text", "image"],
  "stream": false,
  "image_generation": {
    "mode": "thinking",
    "image_h": 1024,
    "image_w": 1024,
    "cfg_scale": 4.0,
    "seed": 42
  }
}
```

The response retains #1499's single-image shape: `choices[0].message.content` contains the thinking text when text output is requested, and `message.image` contains the PNG's `data` and `format`. Image-only requests omit the text output.

The native `POST /v1/images/generations` entry point inherited from #1499 also accepts `mode: "thinking"`. Its controls are top-level, dimensions use `size` or `width`/`height`, and its response remains `data[]`; it does not return the thinking trace.

The 2048-token first-pass budget is not a separate public control. The generated image header is preserved in the conditional prefix, while the requested grid determines the image-token budget, unconditional header, and decoder dimensions, matching the reference thinking flow. Decoder backend and `normal`/`decoder-turbo` controls are inherited from #1499.

## Scope and dependencies

Depends directly on #1499 and transitively on #2257. Review the thinking-mode delta relative to #1499.

This PR adds model-specific two-pass construction and routing, not streaming HTTP responses or multi-frame output. The shared asynchronous relay and interleaved collector are included in #1502. Thinker TP and decoder SP are separate optimization branches.

## Related Issues

Tracked in #2207; continues the work in #445.

## Validation

Candidate revision: `9cc17c25f10025f37432a03d83b5fabdf165bd17`, based on `main` at `7dc8909e` and #1499. The suites below ran on `ce5d2f45`, before a realtime-transcription-only rebase; the contribution, model runtime, image APIs, and dependencies are unchanged.

- Image API, LLaDA2-Uni, and dLLM scheduler suites: **143 passed, 0 skipped**, including the inherited small-checkpoint GPU decoder comparison.
- Coverage includes the first-BOI boundary, missing-boundary errors, CFG enabled/disabled, state round trips, original prompt versus two-pass completion accounting, and normal-mode regression tests.
- Environment: Linux, NVIDIA H20-3e, SGLang 0.5.20, Diffusers 0.37.0, PyTorch 2.13.0+cu130, Transformers 5.12.1, and Triton 3.7.1. Applicable formatting and static checks passed.

Reference-control-flow checks and small-checkpoint tests do not establish full-model image-quality parity. No performance improvement is claimed here.

The September 27 full-checkpoint review also exercised a thinking T2I request through the SGLang decoder and visually inspected the returned image. This is a single-sample generation check, not a thinking-mode quality benchmark; the full GenEval/ImgEdit scores reported in #1499 exercise normal mode.

## Contributors

- @kunchengit
- @btw616
- @wzy-ustc
- @Anmuliar
