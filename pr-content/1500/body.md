## Motivation

Add thinking-mode text-to-image generation on top of #1499. The thinker first produces text and an image-begin boundary, then generates the image-token grid in a second pass before image decoding.

## Modifications

- Accept `image_generation.mode="thinking"` for T2I; thinking-mode editing is unsupported.
- Run the first pass with a fixed 2048-token budget and `<boi>` as a stop token, without image-vocabulary constraints or CFG.
- Build the second-pass prompt through the first `<boi>`, then generate the requested image-token grid with CFG and decode it through #1499's backend.
- Return the thinking text when requested and count both passes as completion tokens.
- Add phase-transition, CFG, usage, and normal-mode regression tests.

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

- The two-pass thinking pipeline was tested on **this PR's branch** with the SGLang SP1 decoder, returning thinking text and one PNG. The image was visually inspected. The thinker used TP1 eager execution with compilation disabled.
- Focused LLaDA2-Uni regression: **44 passed, 1 skipped**, including phase-transition and usage-accounting coverage.

Thinking generation is introduced here; decoder SP is added separately in #1501. No thinking-mode quality score or performance gain is claimed; #1499's GenEval/ImgEdit scores cover normal mode.

## Contributors

- @kunchengit
- @btw616
- @wzy-ustc
- @Anmuliar
