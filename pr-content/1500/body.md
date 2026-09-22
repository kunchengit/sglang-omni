## Motivation

Add thinking-mode T2I on top of native image generation: the thinker first generates text up to the image-begin token, then generates the image-token grid and sends it to the existing decoder.

## Modifications

- Extend `image_generation.mode` with `thinking` for text-to-image requests; image editing remains normal mode only.
- Run the first phase with a 2048-token thinking budget and `<boi>` as a stop token, without image-vocabulary constraints or image CFG.
- Build the second-phase prompt from the original prompt and generated tokens through the first `<boi>`. Preserve that generated prefix and construct the image CFG branches at the transition.
- Validate the transition and context budget before updating request state; treat a missing image-begin boundary as a request error.
- Re-enter the thinker through the existing stage routing, generate the image-token grid, and use the decoder from #1499.
- Retain the generated thinking text for the text response when text output is requested.

## Public API contract

Use the existing non-streaming `/v1/chat/completions` image-generation API:

```json
{
  "messages": [{"role": "user", "content": "Design and illustrate a small coastal village"}],
  "modalities": ["text", "image"],
  "stream": false,
  "image_generation": {
    "mode": "thinking",
    "cfg_scale": 4.0,
    "seed": 42
  }
}
```

The PNG is returned in `choices[0].message.image` with `data` and `format`, as in #1499. With both text and image modalities, `message.content` contains the first-phase generated text. Image-only requests do not request that text output.

The first-phase 2048-token budget is not a separate public control. Image dimensions, guidance, decoder mode, decoder steps, and seed reuse #1499's controls. This PR does not add thinking-mode edit, multi-frame interleaving, or streaming HTTP responses.

## Scope and dependencies

Stacked directly on #1499 (`llada2/native-image-generation`). Review the delta from that branch; image preprocessing, decoder loading, CFG scheduling, and the single-image response are inherited.

This change is model-specific phase construction and routing. The shared multi-inflight relay implementation is included with #1502, not introduced here. Thinker TP and image-decoder SP remain separate follow-ups.

## Roadmap

This PR is re-submitted under the new [LLaDA-Uni roadmap (#2207)](https://github.com/sgl-project/sglang-omni/issues/2207), carrying forward the earlier work tracked in #445 with a rebased implementation and updated scope. It covers **Phase 1: thinking-mode image generation**. It does not close the full roadmap.

## Accuracy Test

The LLaDA2-Uni and image API unit suites passed after rebasing onto the updated native-image branch, covering phase transitions, missing-`<boi>` failures, CFG construction, and normal/thinking request separation.

Earlier real-checkpoint tests exercised both ordinary T2I and thinking T2I during the branch split. Those are historical smoke results, not fresh GPU validation of this head after the latest input-preprocessing synchronization. No full image-quality benchmark or token-exact parity claim is made.

## Benchmark & Profiling

No current-head performance claim. Thinking latency includes an additional generation phase and should not be compared to normal T2I without reporting both phase times.

## Contributors

- @kunchengit
- @btw616
- @wzy-ustc
- @Anmuliar
