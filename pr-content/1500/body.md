> Draft: requesting feedback on the thinking-mode transition and same-stage re-entry lifecycle. Real-checkpoint and GPU validation remain pending.

## Motivation

LLaDA2-Uni thinking-mode image generation first produces a reasoning trace and an image-begin boundary, then generates the VQ image-token grid in a second dLLM pass. This PR adds that two-phase flow on top of the native image-generation support in #1499.

## Modifications

- Accept `image_generation.mode="thinking"` for text-to-image requests while keeping image editing in normal mode.
- Run Phase 1 with the thinking prompt, a fixed 2048-token budget, and `<boi>` as a stop token, without image-vocabulary constraints or CFG.
- Atomically build the Phase 2 prompt from the Phase 1 output and create the existing CFG group only after the transition is fully validated.
- Re-enter the thinker stage for Phase 2 and consume the re-entry marker exactly once.
- Validate the combined prompt, Phase 1, and image-grid context budget before execution.
- Keep missing-`<boi>` and transition failures request-scoped without partially mutating pipeline state.
- Retire a completed same-process pass before synchronous self-dispatch and preserve abort behavior during re-entry.
- Return the Phase 1 reasoning trace only when both text and image modalities are requested.
- Add focused coverage for both phases, context boundaries, transition atomicity, CFG restoration, routing, self-reentry, abort races, and non-thinking behavior.

## Public API contract

Thinking mode reuses the image API introduced in #1499:

```json
{
  "modalities": ["image"],
  "stream": false,
  "image_generation": {
    "mode": "thinking"
  }
}
```

The generated image is returned in `choices[0].message.images[]`. When both `"text"` and `"image"` are requested, `message.content` additionally contains the Phase 1 reasoning trace.

The Phase 1 generation budget is fixed at 2048 tokens and is not a public request parameter. Streaming and thinking-mode image editing are not supported.

## Dependencies and review scope

This branch is stacked on #1499, so the comparison with `main` temporarily includes that dependency. Review should focus on the thinking-mode changes. The branch will be rebased after #1499 is resolved.

The prompts and two-phase transition are LLaDA2-Uni specific. The same-stage lifecycle change in `pipeline/stage/runtime.py` is shared runtime code and has independent self-route and abort coverage.

## Related Issues

Part of #445. This PR does not close the full LLaDA2-Uni support issue.

Required dependency: #1499.

## Accuracy Test

### Local verification

- Thinking-mode and same-stage re-entry tests: `16 passed`
- Black, isort, Ruff, Python `compileall`, and `git diff --check`: passed

Real-checkpoint thinking-token parity, CUDA-graph capture/replay, and image-quality validation have not been run for this Draft.

## Benchmark & Profiling

Not run for this Draft. No performance result is claimed.

## Contributors

- @kunchengit
- @btw616
- @wzy-ustc
- @Anmuliar

## Checklist

- [x] Format the changed code.
- [x] Add focused unit and lifecycle tests.
- [x] Document the public behavior and stacked dependency.
- [ ] Complete real-checkpoint and GPU validation.
- [ ] Provide accuracy and performance results before marking the PR Ready.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

This PR is intentionally opened as a Draft. Self-hosted GPU CI has not been requested.
