> Draft: requesting feedback on the LLaDA2-Uni interleaved state machine and ordered response contract. Real-checkpoint and GPU validation remain pending.

## Motivation

Interleaved generation alternates text and image segments within one logical request. Each completed image frame must be decoded asynchronously while the thinker re-enters itself to continue generation, and the final response must preserve segment order without duplicating image bytes or introducing a model-specific endpoint.

This PR adds the LLaDA2-Uni interleaved integration by building on the native image-generation flow, same-stage re-entry, and multi-inflight lifecycle.

## Modifications

- Add strict validation for interleaved image-generation requests and their supported controls.
- Add an atomic text/image state machine with dynamic image headers, exact VQ-token and EOI handling, context budgeting, grouped CFG alignment, and bounded multi-frame generation.
- Re-enter the thinker after each frame while asynchronously dispatching isolated payloads to the image decoder.
- Support multiple in-flight payloads only for the nonterminal interleaved decoder and collect decoded frames by request and frame index.
- Add stable frame IDs and ordered text/image-reference responses while keeping image bytes only in `message.images[]`.
- Preserve ordered content in the client, reuse existing LLaDA2-Uni GPU budgets, add an example pipeline configuration, and cover API, routing, lifecycle, collector, and response behavior with unit tests.

## Public API contract

Interleaved generation uses the existing chat-completions endpoint:

```json
{
  "modalities": ["text", "image"],
  "stream": false,
  "image_generation": {
    "mode": "interleaved"
  }
}
```

This initial implementation supports text-only input, PNG output, and the normal decoder. It rejects source-image input, explicit `width`, `height`, or `size`, streaming, alternate output formats or decoder modes, and unknown parameters.

Optional controls are `max_frames`, `text_max_new_tokens`, `max_image_tokens`, `cfg_scale`, `cfg_text_scale`, `cfg_image_scale`, `cfg_rescale`, `decoder_steps`, and `seed`.

Image bytes are returned only in `choices[0].message.images[]`. Ordered output is represented in `choices[0].message.content[]`:

```json
{
  "content": [
    {"type": "text", "text": "First segment"},
    {"type": "image_ref", "image_id": "image-request-0"},
    {"type": "text", "text": "Second segment"}
  ],
  "images": [
    {
      "id": "image-request-0",
      "data": "<base64 PNG>",
      "format": "png",
      "width": 1024,
      "height": 1024
    }
  ]
}
```

Each `image_ref` must correspond to exactly one entry in `message.images[]` in the same order. Base64 data is not duplicated in `content[]`. Existing T2I and image-edit response behavior is unchanged.

## Dependencies and review scope

This PR depends on #1499, #1500, and #1487. The current comparison with `main` temporarily includes integration commits for those unmerged prerequisites so the full interleaved path can be reviewed and tested. After the prerequisites are resolved, the branch will be rebased so the Ready PR contains only the LLaDA2-Uni interleaved integration.

## Related Issues

Part of #445. This PR does not close the full LLaDA2-Uni support issue.

Required dependencies:

- #1499: native image-generation pipeline and shared image response
- #1500: thinking-mode generation and same-stage thinker re-entry
- #1487: model-neutral multi-inflight stage lifecycle

## Accuracy Test

### Local verification

- Interleaved feature and contract tests: `68 passed`
- Client completion regressions: `10 passed`
- Black, isort, Ruff, Python `compileall`, and `git diff --check`: passed

Real-checkpoint output parity, GPU execution, CUDA-graph capture/replay, and multi-frame image quality have not been validated for this Draft.

## Benchmark & Profiling

Not run for this Draft; no latency, memory, or throughput claim is made. GPU parity, image quality, memory, and latency measurements are required before the PR is marked Ready.

## Contributors

- @kunchengit
- @Anmuliar

## Checklist

- [x] Format the changed code.
- [x] Add unit tests.
- [x] Add the public request/response contract and example configuration.
- [ ] Provide real-checkpoint and GPU validation before marking the PR Ready.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

This PR is intentionally opened as a Draft. Self-hosted GPU CI has not been requested.