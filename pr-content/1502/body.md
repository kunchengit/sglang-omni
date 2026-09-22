## Motivation

Allow one LLaDA2-Uni request to alternate text and multiple generated images. Completed image-token spans can be sent to the decoder while the thinker continues, and the final response preserves their order.

## Modifications

- Add the interleaved text/image state machine, generated image-header handling, VQ-span extraction, frame limits, and request context budgets.
- Preserve the original system prompt, accumulated generation history, and phase-specific budgets/CFG across thinker re-entry.
- Dispatch a separate frame payload when an image span completes, then continue the thinker without waiting for that frame's decoder result.
- Include shared self-route and multi-inflight stage lifecycle support. The interleaved decoder is nonterminal; frame results and thinker completion meet in a collector keyed by request and frame identity.
- Isolate collector state across request completion, abort, and request-ID reuse, and wait for outstanding frames before finalizing a successful response.
- Add ordered text/image-reference content and stable frame IDs, with image bytes stored once per frame.
- Provide `examples/configs/llada2_uni_interleaved.yaml` and matching client handling.

## Public API contract

Use `/v1/chat/completions` on a deployment using the interleaved pipeline configuration:

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

This path accepts text-only input and PNG output. If `modalities` is supplied, it must request both text and image. Source images, client-specified output dimensions, unknown interleaved controls, and `stream: true` are rejected. Image dimensions come from the generated image headers.

Supported controls are `max_frames`, `text_max_new_tokens`, `image_max_new_tokens`, `max_image_tokens`, `dllm_steps`, `cfg_scale`, `cfg_text_scale`, `cfg_image_scale`, `cfg_rescale`, `decoder_steps`, `seed`, `format`, and `decode_mode`. Decoder mode may be `normal` or `decoder-turbo`. `max_frames` defaults to 10 and has no fixed 64-frame ceiling; token/context budgets still apply.

The response uses `choices[0].message.content[]` for ordered segments and `choices[0].message.images[]` for PNG payloads:

```json
{
  "content": [
    {"type": "text", "text": "First scene"},
    {"type": "image_ref", "image_id": "image-request-0"},
    {"type": "text", "text": "Then the story continues"}
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

Every image reference maps to one image in the same order. Existing single-image normal/thinking/edit responses retain their `message.image` field. This is an Omni extension to chat completions, not a new endpoint or an external SSE image-streaming protocol. Internal asynchronous relay does not imply cross-request dynamic batching.

## Scope and dependencies

Stacked on #1500 (`llada2/thinking-image-generation`), inheriting #1499 and the thinker correctness prerequisite.

The current branch includes the shared relay/lifecycle work formerly separated as #1487. It must not be described as requiring a second copy of that code to land first. Reconcile the overlapping #1487 scope before merge; review both the model state machine and the shared runtime delta here.

Thinker TP (#1486) and decoder SP (#1501) are independent follow-ups based on this branch. Part of #445; this does not close the full roadmap.

## Accuracy Test

The latest stack synchronization passed LLaDA2-Uni and image API unit checks covering request/response contracts, phase transitions, frame collection, and lifecycle behavior.

Earlier GPU runs exercised multi-frame generation and inspected the returned images. They do not establish exact agreement with an HF baseline or a fresh quality result for the latest head. Final-head multi-frame quality and abort/concurrent-request serving should be rechecked before merge.

## Benchmark & Profiling

No current-head latency or throughput claim. Internal relay permits thinker/decoder overlap but does not guarantee a speedup on every GPU placement.

## Contributors

- @kunchengit
- @Anmuliar
