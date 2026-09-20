> Draft: requesting feedback on the public image API, grouped dLLM scheduling, and decoder integration. Real-checkpoint and GPU validation remain pending.

## Motivation

LLaDA2-Uni supports multimodal understanding in SGLang-Omni, but it cannot yet return generated or edited images. This PR adds non-thinking text-to-image generation and single-image editing, including request preprocessing, grouped classifier-free guidance (CFG), and native VQ-to-image decoding.

## Modifications

- Add image output through `/v1/chat/completions` using `modalities` and `image_generation`, with generated PNGs returned in `message.images[]`.
- Add non-thinking text-to-image and single-image-edit preprocessing, exact image-token budgets, task routing, and an image-decoding terminal stage.
- Derive the image vocabulary boundary from checkpoint configuration instead of a model-specific constant.
- Represent two-branch text CFG and three-branch edit CFG as typed request groups with atomic admission, completion, abort, and cleanup.
- Preserve ragged CFG prompt alignment in eager and CUDA-graph attention planning, with eager execution for layouts that cannot be replayed safely.
- Isolate request-building and result-adaptation failures so one malformed request cannot terminate the shared dLLM scheduler.
- Add lazy SigVQ, Diffusers ZImage, and VAE loading, with focused API, preprocessing, scheduler, attention, decoder, routing, and error-lifecycle tests.

## Public API contract

Image output is requested by including `"image"` in `modalities`. `image_generation` configures the request but does not independently request image output. Image requests are non-streaming.

Successful responses return images in `choices[0].message.images[]`. Each image contains `id`, base64-encoded PNG `data`, `format`, `width`, and `height`.

For text-to-image generation:

- Output dimensions may be supplied with `size: "WIDTHxHEIGHT"` or with `width` and `height`; the default is 1024×1024.
- `resolution_multiplier`, `cfg_scale`, `cfg_rescale`, `seed`, and `decoder_steps` are supported.
- Output dimensions must be divisible by `16 * resolution_multiplier`.

For image editing:

- Exactly one source image and a non-empty instruction are required.
- The output grid follows the processed source image, so `size`, `width`, and `height` are rejected.
- `resolution_multiplier`, `cfg_text_scale`, `cfg_image_scale`, the `cfg_scale` alias, `cfg_rescale`, `seed`, and `decoder_steps` are supported.

This PR supports `mode: "normal"` and PNG output. Thinking mode and decoder-turbo are not included.

## Scope and dependencies

The protocol and client changes for `message.images[]` overlap with #878 and use the same response shape. This overlap will be resolved before the PR is marked Ready; if #878 lands first, the duplicate changes will be removed during rebase.

The grouped request lifecycle and padding-aware attention metadata are implemented in shared dLLM scheduling code. LLaDA2-Uni prompt construction, image vocabulary handling, edit CFG branches, and decoding are model-specific.

## Source attribution

The SigVQ module, image decoder, and portions of image-edit preprocessing are adapted from the Apache-2.0 [LLaDA2.0-Uni reference implementation](https://github.com/inclusionAI/LLaDA2.0-Uni) at commit `3457030a9c737f77f38ad5ff657e7659243d3444`. File-level attribution is recorded in the corresponding source headers.

## Related Issues

Part of #445. This PR does not close the full LLaDA2-Uni support issue.

Related: #878.

## Accuracy Test

### Local verification

- Affected image API, preprocessing, grouped scheduler, attention, decoder, routing, and stage tests: `80 passed`
- Black, isort, Ruff, Python `compileall`, and `git diff --check`: passed

Real-checkpoint token parity, CUDA-graph capture/replay, and image-quality validation have not been run for this Draft.

## Benchmark & Profiling

Not run for this Draft. No performance result is claimed.

## Contributors

- @kunchengit
- @btw616
- @wzy-ustc
- @Anmuliar
- @LiRongchuan

## Checklist

- [x] Format the changed code.
- [x] Add focused unit tests.
- [x] Document the public API and source attribution.
- [ ] Complete real-checkpoint and GPU validation.
- [ ] Provide accuracy and performance results before marking the PR Ready.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

This PR is intentionally opened as a Draft. Self-hosted GPU CI has not been requested.