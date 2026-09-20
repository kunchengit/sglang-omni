## Status

This is a Draft for review of the generic transfer and request-lifecycle contract. It does not add model-specific image-generation behavior and has been validated with unit and static checks only.

## Motivation

A pipeline request may produce multiple payloads that overlap in flight. The previous lifecycle assumed one transfer and one result per request:

- concurrent payloads for the same source, destination, and request reused the same shared-memory object ID;
- the first routed result could clear request state while later work was still active;
- an abort marker could be consumed once, allowing already queued or late work to run;
- stale TP-follower output could repeat cleanup or surface an error after cancellation.

This capability was extracted to support LLaDA2-Uni interleaved image generation. Its transfer and lifecycle contracts are deliberately model-independent, so other models and pipelines with multiple in-flight stage payloads can reuse it.

Stream-before-payload delivery and stream-chunk relay are separate existing capabilities and are not reimplemented here. This PR is also separate from #1455, which focuses on SSE text-output streaming.

## Modifications

- Assign a unique transfer identity to each `CommEngine` payload while preserving the legacy `StageIO` default when no identity is supplied.
- Add an explicit, default-off scheduler capability for multiple in-flight work items per request.
- Track work consistently across coordinator submissions, upstream stage payloads, and TP follower fanout, including task re-entry while downstream sends are awaiting.
- Keep abort state persistent, filter aborted batches and stale follower output, and make request cleanup idempotent.
- Reject terminal stages that opt into multiple in-flight work until result aggregation semantics are defined, avoiding premature success.
- Add focused regression coverage for transfer identity, ordering, concurrent admission, TP fanout, abort races, stale output, terminal-stage rejection, and cleanup-once behavior.

### Scope boundaries

This PR does **not** implement LLaDA2-Uni request grouping, image API fields, image segment assembly, SSE streaming, or terminal multi-result aggregation.

## Related Issues

Part of #445. This PR does not close the roadmap issue.

Related but non-overlapping: #1455.

## Accuracy Test

No model math or model architecture is changed.

### Local verification

- Focused communication, scheduler, and stage tests: **42 passed, 4 deselected**
- Adjacent KV-transfer, router, and streaming-scheduler regression tests: **27 passed, 3 skipped**
- Ruff check and format check: **passed**
- Python `compileall`: **passed**
- `git diff --check`: **passed**

No GPU or full-runtime validation has been run for this Draft.

## Benchmark & Profiling

Not applicable. This PR makes a correctness and lifecycle change and makes no throughput or latency claim.

## Contributors

- @kunchengit
- @Anmuliar

## Checklist

- [x] Format the changed code.
- [x] Add focused unit tests.
- [x] Document the transfer and lifecycle contract.
- [ ] Complete broader runtime validation before marking the PR Ready.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

This PR is intentionally opened as a Draft. Self-hosted GPU CI has not been requested.