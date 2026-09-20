## Motivation

The LLaDA2-Uni image decoder requires sequence-parallel execution across multiple stage processes. The existing pipeline runtime only provides a TP-specific follower control path and does not model sequence parallelism in configuration, topology, placement, launch metadata, or process lifecycle.

This PR extracts the required sequence-parallel runtime as a model-independent core capability. Although LLaDA2-Uni is the motivating consumer, the configuration and runtime contracts can be reused by other models and pipeline stages that provide an SP-aware stage factory and policy.

## Modifications

- Add model-neutral sequence-parallel configuration and policy validation for pipeline stages.
- Add one-process-per-rank SP placement, topology, launch metadata, and distributed process environment setup.
- Generalize the TP follower control path into shared TP/SP runtime control while preserving the existing TP-facing compatibility aliases.
- Correlate fan-out work, aborts, and terminal acknowledgements by dispatch ID.
- Drain already-fanned-out collective work before cleanup so abort-before-work and cross-queue races cannot strand ranks in collectives.
- Preserve request-ID reuse and the existing behavior of non-parallel schedulers.
- Destroy initialized `torch.distributed` process groups exactly once on both normal and exceptional worker exit.
- Add focused tests for configuration, topology, placement, process lifecycle, dispatch-aware abort draining, scheduler capabilities, and non-SP compatibility.

This PR intentionally does not include LLaDA2-Uni decoder math, image APIs, T2I/edit behavior, model-specific kernels, or the generic multi-inflight lifecycle changes from #1487.

## Related Issues

Part of #445. This PR does not close the roadmap issue.

Related framework work:

- #1486: LLaDA2-Uni thinker TP correctness and model-level CUDA graph support.
- #1487: generic multi-inflight stage payload lifecycle.

## Accuracy Test

No model math, model weights, or model architecture is changed.

### Local verification

- Focused SP configuration, topology, placement, and runtime tests: **93 passed, 2 deselected**
- Non-SP scheduler regression tests: **26 passed**
- Black, isort, Ruff, Python `compileall`, and `git diff --check`: **passed**

Real multi-GPU/NCCL execution and LLaDA2-Uni SP1/SP2/SP4 parity validation have not been run for this Draft. They remain required before the PR is marked Ready.

## Benchmark & Profiling

Not applicable for this Draft. This PR introduces a reusable runtime capability and makes no throughput or latency claim.

## Contributors

- @kunchengit
- @Anmuliar

## Checklist

- [x] Format the changed code.
- [x] Add focused unit tests.
- [x] Document the sequence-parallel runtime contract.
- [ ] Complete real multi-GPU/NCCL and decoder-integration validation.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

This PR is intentionally opened as a Draft. Self-hosted GPU CI has not been requested. Real multi-GPU/NCCL and LLaDA2-Uni decoder integration validation will be added after maintainers confirm the core runtime interface and scope.