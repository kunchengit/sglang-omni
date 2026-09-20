## Status

Draft for scope and design feedback. This PR covers LLaDA2-Uni thinker tensor-parallel correctness and model-level decode CUDA graph support. Real-model parity, GPU validation, and performance measurements are required before either part becomes Ready.

## Motivation

The pipeline already launches TP stage processes, but the thinker factory did not propagate `tp_rank`, `tp_size`, or the NCCL port. `DllmScheduler` also did not opt into TP work fanout, so follower ranks did not receive the leader payload.

The sparse MoE block mixed a globally reduced shared-expert result with rank-local routed-expert partials. For LLaDA2-Uni, both paths must be combined and reduced in FP32; lower-precision accumulation/reduction caused an accuracy regression in TP validation.

The model factory also forced `disable_cuda_graph=True`, preventing reuse of the generic SGLang decode CUDA graph path. Removing that model-level opt-out is expected to reduce repeated decode launch overhead, but this Draft claims no measured speedup.

This work addresses the TP and related thinker-execution items in #445. It does not add EP.

## Modifications

- Accept and validate pipeline-injected TP rank, size, GPU ID, and NCCL rendezvous settings.
- Make stage topology authoritative over an optional conflicting `tp_size` server override.
- Forward rank and rendezvous data through the existing SGLang bootstrap.
- Expose TP rank/size and `requires_tp_work_fanout` on `DllmScheduler`; fan out only for TP greater than 1.
- Keep routed and shared MoE outputs rank-local, combine them in FP32, perform one TP all-reduce, and cast back to the routed output dtype. TP=1 avoids the collective.
- Stop forcing `disable_cuda_graph=True` and reuse the existing server-args builder, bootstrap, and decode graph runtime.
- Preserve explicit eager fallback with `server_args_overrides={"disable_cuda_graph": True}`. The generic eager-prefill default is unchanged.
- Add focused tests for TP propagation, fanout, inner-reduction disabling, collective count and dtype, default CUDA graph behavior, and explicit opt-out.

## Scope and reuse

The TP factory plumbing, scheduler fanout contract, explicit CUDA graph override, and generic bootstrap are reusable by other model stages. The sparse-MoE reduction order and mandatory FP32 combine/reduce are LLaDA2-Uni correctness requirements; other models must validate their own numerical and sharding semantics.

Out of scope:

- expert parallelism
- image API, CFG, image decoder, or image-generation scheduler wiring
- production throughput and latency tuning

## Planned split

After maintainer feedback, this umbrella Draft will be replaced by two independently reviewable Ready PRs:

1. **Thinker TP correctness:** runtime propagation, scheduler fanout, MoE collective semantics, and TP=1/2/4 T2T/I2T parity.
2. **Model-level CUDA graph:** default decode graph enablement, explicit eager opt-out, eager/graph parity, memory, warm latency, and TP validation.

## Related Issues

Part of #445. This PR must not close the full LLaDA2-Uni support issue.

## Accuracy Test

### Local verification

- Focused CUDA graph and thinker TP tests: **3 passed, 4 skipped**
- Pipeline topology and placement regression tests: **72 passed, 2 deselected**
- Ruff check and format check: **passed**
- Python `compileall`: **passed**
- `git diff --check`: **passed**

Real-model TP parity, GPU execution, eager/CUDA-graph parity, and TP=2/4 validation have not been run for this Draft.

## Benchmark & Profiling

No performance result is claimed. Before the split PRs become Ready, validation will report TP=1/2/4 parity, latency, throughput, memory, collective overhead, eager/graph parity, graph capture overhead, and explicit eager fallback.

## Contributors

- @kunchengit
- @btw616
- @LiRongchuan

## Checklist

- [x] Format and static checks completed.
- [x] Focused unit tests added.
- [x] Model/runtime boundary and explicit opt-out documented.
- [ ] Real-model TP accuracy and GPU measurements pending.
- [ ] Eager/CUDA graph parity and benchmarks pending.
- [ ] For reviewers: If you have not contributed and are only assisting with merging main, please remove yourself as a co-author when merging.

## CI

This PR is intentionally a Draft. Self-hosted GPU CI and real-model validation will be requested after initial scope and design feedback.