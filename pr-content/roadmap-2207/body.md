## Goal

Complete LLaDA2.0-Uni's core generation capabilities in SGLang-Omni before expanding inference and scheduling optimizations. Shared runtime components should be reusable where their contracts apply, while prompts, CFG semantics, numerical behavior, and model adapters remain explicitly model-specific.

This roadmap continues the work tracked in #445. LLaDA2.0-Uni is the current implementation target; future members of the LLaDA-Uni family require their own model integration.

## Phase 1: Core generation capabilities

- [ ] **A — Thinker CFG and numerical correctness:** #2257. Multi-way CFG, padding-aware attention, shared dLLM admission/cleanup, MoE numerics, and understanding preprocessing.
- [ ] **B — Native image generation and editing:** #1499. Non-thinking T2I and raw-image editing, native image endpoints, chat image output, and Diffusers/default or optional SGLang decoder backends.
- [ ] **C — Thinking-mode image generation:** #1500. Two-pass T2I with a 2048-token thinking budget, an image-generation transition, and two-pass token accounting.
- [ ] **E — Interleaved generation:** #1502. Ordered multi-frame text/image output, generated frame dimensions, and the shared self-route/multi-inflight relay needed by the pipeline.

The image contracts are intentionally distinct: native image endpoints return `data[]`; single-image chat generation returns `message.image`; interleaved chat generation returns ordered `message.segments` with inline PNG media and a concatenated plain-text `message.content`. Normal and `decoder-turbo` decoding are supported by the image stack.

## Phase 2: Inference optimization

- [ ] **F — Thinker TP and opt-in CFG CUDA graphs:** #1486. TP execution, FP32 expert combine/reduce, SGLang routing reuse, and eligible graph replay. Eager execution remains the default. Optional H20 TP2 settings configure existing MoE kernels and are hardware/version-specific.
- [ ] **H — Image-decoder SP and generic stage-SP runtime:** #1501. SGLang-backed Ulysses/Ring decoder integration, global token ordering, and leader/follower lifecycle.
- [ ] **Planned:** thinker and image-decoder quantization.

These optimizations are not prerequisites for merging basic T2I/edit support.

## Phase 3: Scheduling and serving optimization

The following are planned work, not capabilities established by the interleaved relay:

- [ ] Diffusion continuous batching.
- [ ] Dynamic batching in the LLaDA-Uni thinker.
- [ ] Cross-node and cross-stage scheduling.

## Dependencies and PR ownership

The implementation stack is:

```text
main
└── A #2257 — thinker correctness
    └── B #1499 — native T2I/edit
        └── C #1500 — thinking T2I
            └── E #1502 — interleaved generation + shared relay
                ├── F #1486 — thinker TP + opt-in CUDA graphs
                └── H #1501 — decoder SP + generic stage SP
```

These arrows describe the code stack, not a claim that every optimization conceptually requires every earlier feature. Review incremental changes against the immediate prerequisite and rebase the remaining stack as prerequisites merge. F and H are siblings; neither depends on the other.

The older shared-relay proposal #1487 is superseded by E/#1502. The older generic stage-SP proposal #1490 is superseded by H/#1501. They are no longer active work items or separate prerequisites, and their code should not be applied a second time.
