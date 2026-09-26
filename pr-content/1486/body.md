## Motivation

Enable tensor-parallel LLaDA2-Uni thinker execution for understanding and image generation while preserving its MoE numerical behavior. Add opt-in CFG-aware CUDA graph execution to reduce eligible repeated-forward launch overhead.

## Modifications

- Propagate stage TP rank, size, GPU placement, and rendezvous settings into thinker workers and fan out dLLM work to participating ranks.
- Combine rank-local shared and routed expert outputs in FP32, perform one TP all-reduce, and cast back to the output dtype.
- Reuse SGLang's `TopK` routing while preserving sigmoid scores, expert correction bias, normalization, and routing scale.
- Keep the participating GPU set visible to each thinker rank so SGLang can initialize custom all-reduce with the correct rank-to-device mapping.
- Select the CFG-aware model-runner hook for `LowConfidenceCFG` without replacing the graph path for unrelated models.
- Use CPU padding metadata to decide graph eligibility: active-query padding runs eagerly; eligible blocks can replay after padding is entirely in the cached prefix.
- Include optional H20 TP2 MoE configurations for existing SGLang kernels. These are parameter configurations, not new kernel implementations.

## Usage and execution scope

The thinker remains eager by default. With SGLang 0.5.20, use `--thinker.engine.cuda_graph_backend_decode full` to opt into full decode graphs; prefill remains eager. Use `--thinker.engine.cuda_graph_backend_decode disabled` to explicitly select eager decode. Phase-specific or JSON graph settings take precedence over the compatibility `disable_cuda_graph` flag.

TP configuration supplies the stage's rank count and GPU list and requires a dedicated thinker process. For example, a two-rank deployment uses `--thinker.process thinker --thinker.tp_size 2 --thinker.gpu '[0, 1]'`. Enabling graphs does not guarantee that every prefill or dLLM block is replayable; ineligible padded layouts still use eager execution.

Optional tuning is enabled separately with `SGLANG_MOE_CONFIG_DIR` set to the absolute path of `examples/tuning/llada2_uni/h20_tp2`, not its `configs/` subdirectory. SGLang appends the versioned configuration subdirectory. The supplied configurations target BF16 LLaDA2.0-Uni on H20-3e, TP2, SGLang 0.5.20, and Triton 3.7.1. They replace the MoE configuration search root, so omit the setting for other models, TP sizes, precision modes, or hardware/software combinations without their own validation.

## Dependencies

Depends directly on #1502, inheriting the thinker correctness and image-generation feature stack. Review the incremental TP, routing, and graph changes relative to that PR.

#1501 is a sibling decoder-SP branch, not a dependency. The shared TP plumbing can support other model stages, but LLaDA2-Uni's reduction order and CFG attention semantics require model-specific validation. Expert parallelism, quantization, and request batching are outside this PR.

## Related Issues

Tracked in #2207; continues the work in #445.

## Validation

Regression-tested revision: `3f788395f9c97aae2280aa026dd58f1611933134`, based on `main` at `bddad43b` and #1502. Full-checkpoint results below were recorded on `e47537ef`, before the Qwen3-TTS-only upstream refresh; the relevant model, runtime, configuration, and dependency files are unchanged.

- Related unit/regression suites: **1379 passed, 2 skipped**. The skips are engine-contract checks in configurations without an SGLang engine. Applicable formatting and static checks passed.
- Full-checkpoint matrix on H20-3e: TP1 and TP2, each with eager execution and explicit decode graphs. All **12 requests** passed: two-way CFG T2I, three-way CFG raw-image editing, and a long-prompt T2I case per deployment. Requests used seed 42, 16 dLLM steps and 5 decoder steps, with the Diffusers decoder on a separate GPU.
- Both graph deployments produced real CUDA graph capture/replay evidence, including both TP2 ranks. Active-query padding rejected graph execution; replay resumed once the padding was entirely in the cached prefix. Requests completed without leftover pending work.
- Within each TP size, all three eager/graph PNG pairs were byte-identical. PNGs differed between TP1 and TP2; cross-TP numerical and perceptual equivalence has not been quantified.
- Environment: Linux, NVIDIA H20-3e, SGLang 0.5.20, Diffusers 0.37.0, PyTorch 2.13.0+cu130, Transformers 5.12.1, and Triton 3.7.1.

The matrix did not enable the optional MoE tuning overrides. These are functional and sample-consistency checks, not quality or performance benchmarks. No general speedup, TP4 result, or expert-parallel validation is claimed.

## Contributors

- @kunchengit
- @btw616
- @LiRongchuan
