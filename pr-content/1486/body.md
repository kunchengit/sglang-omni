## Motivation

Enable tensor-parallel LLaDA2-Uni thinker execution for understanding and image generation with FP32 expert-output reduction. Add opt-in CFG-aware CUDA graph execution to reduce eligible repeated-forward launch overhead. Keep reference Torch routing by default, with native SGLang TopK available as an explicit performance option; the measured numerical and accuracy tradeoffs are below.

## Modifications

- Propagate stage TP rank, size, GPU placement, and rendezvous settings into thinker workers and fan out dLLM work to participating ranks.
- Combine rank-local shared and routed expert outputs in FP32, perform one TP all-reduce, and cast back to the output dtype.
- Default to #2257's FP32 Torch routing. Optionally reuse SGLang's `TopK` with sigmoid scoring, expert correction bias, normalization, and routing scale. Its fused sigmoid approximation is not numerically identical to the reference implementation.
- Keep the participating GPU set visible to each thinker rank so SGLang can initialize custom all-reduce with the correct rank-to-device mapping.
- Select the CFG-aware model-runner hook for `LowConfidenceCFG` without replacing the graph path for unrelated models.
- Use CPU padding metadata to decide graph eligibility: active-query padding runs eagerly; eligible blocks can replay after padding is entirely in the cached prefix.
- Include optional H20 TP2 MoE configurations for existing SGLang kernels. These are parameter configurations, not new kernel implementations.

## Usage and execution scope

Routing is selected at model startup for both TP1 and TP2. The default is `torch`, preserving #2257's routing arithmetic. Opt into SGLang TopK with:

```bash
--thinker.engine.json_model_override_args '{"llada2_uni_topk_backend":"sglang"}'
```

Use `"torch"` to explicitly select the default. Combine this field with any other model overrides in the same JSON object. Both routes retain the same expert implementation and FP32 TP reduction; the option works with eager execution and CUDA graphs and requires no checkpoint or SGLang source changes.

The thinker remains eager by default. With SGLang 0.5.20, use `--thinker.engine.cuda_graph_backend_decode full` to opt into full decode graphs; prefill remains eager. Use `--thinker.engine.cuda_graph_backend_decode disabled` to explicitly select eager decode. Phase-specific or JSON graph settings take precedence over the compatibility `disable_cuda_graph` flag.

The shared server-argument default enables `torch.compile` for selected graph capture shapes. Use `--thinker.engine.enable_torch_compile false` to run graphs without compilation. The compile setting does not by itself enable CUDA graphs, and compiled and non-compiled outputs are not guaranteed to be byte-identical.

TP configuration supplies the stage's rank count and GPU list and requires a dedicated thinker process. For example, a two-rank deployment uses `--thinker.process thinker --thinker.tp_size 2 --thinker.gpu '[0, 1]'`. Enabling graphs does not guarantee that every prefill or dLLM block is replayable; ineligible padded layouts still use eager execution.

Optional tuning is enabled separately with `SGLANG_MOE_CONFIG_DIR` set to the absolute path of `examples/tuning/llada2_uni/h20_tp2`, not its `configs/` subdirectory. SGLang appends the versioned configuration subdirectory. The supplied configurations target BF16 LLaDA2.0-Uni on H20-3e, TP2, SGLang 0.5.20, and Triton 3.7.1. They replace the MoE configuration search root, so omit the setting for other models, TP sizes, precision modes, or hardware/software combinations without their own validation.

## Dependencies

Depends directly on #1502, inheriting the thinker correctness and image-generation feature stack. Review the incremental TP, routing, and graph changes relative to that PR.

#1501 is a sibling decoder-SP branch, not a dependency. The shared TP plumbing can support other model stages, but LLaDA2-Uni's reduction order and CFG attention semantics require model-specific validation. Expert parallelism, quantization, and request batching are outside this PR.

## Related Issues

Tracked in #2207; continues the work in #445.

## Validation

Current revision: `c1b8b039`, based on `main` at `7dc8909e` and #1502. It adds the startup routing selector to `b142ecd0`. The older regression suite and full-checkpoint matrix below ran on `7905d462`, before the realtime-transcription-only rebase and routing-selector change, using native SGLang routing.

For the routing-selector change, all 9 focused thinker precision tests and applicable repository formatting/static checks passed. Two real TP2 deployments with decode graphs enabled and compilation disabled each completed one T2I request, one original-image edit and one fixed MMMU request. The default Torch path reproduced the earlier `d88f7c2e` reference run's image RGB hashes and answer hash; the explicit `sglang` CLI override reproduced the `b142ecd0` native run's hashes. All six matched their respective prior runs exactly. This was a startup and output-regression check, not a repeat of full-dataset scoring or latency measurement.

- Related unit/regression suites: **1388 passed, 2 skipped**. The skips are engine-contract checks in configurations without an SGLang engine. Applicable formatting and static checks passed.
- Full-checkpoint matrix on H20-3e: TP1 and TP2, each with eager execution, decode graphs without compilation, and decode graphs with the default compilation setting. All **18 requests** passed: two-way CFG T2I, three-way CFG raw-image editing, and a long-prompt T2I case per deployment. Requests used seed 42, 16 dLLM steps and 5 decoder steps, with the Diffusers decoder on a separate GPU.
- All four graph deployments produced real CUDA graph capture/replay evidence, including both TP2 ranks; compile-enabled capture callables were observed in the compiled deployments. This does not establish fullgraph compilation coverage. Active-query padding rejected graph execution; replay resumed once the padding was entirely in the cached prefix. Requests completed without leftover pending work.
- Within each TP size, eager and non-compiled graph PNGs were byte-identical for all three requests. Compile-enabled graph PNGs differed from their non-compiled counterparts in all six comparisons. Compilation changes execution and numerical paths; the matrix does not establish cross-mode image-quality equivalence.
- PNGs also differed between TP1 and TP2. A prior fixed-input eager follow-up located the first difference in layer 0's BF16 row-parallel attention output projection; its QKV, QK normalization, RoPE, and attention outputs matched exactly. Single-GPU full and input-feature-sharded matrix multiplications, followed by summation of the shard outputs, reproduced the captured TP1/TP2 projections. This explains the first difference in that sample, not the separate compilation difference or full-model quality equivalence.
- Environment: Linux, NVIDIA H20-3e, SGLang 0.5.20, Diffusers 0.37.0, PyTorch 2.13.0+cu130, Transformers 5.12.1, and Triton 3.7.1.

The matrix did not enable the optional MoE tuning overrides. These are functional and sample-consistency checks, not quality or performance benchmarks. No general speedup, TP4 result, or expert-parallel validation is claimed.

## Current accuracy (2026-09-28)

At `b142ecd0`, TP2 with CUDA graphs and native `TopK` routing completed all 1,050 MMMU examples. The API judge scored **494/1,050 (47.0476%)**, including **426/900 (47.3333%)** on the validation split; its single fallback judgment was incorrect.

The current #2257 reference at `2bddd0fa` uses TP1 eager execution and scored 516/1,050, including 437/900 on validation. Its answers are byte-identical to the historical post-CFG-correctness source before graph removal. The comparison changes TP size, graph execution, and routing implementation together, so the score difference cannot be attributed to TP alone.

A full control on this same `b142ecd0` revision retained native SGLang `TopK`, enabled decode graphs and disabled `torch.compile` for both TP sizes:

| Thinker | All questions | Validation split | Random fallback |
| --- | --- | --- | --- |
| TP1 | 500/1,050 (47.6190%) | 427/900 (47.4444%) | 2, including 1 correct |
| TP2 | 494/1,050 (47.0476%) | 426/900 (47.3333%) | 1, incorrect |

The VLMEvalKit API judge used `doubao-seed-2-0-lite-260428`. Relative to TP1, TP2 gained 121 questions and lost 127 (paired exact test p=0.751). Excluding the three questions with a fallback in either run gives 499 versus 494 correct. Only ten answer texts were identical. This control does not show a statistically significant accuracy decrease, but it does not establish numerical or per-question equivalence.

Two captured eager forwards checked all 19 MoE layers: both TP2 ranks agreed, and merged outputs exactly matched sums of captured rank-local routed/shared outputs. Changing sum grouping did not change these outputs. With identical first-MoE input, native TopK IDs/weights and reconstructed down-projection activation shards matched exactly; sampled expert weight shards also matched. High-precision reconstruction localized the remaining sampled down-projection difference to separately rounded BF16 partial products. These bounded diagnostics found no expert merge-order defect; they do not attribute the aggregate MMMU difference to a single operator.

### Full MMMU routing ablation

Commit `cf201b3a` replaces #2257's Torch routing sequence with SGLang `TopK` even at TP1. To isolate this change, experimental revision `d88f7c2e` restores only the reference routing on top of `b142ecd0`, retaining this PR's TP implementation and FP32 expert-output reduction. Both routing paths use the same checkpoint and all 1,050 requests, BF16, `torch_sdpa`, decode graphs enabled, `torch.compile` disabled, temperature 0 and `max_tokens=2048`. Scoring uses the same VLMEvalKit API judge named above.

| Routing path | Thinker | All questions | Validation split | Random fallback |
| --- | --- | --- | --- | --- |
| Native SGLang TopK | TP1 | 500/1,050 (47.6190%) | 427/900 (47.4444%) | 2, including 1 correct |
| Reference Torch routing from #2257 | TP1 | 517/1,050 (49.2381%) | 438/900 (48.6667%) | 2, both incorrect |
| Native SGLang TopK | TP2 | 494/1,050 (47.0476%) | 426/900 (47.3333%) | 1, incorrect |
| Reference Torch routing from #2257 | TP2 | 517/1,050 (49.2381%) | 452/900 (50.2222%) | 2, including 1 correct |

**Native SGLang TopK scored lower in this controlled run:** 17 fewer correct answers at TP1 (1.6190 percentage points) and 23 fewer at TP2 (2.1905 percentage points). This is an observed accuracy tradeoff of the routing replacement, not evidence that tensor parallelism alone caused the original cross-PR score gap.

With reference routing, TP1 reproduced **all 1,050 answer texts from #2257 exactly**. Its score of 517 versus #2257's 516 comes from 13 changed API judgments (7 gains and 6 losses), not changed model answers. This full-dataset control establishes that restoring the routing removes the observed A-to-F TP1 generation difference.

A matched eager trace first diverged in MoE routing after identical inputs, attention outputs and router logits. On identical captured inputs, all 608 token/layer rows selected the same expert sets, but ordering and weights differed. The installed FlashInfer 0.6.17 routing kernel uses a fast `tanh`-based sigmoid approximation; the first captured layer's expert-aligned maximum weight difference was 3.90e-6. Restoring reference routing reproduced the captured logits exactly. These observations identify a numerical source of the output divergence, not an expert merge-order bug.

API scoring and TP outputs remain variable: reference TP2 gained 140 questions and lost 117 against native TP2 (paired exact test p=0.170). Excluding the two questions with fallback judgments in either TP2 run gives 516 versus 494 correct. The TP1 routing comparison has p=0.314. Neither score improvement reaches the conventional 0.05 significance threshold in this single run. Reference TP1 and TP2 both score 517, but only 15 complete answer texts match; equal scores do not establish numerical equivalence.

These full-dataset results were measured on the separate revisions above, before the startup selector was added. This PR now includes the same reference-routing arithmetic as the default and retains native SGLang TopK as an opt-in. The routing performance comparison below reports the measured latency benefit alongside the accuracy tradeoff; it is not a new full-dataset evaluation of the selector commit.

On a pre-existing fixed 60-question subset, this candidate's TP1 eager and graph runs produced identical answer texts. Their API judgments scored 32/60 and 31/60 respectively; the one-point difference is judge variability. The same subset in the full runs scored 32/60 for #2257 and 33/60 for TP2. This bounded check found no graph-induced answer change; it does not establish full-dataset TP equivalence or explain the full-score difference.

The candidate's native-image GenEval run achieved a **task-macro score of 0.88023**, with **487/553 individual cases correct (88.07%)**. These are distinct aggregation measures. The current #1499 reference at `36059f66`, TP1 eager, scored **0.88993** and **494/553** on the same 553 request payloads. This comparison changes routing and graph execution as well as TP size, so the 0.00970 task-macro difference is not isolated to TP. Decoder-SP (#1501) reproduced all 553 reference images exactly.

Full ImgEdit evaluation on the same revision completed **737/737** original-image requests. The official local ImgEdit judge reported a sample-mean score of **3.54093**, with zero parsing or inference errors. Requests used the native image-edit API, text CFG 4.0, image CFG 0.0, CFG rescale 0.7, seed 42, 8 dLLM steps, and 8 decoder-turbo steps. The thinker used TP2 with CUDA graphs enabled and `torch.compile` disabled; the SGLang image decoder used SP1 and `torch_sdpa`. This evaluates original-image preprocessing, not precomputed `.pt` inputs. The current #1499 TP1 reference scored **3.52307** on all 737 examples with the same request settings. There is no aggregate edit-score decrease in this comparison; this is not a claim of per-image or MMMU equivalence.

## Warm performance (2026-09-28)

Same revision `b142ecd0`, H20-3e, BF16, native SGLang TopK, decode graphs enabled and `torch.compile` disabled. TP1 uses GPU0, TP2 uses GPU0/1; both place the SGLang SP1 decoder on GPU2 with `torch_sdpa`. Cases run sequentially on reserved GPUs, with 3 warmups and 7 measured requests per workload and measurement mode. Values below are medians in seconds.

| Workload | Thinker TP1 | Thinker TP2 | Thinker speedup | HTTP E2E TP1 | HTTP E2E TP2 | E2E speedup |
| --- | --- | --- | --- | --- | --- | --- |
| T2I | 6.529 | 5.097 | 1.28x | 15.525 | 14.086 | 1.10x |
| Edit | 3.105 | 2.704 | 1.15x | 12.197 | 11.721 | 1.04x |

Both workloads use seed 42, CFG rescale 0.7, 32 dLLM steps and 8 decoder-turbo steps. T2I uses CFG 4.0 at 1024x1024. Original-image edit uses text CFG 4.0 and image CFG 1.5, with dimensions derived from the source. This edit performance configuration differs from the accuracy benchmark's 8 dLLM steps and image CFG 0.0.

HTTP E2E is measured without request profiling and excludes client image saving. Stage wall times are collected in a separate native stage-event run without Torch profiler; they are not GPU-kernel-only times, and decoder stage time includes PNG encoding. Enabling stage events changed median E2E by less than 0.13% across these cases. Decoder medians remained 8.940-8.985 seconds. Each deployment reproduced its image hashes across all repeats and both measurement modes; TP1 and TP2 images differ.

TP2 explicitly sets `SGLANG_MOE_CONFIG_DIR` to the repository's `examples/tuning/llada2_uni/h20_tp2` directory. Logs confirm loading its up/down configurations, but the messages do not identify ranks individually. TP1 uses the installed configuration root, which is not asserted to be untuned. This measures the deployed TP configurations, not the isolated benefit of tuning overrides or a universal TP2 speedup.

### Native versus reference routing

The completed routing comparison uses production `b142ecd0` and routing-only experiment `d88f7c2e`, with the same checkpoint, BF16, `torch_sdpa`, decode graphs enabled and compilation disabled. Each deployment runs one fixed T2I prompt, one original-image edit and one MMMU question (item 405). Each workload has 3 warmups and 7 measured requests in separate unprofiled HTTP and native stage-event passes. This is a short latency experiment, not another full benchmark. Values are medians in seconds; speedup is reference HTTP latency divided by native HTTP latency.

| Thinker | Workload | Native HTTP E2E | Reference HTTP E2E | Native Thinker | Reference Thinker | Native E2E speedup |
| --- | --- | --- | --- | --- | --- | --- |
| TP1 | T2I | 15.539 | 16.306 | 6.527 | 7.320 | 1.049x |
| TP1 | Edit | 12.178 | 12.503 | 3.110 | 3.466 | 1.027x |
| TP1 | MMMU item 405 | 3.531 | 5.454 | 3.495 | 5.424 | 1.544x |
| TP2 | T2I | 14.070 | 14.918 | 5.105 | 5.976 | 1.060x |
| TP2 | Edit | 11.715 | 12.052 | 2.708 | 3.039 | 1.029x |
| TP2 | MMMU item 405 | 3.322 | 4.062 | 3.292 | 4.022 | 1.223x |

At TP2, native routing reduces measured T2I E2E by 0.847 seconds and edit E2E by 0.337 seconds. The corresponding Thinker speedups are 1.170x and 1.122x. The decoder remains SP1 on a separate GPU, limiting the impact of faster routing on image-request E2E. These measurements accompany the observed full-MMMU scores of 494/1,050 with native routing and 517/1,050 with reference routing; they do not establish a universal speed/accuracy tradeoff across datasets or hardware.

TP1 deployments partially overlap on disjoint physical GPU pairs (native 0/2, reference 1/3), sharing host CPU and memory bandwidth. TP2 deployments run sequentially on GPUs 0/1 for the thinker and GPU 2 for the decoder; both explicitly load their identical repository H20 TP2 MoE tuning configurations. This is not a fully isolated same-device TP1 routing comparison. Native stage-event collection changed median HTTP E2E by less than 0.3% in all cases. Stage timings remain wall times, not GPU-kernel-only measurements.

MMMU uses the same request with 533 prompt tokens and a 2,048-token limit, but generation differs: TP1 native/reference report 2,048/1,572 completion tokens and 8,098/4,135 answer characters; TP2 report 2,048/2,048 tokens and 7,290/5,759 characters. Each deployment's answer hash is stable across repeats. Even equal completion lengths do not guarantee equal dLLM forward work, so the MMMU ratios describe actual request latency, not fixed-workload router throughput. Image request settings are those in the warm-performance section above.

## Contributors

- @kunchengit
- @btw616
- @LiRongchuan
