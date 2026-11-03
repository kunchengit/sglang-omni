## Motivation

Add tensor-parallel LLaDA2-Uni thinker execution and optional CUDA graphs for understanding and image generation.

## Modifications

- Pass TP rank, GPU placement, and rendezvous settings to thinker workers; dispatch dLLM work to all ranks.
- Combine shared and routed expert outputs in FP32, perform one TP all-reduce, then cast to the output dtype. Preserve GPU visibility for SGLang custom all-reduce.
- Add CFG-aware decode graphs. Prefill and blocks with active-query padding remain eager.
- Keep #2257's Torch routing as the default; expose native SGLang TopK as an opt-in.
- Provide optional H20 TP2 tuning configurations for existing SGLang MoE kernels.

## Usage

Add these options to the model launch command for TP2 with decode graphs and compilation disabled:

```bash
--thinker.process thinker \
--thinker.tp_size 2 \
--thinker.gpu '[0, 1]' \
--thinker.engine.cuda_graph_backend_decode full \
--thinker.engine.enable_torch_compile false
```

The thinker runs eagerly unless graphs are enabled. The cookbook examples and reported accuracy/performance runs explicitly disable `torch.compile`, including when CUDA Graphs are enabled.

Native SGLang routing is selected separately:

```bash
--thinker.engine.json_model_override_args '{"llada2_uni_topk_backend":"sglang"}'
```

The default retains the reference Torch routing from #2257. Native SGLang TopK is opt-in; see the [routing comparison](#routing-comparison) below for its numerical and accuracy implications.

For the supplied BF16 H20-3e TP2 tuning configurations:

```bash
export SGLANG_MOE_CONFIG_DIR=/absolute/path/to/sglang-omni/examples/tuning/llada2_uni/h20_tp2
```

Point above `configs/` and apply this setting only to TP2 on the validated hardware. The files were tuned with Triton 3.7.1; SGLang 0.5.22 / Triton 3.8.0 loads them through its version fallback. This is reuse of the shipped configuration, not a fresh Triton 3.8 tune. Confirm both `E=256,N=256` up/down configuration paths in the worker logs.

## Validation

- Rebased on main `137eaa97b` at `ee1b47041`, with all 11 TP patches unchanged. Remote focused tests: **781 passed, 47 skipped, 13 deselected**. Changed-file lint passed; all-files checks passed except Rust formatting, unavailable locally because `cargo` is absent. No Rust files changed.
- Both TP1 and TP2 completed native T2I and original-image edit requests on the rebased code. All outputs were 1024x1024 PNGs; each task's three measured repeats had identical RGB hashes within a TP setting. Visual inspection found coherent outputs, with composition/texture differences between TP1 and TP2. This is sample validation, not a new full-dataset accuracy score.
- Earlier full-dataset evaluation with native SGLang routing, TP2, graphs on, and compile off: GenEval macro score **0.88023** (487/553 correct); original-image ImgEdit **3.54093** (737/737 completed). These results predate the current main rebase and do not cover the default Torch route.

### Routing comparison

A routing-only comparison evaluated native SGLang TopK at `b142ecd0` against its reference-Torch variant `d88f7c2e`, retaining the same TP implementation and FP32 expert-output reduction. Both used the same checkpoint and 1,050 MMMU requests on H20-3e, BF16, `torch_sdpa`, decode graphs enabled, compilation disabled, temperature 0 and `max_tokens=2048`. Scoring used the same VLMEvalKit API judge (`doubao-seed-2-0-lite-260428`).

| Routing | Thinker | All questions | Validation split |
| --- | --- | --- | --- |
| Reference Torch | TP1 | 517/1,050 | 438/900 |
| Native SGLang TopK | TP1 | 500/1,050 | 427/900 |
| Reference Torch | TP2 | 517/1,050 | 452/900 |
| Native SGLang TopK | TP2 | 494/1,050 | 426/900 |

Reference Torch TP1 reproduced all answer texts from #2257; separate API judgments explain the score difference. Captured-input diagnostics found differences in expert ordering and weights despite identical router logits, so the implementations should not be assumed numerically interchangeable.

These are single-run API-judge results, including fallback judgments. Neither paired routing comparison reached the 0.05 significance threshold (TP1: p=0.314; TP2: p=0.170), and equal TP1/TP2 totals do not imply identical answers. They do not establish a universal accuracy regression from SGLang routing. Torch remains the default to preserve reference routing behavior; native TopK is an explicit performance option.

The full-dataset comparison predates the startup selector and is not a fresh evaluation of the current PR head.

## Performance

Measured on `ee1b47041` after the main rebase, using H20-3e, SGLang 0.5.22, Torch 2.14.1+cu130, Triton 3.8.0, BF16, **native SGLang TopK**, decode graphs enabled, and compilation disabled. TP1 uses GPU0; TP2 uses GPU0/1. The SGLang `torch_sdpa` decoder remains SP1 on GPU0 and executes after the thinker.

Sequential runs on exclusive GPUs with identical inputs and settings; 2 warmups and 3 measured requests per case. Values are median seconds.

| Task | Thinker TP1 | Thinker TP2 | Speedup | HTTP E2E TP1 | HTTP E2E TP2 | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| T2I | 6.663 | 5.351 | 1.25x | 15.641 | 14.349 | 1.09x |
| Edit | 3.186 | 2.852 | 1.12x | 12.214 | 11.883 | 1.03x |

Settings: seed 42, CFG rescale 0.7, 32 dLLM steps, and 8 decoder-turbo steps. T2I uses CFG 4.0 at 1024x1024; edit uses text CFG 4.0 and image CFG 1.5, with dimensions derived from the source.

Thinker timings cover dispatch through completion using request events without Torch profiler; HTTP timing covers the same requests and excludes client image saving. TP2 loads the repository's up/down MoE configurations through the Triton 3.7.1 fallback; TP1 uses the installed Triton 3.5.1 configuration fallback. These results measure the documented deployments, not the default Torch route, isolated kernel time, or the isolated benefit of tuning.

### Historical TP2 routing latency

An earlier routing comparison used matched image requests, 3 warmups and 7 repeats, with TP2, decode graphs enabled, compilation disabled, and H20 MoE tuning. Its SP1 decoder ran on a separate GPU; these are not measurements of the rebased head.

| Task | Reference Torch HTTP E2E (s) | Native SGLang TopK HTTP E2E (s) |
| --- | --- | --- |
| T2I | 14.918 | 14.070 |
| Edit | 12.052 | 11.715 |

This separate comparison measures routing-dependent end-to-end latency for these workloads, not router-kernel throughput or a general speed/quality guarantee.

## Dependencies

Based on main, which already includes the prerequisite image, thinking, and interleaved pipelines. Decoder SP (#1501) remains a separate PR. Expert parallelism, quantization, and request batching are outside this scope.

Tracked in #2207; continues #445.

## Contributors

- @kunchengit
- @btw616
- @LiRongchuan
- @Anmuliar
