## Motivation

Enable tensor-parallel LLaDA2-Uni thinker execution for understanding and image generation. This also addresses two sources of overhead in the replayed implementation: per-process GPU visibility prevented custom all-reduce initialization, and the model duplicated expert routing already provided by SGLang.

## Modifications

- Propagate stage TP rank, size, GPU placement, and rendezvous settings into the thinker worker; fan out dLLM work to participating ranks.
- Retain rank-local shared and routed expert outputs, combine them in FP32, perform one TP all-reduce, and cast back to the model dtype.
- Reuse SGLang's `TopK` routing component while preserving the checkpoint's sigmoid scores, correction bias, normalization, and routing scale. No new Triton kernel is introduced.
- Keep the stage's GPU set visible to each LLaDA2-Uni rank so SGLang can initialize custom all-reduce; preserve correct rank-to-device mapping.
- Enable CFG-aware CUDA Graph execution through an Omni model-runner hook selected for `LowConfidenceCFG`, without replacing the graph runner for unrelated models.
- Use CPU padding metadata to decide graph eligibility. A dLLM block with padding inside its active query runs eagerly; eligible blocks can replay once padding is entirely in the cached prefix, with padded cached positions excluded from attention.
- Preserve `server_args_overrides={"disable_cuda_graph": True}` and fix inherited interleaved stage configuration so its decoder remains nonterminal.

CUDA Graph support is part of this TP execution change. It does not imply that every prefill or every dLLM forward uses a graph, and it is not a separate planned PR.

## Scope and dependencies

The current branch is stacked on #1502 (`llada2/interleaved-image-generation`). Its comparison against `main` therefore includes the thinker correctness, native image, thinking, and interleaved prerequisites. Review the TP change relative to that branch; rebase after those prerequisites land.

This PR changes thinker execution only. Decoder SP is tracked in #1501 and is a sibling branch, not a dependency. Expert parallelism, quantization, and request batching are out of scope.

## Roadmap

This PR is re-submitted under the new [LLaDA-Uni roadmap (#2207)](https://github.com/sgl-project/sglang-omni/issues/2207), carrying forward the earlier work tracked in #445 with a rebased implementation and updated scope. It covers **Phase 2: LLaDA-Uni thinker tensor parallelism**, including the associated CFG CUDA Graph execution changes described above. It does not close the full roadmap.

## Accuracy Test

The latest stack synchronization passed the relevant LLaDA2-Uni and image API unit checks. Recorded GPU validation of the routing/custom-all-reduce changes also covered TP2 serving and cross-rank output-token agreement.

The paired routing study used the same TP2 setup for both arms, changing the router implementation:

| Evaluation subset | Before routing replacement | SGLang TopK |
| --- | ---: | ---: |
| MMMU, 60 samples, VLMEvalKit API judge | 53.33% | 51.67% |
| ImgEdit, 216 samples, official judge | 3.5370 | 3.5748 |
| GenEval, 60 samples | 86.67% | 86.67% |

These are historical subset results, not full-benchmark scores or proof of noninferiority. The ImgEdit study used precomputed source tokens through the former evaluation path; that public input has since been removed. It does not validate the current raw-image preprocessing path. The benchmark was not rerun after the latest preprocessing/branch synchronization.

TP1 and TP2 are not required to produce identical pixels: routing order and floating-point reduction order can change. Quality comparisons remain necessary.

## Benchmark & Profiling

Recorded before the latest stack synchronization: H20-3e, BF16, Torch 2.13.0+cu130, SGLang 0.5.19, CUDA Graph enabled, 32-token dLLM blocks, 32 thinker steps, text CFG 4, seed 42, and 1024x1024 output. The SGLang decoder remained SP1 with 8 turbo steps; edit additionally used image CFG 1.5.

Each row is the median of five unprofiled HTTP requests after three warmup requests, using SGLang TopK in both TP configurations.

| Task | TP1 thinker | TP2 thinker | TP1 request E2E | TP2 request E2E |
| --- | ---: | ---: | ---: | ---: |
| T2I | 6.550 s | 5.674 s | 15.521 s | 14.666 s |
| Edit | 3.863 s | 3.487 s | 12.868 s | 12.495 s |

Thinker time includes stage/scheduler overhead. TP2 reduced it by 13.4% for T2I and 9.7% for edit in this workload; a short prefill did not benefit. A separate attribution trace confirmed custom all-reduce instead of NCCL and reduced routing-kernel overhead. These measurements are not a fresh benchmark of the current branch head.

## Contributors

- @kunchengit
- @btw616
- @LiRongchuan

## Remaining validation

- Rerun GPU accuracy and warm performance on the final rebased head before making current-head performance claims.
- TP4 and broader workload coverage are not claimed here.
