## Motivation

Distribute LLaDA2-Uni image decoding across GPUs with SGLang Diffusion sequence parallelism. Preserve global image/conditioning order while sharding transformer work, and provide the generic stage-SP lifecycle required by that integration.

## Modifications

- Include generic stage-SP topology, placement, process launch, rank metadata, and lifecycle support.
- Add the LLaDA2-Uni decoder policy for `sp_size`, `ulysses_degree`, and `ring_degree`, requiring `sp_size = ulysses_degree * ring_degree`.
- Use the SGLang Z-Image backend for multi-rank decoding. Diffusers remains the default for SP1.
- Keep SigVQ conditioning and VAE decoding on the leader; synchronize request settings, conditioning, and seeds across ranks without emitting duplicate images.
- Distinguish request failures synchronized across all ranks from unsynchronized TP/SP compute failures. Make the latter fatal to the scheduler so a failed worker cannot execute a later request and mismatch its collectives.
- Propagate decoder preparation and worker failures through the stage/coordinator lifecycle; this is lifecycle hardening, not an NCCL fault-recovery protocol.
- Shard padded noise-refiner and joint image-plus-conditioning sequences with matching position metadata, then gather in canonical global order, including non-square layouts.

The token-order handling stays in SGLang-Omni and reuses SGLang's attention and distributed primitives; it does not require editing the installed SGLang package.

Unsynchronized failures stop further work on the failed scheduler. Reporting to the coordinator can wait for distributed-backend timeout and cleanup; this does not provide automatic rank recovery or retract results already delivered by another rank.

## Usage and execution scope

Set `image_decode.sp_size` and an equally sized list of distinct GPU IDs in `image_decode.gpu`; for example, SP2 can use `gpu=[1, 2]`. Put `backend`, `ulysses_degree`, `ring_degree`, and `attention_backend` under `image_decode.factory`. Multi-rank decoding requires `backend="sglang"`.

- Two-rank Ulysses: `sp_size=2`, `ulysses_degree=2`, `ring_degree=1`.
- Two-rank Ring: `sp_size=2`, `ulysses_degree=1`, `ring_degree=2`; Ring requires `attention_backend="fa"` or `"sage_attn"`.

Normal and `decoder-turbo` decoding reuse the inherited backend controls. SP partitions a transformer's sequence, not separate requests or independent images. The split follows the padded global image-plus-conditioning sequence; attention collectives provide cross-rank token information.

The image APIs, thinker, prompts, and response formats are inherited. This is decoder SP, not thinker TP, CFG parallelism, or dynamic batching.

## Dependencies

Depends directly on #1502, inheriting #1500, #1499, and #2257. #1486 is a sibling thinker-TP branch, not a prerequisite.

This PR contains the generic stage-SP work previously proposed in #1490. The older PR is superseded and is not an additional dependency; review both the generic runtime delta and the model-specific decoder integration here.

## Related Issues

Tracked in #2207; continues the work in #445.

## Validation

Candidate revision: `b29eb2fd53fb144dbf1aaaff75af7dce2070b49d`, based on `main` at `7dc8909e` and #1502. The related regression suite and nine image API cases below ran on `77b76e98`, before a realtime-transcription-only rebase; the contribution, model runtime, image APIs, and dependencies are unchanged. The additional FlashAttention, fault-injection, and fixed-input experiments predate that regression run; their exercised code and dependencies are unchanged.

- Related unit/regression suites: **1458 passed, 5 skipped**. Two engine-contract checks do not apply to configurations without an SGLang engine; three Ring cases require a different attention backend. Applicable formatting and static checks passed.
- After the final rebase, shared realtime/transcription, image/OpenAI API, client, and session-lifecycle regressions on `b29eb2fd`: **591 passed, 0 skipped**. This checks the public import and request interfaces affected by the intervening upstream change, without rerunning full-checkpoint inference.
- Small-checkpoint GPU comparisons exercised SP1 and Ulysses2 with `torch_sdpa`, plus SP1/Ulysses2/Ring2 with FlashAttention. All nine FlashAttention cases completed their original assertions: eight on the initial run, and one on an unchanged retry after a rendezvous-port collision. The tests compare against the Diffusers FP32 reference; Ulysses also checks exact equality with native SP1 for the tested inputs.
- Four real production-entrypoint TP/SP fault scenarios passed with CUDA/NCCL and Gloo. Unknown compute failures prevented the queued successor from entering compute; errors agreed by all ranks allowed the successor to complete. Unknown-failure reporting waited for backend cleanup. The probe used a 15-second process-group timeout, not the decoder's 180-second default.
- Full-checkpoint SP1 and Ulysses2 deployments passed **9 request cases** covering normal/Turbo T2I, raw-image editing, context-budget rejection, and an SP1 thinking-mode request with independently checked prompt-token accounting. Normal/edit PNG pairs were byte-identical across the two deployments; Turbo outputs reproduced their respective earlier SP1/SP2 results.
- A prior Turbo check repeated the same request twice per deployment: each pair was byte-identical, but SP1 versus SP2 differed in RGB pixels (0–255 scale: MAE 1.626, RMSE 2.392, maximum absolute difference 29; PSNR 40.55 dB). A fixed-input follow-up matched conditioning, all noise draws, and sequence order. The first difference occurred in the first noise-refiner block's FFN output projection: single-GPU BF16 matrix multiplication at the full and partitioned sequence lengths reproduced the captured projection outputs without communication. This explains the first difference in that sample, not full-checkpoint perceptual equivalence.
- Environment: Linux, NVIDIA H20-3e, SGLang 0.5.20, Diffusers 0.37.0, PyTorch 2.13.0+cu130, Transformers 5.12.1, and Triton 3.7.1. Full-checkpoint tests used the SGLang decoder with `torch_sdpa`, a separate TP1 eager thinker, seed 42, 16 dLLM steps and 5 decoder steps.

These are functional and bounded numerical checks, not a full-checkpoint quality or performance benchmark. Ring full-checkpoint inference, SP4, and general speedup remain unvalidated.

## Full-checkpoint accuracy (2026-09-28)

At `b29eb2fd`, Ulysses SP2 completed all **553/553** GenEval requests, scoring **0.88993** across the six tasks and **494/553 (89.33%)** across individual examples. All 553 request payloads and decoded RGB hashes match the current #1499 SP1 reference at `36059f66` exactly.

Both runs used a BF16 LLaDA2.0-Uni checkpoint, TP1 eager thinker, SGLang decoder with `torch_sdpa`, CFG 4.0, CFG rescale 0.7, seed 42, 32 dLLM steps, 8 decoder-turbo steps, and 1024x1024 output.

Full original-image ImgEdit also completed **737/737** requests with all request payloads and decoded RGB hashes identical to SP1. The official local judge scored SP2 **3.50814** versus SP1 **3.52307**, with zero parsing or inference errors. Since generated pixels are identical, this difference is judging variation rather than an SP generation change. Edit requests used text CFG 4.0, image CFG 0.0, CFG rescale 0.7, seed 42, 8 dLLM steps and 8 decoder-turbo steps. No precomputed `.pt` inputs were used.

These results establish exact image agreement for the tested benchmarks/configurations, without extending the claim to other shapes, attention backends, Ring, or dtypes.

## Warm performance (2026-09-28)

Same revision `b29eb2fd`, H20-3e, BF16, TP1 eager thinker on GPU0, and SGLang decoder with `torch_sdpa` on GPU1 (SP1) or GPU1/2 (Ulysses SP2). Cases ran sequentially on reserved GPUs, with 3 warmups and 7 measured requests per workload and measurement mode. Values are medians in seconds.

| Workload | Decoder SP1 | Decoder SP2 | Decoder speedup | HTTP E2E SP1 | HTTP E2E SP2 | E2E speedup |
| --- | --- | --- | --- | --- | --- | --- |
| T2I | 8.985 | 5.118 | 1.76x | 39.223 | 35.188 | 1.11x |
| Edit | 8.984 | 5.119 | 1.75x | 18.627 | 14.684 | 1.27x |

Both workloads use seed 42, CFG rescale 0.7, 32 dLLM steps and 8 decoder-turbo steps. T2I uses CFG 4.0 at 1024x1024. Original-image edit uses text CFG 4.0 and image CFG 1.5, with output dimensions derived from the source. The edit performance workload differs from the accuracy benchmark's 8 dLLM steps and image CFG 0.0.

HTTP E2E is measured without request profiling and excludes client image saving. Decoder wall time includes PNG encoding and comes from a separate native stage-event run without Torch profiler; it is not GPU-kernel-only time. Stage-event E2E differed from the control by less than 0.73%. Thinker medians were 30.500/30.201 seconds for T2I and 9.543/9.428 seconds for edit (SP1/SP2), limiting E2E speedup. These eager-thinker E2E results must not be compared directly with #1486's graph-enabled thinker timings.

All 20 outputs per workload/deployment, including warmups and both measurement modes, had the same RGB hash; SP1 and SP2 also matched each other. These measurements cover Ulysses SP2 only, not Ring or combined thinker TP2 plus decoder SP2.

## Contributors

- @kunchengit
- @Anmuliar
