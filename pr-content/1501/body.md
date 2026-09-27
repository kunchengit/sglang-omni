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

## Contributors

- @kunchengit
- @Anmuliar
