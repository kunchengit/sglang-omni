## Motivation

Distribute LLaDA2-Uni image decoding across GPUs with SGLang Diffusion sequence parallelism. Preserve global image/conditioning order while sharding transformer work, and provide the generic stage-SP lifecycle required by that integration.

## Modifications

- Include generic stage-SP topology, placement, process launch, rank metadata, and lifecycle support.
- Add the LLaDA2-Uni decoder policy for `sp_size`, `ulysses_degree`, and `ring_degree`, requiring `sp_size = ulysses_degree * ring_degree`.
- Use the SGLang Z-Image backend for multi-rank decoding. Diffusers remains the default for SP1.
- Keep SigVQ conditioning and VAE decoding on the leader; synchronize request settings, conditioning, and seeds across ranks without emitting duplicate images.
- Propagate preparation and worker failures to the coordinator. Stop the scheduler after unsynchronized compute failures to prevent mismatched collectives on later requests.
- Shard padded noise-refiner and joint image-plus-conditioning sequences with matching position metadata, then gather in canonical global order, including non-square layouts.

The token-order handling stays in SGLang-Omni and reuses SGLang's attention and distributed primitives; it does not require editing the installed SGLang package.

Failure reporting may wait for distributed-backend cleanup; automatic rank recovery is not provided.

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

- At `b29eb2fd`, **591 API/session regression tests passed** after rebase. The preceding `77b76e98` revision passed **1,458 tests, 5 skipped**; the decoder runtime was unchanged by the rebase.
- Small-checkpoint GPU comparisons passed for SP1/Ulysses2 with SDPA and SP1/Ulysses2/Ring2 with FlashAttention. Four distributed failure-handling checks passed.
- Nine full-checkpoint API cases passed at `77b76e98` on this decoder-SP branch, including an **SP1 thinking request**. That request is downstream regression coverage for #1500, not a test run on #1500's branch.

## Accuracy

Full-checkpoint results at `b29eb2fd` used BF16, a TP1 eager thinker with compilation disabled, and the SGLang decoder with `torch_sdpa`. The SP1 reference is #1499 at `36059f66`.

| Benchmark | SP1 | Ulysses SP2 | Output comparison |
| --- | --- | --- | --- |
| GenEval | **0.88993** macro; 494/553 correct | **0.88993** macro; 494/553 correct | All 553 decoded RGB images identical |
| Original-image ImgEdit | **3.52307** | **3.50814** | All 737 decoded RGB images identical |

All requests completed without judge errors. The ImgEdit score difference comes from judging identical images. Both runs used the same requests: seed 42, CFG rescale 0.7, and 8 decoder-turbo steps. T2I used 1024x1024, CFG 4.0, and 32 dLLM steps; edit used text/image CFG 4.0/0.0 and 8 dLLM steps.

Pixel agreement applies to these benchmark settings. An earlier Turbo sample with different settings showed small BF16 differences. Full-checkpoint Ring and SP4 accuracy are not established here.

## Performance

Measured at `b29eb2fd` on H20-3e: BF16, TP1 eager thinker, compilation disabled, and SGLang SDPA decoding. SP1/SP2 ran sequentially on reserved GPUs with identical inputs and generation settings. Medians in seconds, after 3 warmups and 7 measured requests:

| Task | Decoder SP1 | Decoder SP2 | Speedup | HTTP E2E SP1 | HTTP E2E SP2 | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| T2I | 8.985 | 5.118 | 1.76x | 39.223 | 35.188 | 1.11x |
| Edit | 8.984 | 5.119 | 1.75x | 18.627 | 14.684 | 1.27x |

Both tasks used 32 dLLM steps and 8 decoder-turbo steps; edit used image CFG 1.5, unlike the accuracy benchmark above. Other request settings were unchanged.

HTTP timing excludes client image saving. Decoder stage time includes PNG encoding and was collected separately without Torch profiler. These are Ulysses SP2 results with an eager thinker; they are not directly comparable to #1486's graph-enabled E2E timings or a combined TP2+SP2 deployment.

## Contributors

- @kunchengit
- @Anmuliar
