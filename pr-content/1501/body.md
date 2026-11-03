## Motivation

Distribute LLaDA2-Uni image decoding across GPUs with SGLang Diffusion sequence parallelism. Preserve global image/conditioning order while sharding transformer work, and provide the generic stage-SP lifecycle required by that integration.

## Modifications

- Include generic stage-SP topology, placement, process launch, rank metadata, and lifecycle support.
- Add the LLaDA2-Uni decoder policy for `sp_size`, `ulysses_degree`, and `ring_degree`, requiring `sp_size = ulysses_degree * ring_degree`.
- Use the SGLang Z-Image backend for multi-rank decoding; retain both inherited decoder backends for SP1.
- Keep SigVQ conditioning and VAE decoding on the leader; synchronize request settings, conditioning, and seeds across ranks without emitting duplicate images.
- Propagate preparation and worker failures to the coordinator. Stop the scheduler after unsynchronized compute failures to prevent mismatched collectives on later requests.
- Shard padded noise-refiner and joint image-plus-conditioning sequences with matching position metadata, then gather in canonical global order, including non-square layouts.

The token-order handling stays in SGLang-Omni and reuses SGLang's attention and distributed primitives; it does not require editing the installed SGLang package.

Failure reporting may wait for distributed-backend cleanup; automatic rank recovery is not provided.

## Usage and execution scope

Set `image_decode.sp_size` and an equally sized list of distinct GPU IDs in `image_decode.gpu`; for example, SP2 can use `gpu=[1, 2]`. Put `backend`, `ulysses_degree`, `ring_degree`, and `attention_backend` under `image_decode.factory`. Multi-rank decoding requires `backend="sglang"`.

- Two-rank Ulysses: `sp_size=2`, `ulysses_degree=2`, `ring_degree=1`.
- Two-rank Ring: `sp_size=2`, `ulysses_degree=1`, `ring_degree=2`; Ring requires `attention_backend="fa"` or `"sage_attn"`.
- Four-rank Ring: `sp_size=4`, `ulysses_degree=1`, `ring_degree=4`.
- Four-rank hybrid: `sp_size=4`, `ulysses_degree=2`, `ring_degree=2`.

Pure Ulysses4 is unsupported by this checkpoint because its 30 attention heads are not divisible by four.

Normal and `decoder-turbo` decoding reuse the inherited backend controls. SP partitions a transformer's sequence, not separate requests or independent images. The split follows the padded global image-plus-conditioning sequence; attention collectives provide cross-rank token information.

The image APIs, thinker, prompts, and response formats are inherited. This is decoder SP, not thinker TP, CFG parallelism, or dynamic batching.

## Dependencies

Rebased on `main`, which includes #1502, #1500, #1499, and #2257. #1486 provides separate thinker-TP support and is not a prerequisite.

This PR contains the generic stage-SP work previously proposed in #1490. The older PR is superseded and is not an additional dependency; review both the generic runtime delta and the model-specific decoder integration here.

## Related Issues

Tracked in #2207; continues the work in #445.

## Validation

Results below use revision `6076c771`, SGLang 0.5.22, BF16, FlashAttention, and compilation disabled. Thinking and interleaved generation are inherited from `main`.

## Accuracy

The full ImgEdit benchmark uses a TP1 eager thinker and SGLang Ulysses SP2 decoding through `POST /v1/images/edits`.

| Benchmark | Mean judge score | Coverage | Judge errors |
| --- | --- | --- | --- |
| Original-image ImgEdit | **3.51696** | **737/737** | **0** |

Inputs are original images with server-side resize/crop. Requests use 8 dLLM steps, 8 decoder-turbo steps, text/image CFG 4.0/0.0, CFG rescale 0.7, and seed 42. Scoring uses the official local ImgEdit_Judge with Transformers 4.51.3 and the benchmark's JPEG-quality95 output convention; original PNGs are retained.

Earlier SDPA runs scored 3.52307 at SP1 and 3.50814 at SP2 with identical RGB outputs. They are historical results, not a matched SP1 baseline for this FA run. Full-dataset Ring/SP4 accuracy has not been evaluated.

## Performance

Decoder-only measurements on H20-3e use identical captured VQ tokens, 1024x1024 output, 8 decoder-turbo steps, and seed 42. Each configuration ran on reserved GPUs with two warmups and three measured repetitions, without Torch profiler. SP1 was measured separately for the SP2 and SP4 comparisons.

| Configuration | Decoder median (s) | Matched SP1 (s) | Speedup |
| --- | --- | --- | --- |
| Ulysses2 | **4.129** | 7.552 | **1.83x** |
| Ring2 | **4.362** | 7.552 | **1.73x** |
| Ring4 | **2.578** | 7.558 | **2.93x** |
| Ulysses2 x Ring2 | **2.557** | 7.558 | **2.96x** |

Timing includes SigVQ, DiT, VAE, and decoder communication, using synchronized wall time and the slowest rank. It excludes Thinker, HTTP, model loading, and PNG serialization; these are not serving E2E measurements.

For this decoder sample, Ulysses2 is RGB-identical to SP1. Ring2, Ring4, and hybrid4 preserve the composition with local texture differences (RGB MAE 1.249, 1.340, and 1.249 on a 0-255 scale). This sample check does not establish full-dataset equivalence.

## Contributors

- @kunchengit
- @Anmuliar
