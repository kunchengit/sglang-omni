## Motivation

Distribute LLaDA2-Uni image decoding across GPUs using SGLang Diffusion's sequence-parallel runtime. Shard the transformer sequence while preserving the global image/conditioning order, so SP reduces backbone work without changing which tokens attend to one another.

## Modifications

- Include generic stage SP topology, placement, process launch, rank metadata, and lifecycle support, then connect it to the LLaDA2-Uni image decoder.
- Configure `sp_size`, `ulysses_degree`, and `ring_degree`, requiring `sp_size = ulysses_degree * ring_degree`. Multi-rank decoding requires `factory.backend="sglang"`; Diffusers remains available for SP1.
- Reuse SGLang's Z-Image transformer, sharding/gathering utilities, attention, and distributed process groups. Ulysses and Ring use the existing SGLang attention implementations.
- Keep SigVQ conditioning and VAE decoding on the leader. Synchronize the conditioning/seed and propagate preparation failures so followers participate in the same decoder collectives without emitting duplicate images.
- Wrap the noise-refiner and joint transformer blocks in Omni. Partition their already padded sequences and matching position metadata, run the blocks on local shards, then gather in global sequence order.
- Shard the joint image-plus-conditioning sequence rather than replicating its conditioning suffix on each rank. Preserve the canonical order through attention and gathering, including non-square padded layouts.

The token-order correction lives entirely in SGLang-Omni. This implementation does not require a source patch to the installed SGLang package or introduce a new attention kernel.

## Configuration and execution boundary

Set the decoder stage's `sp_size` and matching factory degrees. For two-rank Ulysses use `ulysses_degree=2, ring_degree=1`; for two-rank Ring use `ulysses_degree=1, ring_degree=2`. Ring requires a supported attention backend (`fa` or `sage_attn`); the configuration rejects unsupported combinations.

SP partitions transformer sequence tokens, not requests or independent images. Image and conditioning tokens need not each split exactly in half: the split follows the padded global sequence. Attention collectives provide access to the other ranks' token information.

The API, prompt construction, and thinker remain inherited. This is decoder SP, not thinker TP, CFG parallelism, or dynamic batching.

## Scope and dependencies

The `llada2/decoder-sp` branch contains the implementation synchronized from `pipeline/stage-sp` and is stacked directly on #1502 (`llada2/interleaved-image-generation`). It inherits native and thinking image generation through that dependency. #1486 is a sibling thinker-TP branch, not a prerequisite.

The generic stage SP work associated with #1490 is already included in this stack. Reconcile that overlap before merge rather than applying it twice. The comparison with `main` currently includes the earlier image-generation prerequisites.

Part of #445; this does not close the full roadmap.

## Accuracy Test

- The latest synchronization passed the relevant unit suites; GPU-dependent skipped cases are not counted as GPU validation.
- Recorded validation of the Omni-only token-order fix compared SP1 and Ulysses SP2 on six fixed-input cases, including historical interleaved frames and a padded 22x46 conditioning grid. All 312 compared tensors and all six output images matched exactly in that run.
- Fixed T2I/edit decoder-input checks also produced identical SP1/Ulysses SP2 PNGs. These isolate decoder correctness; they do not prove end-to-end raw-image-edit quality.

These GPU results predate the latest input-preprocessing synchronization. Exact equality is limited to the tested environment, attention configuration, and inputs. No equivalent final-fix Ring parity or SP4 result is claimed here.

## Benchmark & Profiling

Recorded Omni-only decoder measurements: H20-3e, BF16, Torch 2.13.0+cu130, FlashAttention, decoder-turbo with 8 steps, seed 42, and 1024x1024 output. Reuse fixed VQ inputs from serving, perform three warmup iterations, and report the median of seven measured decoder executions.

| Fixed decoder input | SP1 | Ulysses SP2 | Speedup |
| --- | ---: | ---: | ---: |
| T2I | 7.503 s | 4.161 s | 1.80x |
| Edit | 7.499 s | 4.163 s | 1.80x |

These are decoder timings, not request E2E or steady-state serving throughput. They are prior validation of the decoder change, not a new benchmark of the latest rebased head. Final-head Ring performance and broader image-quality coverage remain to be reported.

## Contributors

- @kunchengit
- @Anmuliar
