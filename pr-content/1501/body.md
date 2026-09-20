> Draft: requesting feedback on the LLaDA2-Uni image-decoder sequence-parallel integration. Multi-GPU/NCCL execution and output-parity validation remain pending.

## Motivation

LLaDA2-Uni native image generation ends in a Z-Image diffusion decoder. Large output grids can benefit from distributing that decoder across stage processes while reusing the model-neutral sequence-parallel topology and lifecycle introduced in #1490.

This PR adds the decoder integration using SGLang's Z-Image runtime, latent sharding/gathering, attention, forward-context, and weight-loading interfaces.

## Modifications

- Add a model-owned image-decoder SP policy with backend, attention backend, SP/Ulysses/Ring decomposition, and checkpoint-loading configuration.
- Use the SGLang Z-Image backend for multi-rank decoding while preserving the existing Diffusers path as the default for SP1.
- Pass stage role, rank metadata, parallel decomposition, attention backend, and checkpoint-loading device through the decoder factory.
- Keep SigVQ conditioning and VAE decoding on the leader while follower ranks participate in diffusion collectives without returning duplicate images.
- Synchronize leader validation status, conditioning features, and random seeds before diffusion, and add focused coverage for configuration, loading, collectives, and leader/follower behavior.

## Dependencies and review scope

This branch is stacked on #1499 and #1490, so the comparison with `main` temporarily includes both prerequisites. Review should focus on the LLaDA2-Uni decoder SP integration.

After the prerequisites land, the branch will be rebased so the Ready PR contains only the decoder integration.

## Related Issues

Part of #445. This PR does not close the full LLaDA2-Uni support issue.

Required dependencies:

- #1499: native image decoder and image-generation pipeline
- #1490: model-neutral sequence-parallel stage configuration and runtime

## Accuracy Test

### Local verification

- Focused image-decoder SP tests: `19 passed`
- Draft 1 and generic SP regression tests: `47 passed`
- Black, isort, Ruff, Python `compileall`, and `git diff --check`: passed

Real GPU/NCCL execution and SP1/SP2/SP4 output parity have not been validated for this Draft.

## Benchmark & Profiling

Not run for this Draft; no latency, memory, or throughput claim is made. GPU parity, memory, and latency measurements are required before the PR is marked Ready.

## Contributors

- @kunchengit
- @Anmuliar

## Checklist

- [x] Format the changed code.
- [x] Add unit tests.
- [x] Update documentation and configuration as needed.
- [ ] Provide GPU accuracy and performance results before marking the PR Ready.
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

This PR is intentionally opened as a Draft. Self-hosted GPU CI has not been requested.