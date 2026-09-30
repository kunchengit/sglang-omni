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

The thinker runs eagerly unless graphs are enabled. The cookbook's graph examples inherit `torch.compile=true`; the command above explicitly disables it to match the reported accuracy and performance runs. No compile-enabled accuracy or performance result is claimed here.

Native SGLang routing is selected separately:

```bash
--thinker.engine.json_model_override_args '{"llada2_uni_topk_backend":"sglang"}'
```

The default is `torch`. See #2257 for the MMMU routing ablation and its numerical implications.

For the supplied BF16 H20-3e TP2 tuning configurations (SGLang 0.5.20 / Triton 3.7.1):

```bash
export SGLANG_MOE_CONFIG_DIR=/absolute/path/to/sglang-omni/examples/tuning/llada2_uni/h20_tp2
```

Replace the repository path; point above `configs/`. Omit this setting for other hardware or configurations without separate validation.

## Validation

- Thinker TP regression: **69 passed, 1 skipped**. The routing selector added in this PR passed 9 focused precision tests and six real TP2 T2I/edit/MMMU output checks.
- With native SGLang routing, TP2, graphs on, and compile off: GenEval macro score **0.88023** (487/553 correct); original-image ImgEdit **3.54093** (737/737 completed). These are not full-dataset results for the default Torch route.

## Performance

Measured on this PR's thinker TP implementation on H20-3e with BF16, **native SGLang TopK**, decode graphs enabled, and compilation disabled. Both deployments use the same checkpoint, prompt/source image, seed, and generation settings. The decoder remains SP1 on a separate GPU with `torch_sdpa`.

Sequential runs on reserved GPUs; 3 warmups and 7 measured requests per case. Values are median seconds.

| Task | Thinker TP1 | Thinker TP2 | Speedup | HTTP E2E TP1 | HTTP E2E TP2 | Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| T2I | 6.529 | 5.097 | 1.28x | 15.525 | 14.086 | 1.10x |
| Edit | 3.105 | 2.704 | 1.15x | 12.197 | 11.721 | 1.04x |

Settings: seed 42, CFG rescale 0.7, 32 dLLM steps, and 8 decoder-turbo steps. T2I uses CFG 4.0 at 1024x1024; edit uses text CFG 4.0 and image CFG 1.5, with dimensions derived from the source.

HTTP timing excludes client image saving. Thinker timings come from a separate stage-event run without Torch profiler. TP2 loads the supplied MoE tuning configurations; TP1 uses the installed configuration root. These results measure those deployments, not the default Torch route or the isolated benefit of tuning.

## Dependencies

Depends on #1502. Decoder SP (#1501) is a separate PR. Expert parallelism, quantization, and request batching are outside this scope.

Tracked in #2207; continues #445.

## Contributors

- @kunchengit
- @btw616
- @LiRongchuan
- @Anmuliar
