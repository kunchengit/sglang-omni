## Motivation

LLaDA2-Uni image generation needs classifier-free guidance (CFG) over thinker-generated semantic tokens. Padding in CFG branches must not affect attention, and the MoE path must preserve the checkpoint's numerical behavior. This PR provides that correctness foundation before adding the image-generation pipeline.

## Modifications

- Add the synchronous `LowConfidenceCFG` dLLM algorithm with two-way text CFG and three-way text/image CFG.
- Add the `llada2_uni_cfg_flashinfer` attention backend to mask left-padding positions in CFG branches.
- Extend shared dLLM admission and cleanup for CFG branches, including block-wise admission accounting, staged request rows, and cancellation.
- Compute router logits in FP32, combine shared and routed expert outputs in FP32, and load the checkpoint's expert-bias buffer.
- Align understanding-side preprocessing with the multimodal system prompt and one thinker token per image patch.
- Adapt the affected server-argument and FlashInfer attention interfaces to SGLang 0.5.20, and add focused scheduler, attention, numerical, and preprocessing tests.

## Scope and dependencies

This is the first PR in the feature stack and targets `main`. It supplies the thinker correctness prerequisite for #1499; it does not add image HTTP endpoints or image decoding.

The scheduler lifecycle changes are shared runtime work motivated by LLaDA2-Uni. CFG prompt construction, attention semantics, and MoE numerical behavior remain model-specific; other models must validate those assumptions before reuse.

Thinker tensor parallelism and opt-in CUDA graph execution are handled separately in #1486.

## Related Issues

Tracked in #2207; continues the work in #445.

## Validation

The unit/regression results below were measured at `a156aa62`; the full-checkpoint accuracy baseline was measured at `2bddd0fa`.

- LLaDA2-Uni and dLLM scheduler suites: **29 passed, 0 skipped**.
- Environment: Linux, NVIDIA H20-3e, SGLang 0.5.20, PyTorch 2.13.0+cu130, Transformers 5.12.1, and Triton 3.7.1.
- Applicable formatting and static checks passed.

The tests cover CFG attention metadata/masking, admission and cancellation, router numerics, and preprocessing. They are not a full-checkpoint quality evaluation or evidence of TP/graph execution. No performance improvement is claimed here.

## Accuracy baseline

At `2bddd0fa`, TP1 eager inference with reference Torch routing completed all 1,050 MMMU examples. All answer texts are byte-identical to the post-CFG-correctness reference. The answer JSONL SHA256 is `e6ea3e0d918498d6bb2fa9b6f49c440871ffa6e2361dfcd61d0092565b359722`.

The recorded API-judge result is **516/1,050 (49.1429%)**, including **437/900 (48.5556%)** on the validation split. The historical judgment scored 515/1,050, including 438/900 on validation. Thirteen judge outcomes changed despite identical answers. Both runs had two fallback judgments: one was correct historically, while neither was correct in the current run. These score changes reflect judging variation, not changed model answers.

These results establish answer reproducibility for this PR's TP1 eager correctness baseline. See #1486 for controlled routing comparisons and TP/graph performance results. Cross-PR scores change more than TP size and should not be interpreted as an isolated measure of TP effects.

## Contributors

- @kunchengit
- @btw616
- @wzy-ustc
- @Anmuliar
- @LiRongchuan
