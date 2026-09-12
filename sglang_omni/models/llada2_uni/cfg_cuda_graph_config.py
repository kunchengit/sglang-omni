# SPDX-License-Identifier: Apache-2.0
"""CFG graph configuration checks, without importing CUDA runner classes."""

from sglang.srt.arg_groups.model_override_base import resolved_view
from sglang.srt.model_executor.cuda_graph_config import Backend


def validate_cfg_cuda_graph_config(server_args) -> None:
    """Validate requested features; never rewrite user-selected graph settings."""
    view = resolved_view(server_args)
    if view.dllm_algorithm != "LowConfidenceCFG":
        return
    config = view.cuda_graph_config
    if config.prefill.backend != Backend.DISABLED:
        raise ValueError("DLLM CFG does not support prefill CUDA graphs")
    if config.decode.backend == Backend.DISABLED:
        return
    if config.decode.backend != Backend.FULL:
        raise ValueError("DLLM CFG currently supports only full decode CUDA graphs")
    if view.attention_backend != "llada2_uni_cfg_flashinfer":
        raise ValueError(
            "DLLM CFG CUDA graphs require the llada2_uni_cfg_flashinfer backend; "
            "check backend capability registration before building server args"
        )
    from sglang.srt.environ import envs

    if envs.SGLANG_ENABLE_METADATA_GLUE_GRAPH.get():
        raise ValueError(
            "DLLM CFG requires host-refreshed attention plans; "
            "SGLANG_ENABLE_METADATA_GLUE_GRAPH must be disabled"
        )
