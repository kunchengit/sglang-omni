# SPDX-License-Identifier: Apache-2.0
"""FlashInfer backend for LLaDA2 image-edit CFG padding."""

from __future__ import annotations

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper
from sglang.srt.server_args import (
    ATTENTION_BACKEND_CHOICES,
    add_attention_backend_choices,
)
from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    PrefillMetadata,
    merge_state,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc

from sglang_omni.models.llada2_uni.cfg_cuda_graph_metadata import cfg_attention_geometry

CFG_ATTENTION_BACKEND = "llada2_uni_cfg_flashinfer"


class LLaDA2CFGFlashInferAttnBackend(FlashInferAttnBackend):
    """Stock FlashInfer plus opt-in image-edit CFG left-pad masking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_wrappers != 1:
            raise ValueError("DLLM CFG requires one attention wrapper")
        if self.flashinfer_kv_cache_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("DLLM CFG requires FP16/BF16 KV cache")
        self._cfg_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend="fa2"
        )

    @staticmethod
    def _clear_stale_ragged_custom_mask(ragged_prefill_wrapper) -> None:
        """Clear a previous edit batch's private FlashInfer custom-mask state."""
        if ragged_prefill_wrapper.is_cuda_graph_enabled:
            return
        for attr in ("_custom_mask_buf", "_mask_indptr_buf"):
            if not hasattr(ragged_prefill_wrapper, attr):
                raise RuntimeError(
                    f"Unsupported FlashInfer ragged wrapper: missing {attr}"
                )
            setattr(ragged_prefill_wrapper, attr, None)

    def init_forward_metadata(self, forward_batch):
        self._cfg_local_left_pad_active = False
        cfg_prefill_wrapper = self._cfg_prefill_wrapper_ragged
        self._clear_stale_ragged_custom_mask(cfg_prefill_wrapper)
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().init_forward_metadata(forward_batch)
        geometry = cfg_attention_geometry(forward_batch)
        if not any(geometry.pad):
            return super().init_forward_metadata(forward_batch)
        self._plan_cfg_attention(
            forward_batch, geometry, self.prefill_wrappers_paged, graph=False
        )

    def init_cuda_graph_state(self, max_bs, max_num_tokens, kv_indices_buf=None):
        super().init_cuda_graph_state(max_bs, max_num_tokens, kv_indices_buf)
        device = self.cuda_graph_kv_indices[0].device
        self._cfg_graph_cached_pad = torch.zeros(
            max_bs, dtype=torch.int32, device=device
        )
        self._cfg_graph_paged_lens = torch.zeros_like(self._cfg_graph_cached_pad)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        """Refresh host plans before capture/replay, never via metadata-glue capture."""
        self._cfg_local_left_pad_active = False
        if not forward_batch.forward_mode.is_dllm_extend():
            return super().init_forward_metadata_out_graph(forward_batch, in_capture)
        if not hasattr(forward_batch, "dllm_left_pad_lens"):
            from sglang_omni.models.llada2_uni.cfg_cuda_graph import (
                attach_cfg_graph_views,
            )

            registry = getattr(self, "_cfg_graph_registry", None)
            if registry is None:
                raise RuntimeError("DLLM CFG replay metadata registry is not bound")
            attach_cfg_graph_views(forward_batch, registry)
        geometry = cfg_attention_geometry(forward_batch)
        if any(geometry.local_pad):
            raise ValueError(
                "DLLM CFG CUDA graph requires left padding entirely in the cached "
                "prefix; query-local padding must use eager attention"
            )
        bs = forward_batch.batch_size
        if in_capture:
            self._prepare_cuda_graph_metadata(
                bs, forward_batch.positions.numel(), forward_batch.forward_mode, None
            )
        self._plan_cfg_attention(
            forward_batch, geometry, self.prefill_cuda_graph_metadata[bs], graph=True
        )

    def _plan_cfg_attention(self, forward_batch, geometry, wrappers, *, graph):
        seq_lens = forward_batch.seq_lens
        prefix_lens = forward_batch.extend_prefix_lens
        bs = len(geometry.query)
        # Length decisions use CPU mirrors. Stable derived buffers are backend-owned.
        if graph:
            kv_view = self.kv_index_translator.build_index_table(
                req_pool_indices=forward_batch.req_pool_indices[:bs],
                seq_lens=seq_lens[:bs],
                into=self.kv_read_tables,
            )
            cached_left_pad_lens = self._cfg_graph_cached_pad[:bs]
            paged_kernel_lens = self._cfg_graph_paged_lens[:bs]
            torch.minimum(
                forward_batch.dllm_left_pad_lens,
                prefix_lens,
                out=cached_left_pad_lens,
            )
            torch.sub(prefix_lens, cached_left_pad_lens, out=paged_kernel_lens)
        else:
            kv_view = self.kv_index_translator.index_table_for_batch(forward_batch)
            cached_left_pad_lens = torch.tensor(
                geometry.cached_pad, dtype=seq_lens.dtype, device=seq_lens.device
            )
            paged_kernel_lens = torch.tensor(
                geometry.paged_lens, dtype=seq_lens.dtype, device=seq_lens.device
            )
        local_pad_active = any(geometry.local_pad)
        cfg_prefill_wrapper = self._cfg_prefill_wrapper_ragged
        prefill_indices_updater = self.indices_updater_prefill
        if local_pad_active:
            flattened_request_masks = []
            for query_length, local_left_pad_length in zip(
                geometry.query, geometry.local_pad
            ):
                request_attention_mask = torch.ones(
                    (query_length, query_length),
                    dtype=torch.bool,
                    device=seq_lens.device,
                )
                if local_left_pad_length:
                    request_attention_mask[:, :local_left_pad_length] = False
                    # Give discarded pad queries a diagonal key to avoid invalid softmax.
                    pad_indices = torch.arange(
                        local_left_pad_length, device=seq_lens.device
                    )
                    request_attention_mask[pad_indices, pad_indices] = True
                flattened_request_masks.append(request_attention_mask.flatten())
            custom_mask = torch.cat(flattened_request_masks)
            qo_indptr = torch.zeros(
                seq_lens.numel() + 1,
                dtype=torch.int32,
                device=seq_lens.device,
            )
            qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            # Exclude edit pads from cached-prefix attention without replacing the custom mask.
            prefill_indices_updater.call_begin_forward(
                cfg_prefill_wrapper,
                wrappers[0],
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                sum(geometry.paged_lens),
                seq_lens,
                prefix_lens,
                cached_left_pad_lens,
                prefill_indices_updater.kv_indptr[0],
                prefill_indices_updater.qo_indptr[0],
                False,
                None,
                fixed_split_size=self.prefill_split_tile_size,
                kv_view=kv_view,
            )
            cfg_prefill_wrapper.begin_forward(
                qo_indptr,
                qo_indptr,
                prefill_indices_updater.num_qo_heads,
                prefill_indices_updater.num_kv_heads,
                prefill_indices_updater.head_dim,
                custom_mask=custom_mask,
                causal=False,
                q_data_type=prefill_indices_updater.q_data_type,
                kv_data_type=prefill_indices_updater.data_type,
                non_blocking=True,
                fixed_split_size=self.prefill_split_tile_size,
            )
            self._cfg_local_left_pad_active = True
            self._cfg_has_cached_prefix = any(geometry.paged_lens)
        else:
            self._cfg_local_left_pad_active = False
            prefill_indices_updater.call_begin_forward(
                prefill_indices_updater.prefill_wrapper_ragged,
                wrappers[0],
                forward_batch.req_pool_indices,
                paged_kernel_lens,
                sum(geometry.paged_lens),
                seq_lens,
                prefix_lens,
                cached_left_pad_lens,
                prefill_indices_updater.kv_indptr[0],
                prefill_indices_updater.qo_indptr[0],
                True,
                None,
                fixed_split_size=self.prefill_split_tile_size,
                kv_view=kv_view,
            )

        self.forward_metadata = PrefillMetadata(
            wrappers,
            use_ragged=True,
            extend_no_prefix=False,
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
    ):
        if not getattr(self, "_cfg_local_left_pad_active", False):
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        if k is None or v is None:
            raise RuntimeError("CFG first-block attention requires explicit K/V")

        q_view = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        k_view = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_view = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        if getattr(self, "_cfg_has_cached_prefix", False):
            current_output, current_lse = (
                self._cfg_prefill_wrapper_ragged.forward_return_lse(
                    q_view,
                    k_view,
                    v_view,
                    causal=False,
                    sm_scale=layer.scaling,
                    logits_soft_cap=layer.logit_cap,
                )
            )
            cached_output, cached_lse = self.prefill_wrappers_paged[
                0
            ].forward_return_lse(
                q_view,
                self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
                k_scale=layer.k_scale_float,
                v_scale=layer.v_scale_float,
            )
            attention_output, _ = merge_state(
                current_output, current_lse, cached_output, cached_lse
            )
        else:
            attention_output = self._cfg_prefill_wrapper_ragged.forward(
                q_view,
                k_view,
                v_view,
                causal=False,
                sm_scale=layer.scaling,
                logits_soft_cap=layer.logit_cap,
            )
        if save_kv_cache:
            kv_cache_location = (
                forward_batch.out_cache_loc
                if not layer.is_cross_attention
                else forward_batch.encoder_out_cache_loc
            )
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                KVWriteLoc(kv_cache_location, self.forward_metadata.swa_out_cache_loc),
                k,
                v,
                *self._kv_write_scales(layer),
            )
        return attention_output.view(-1, layer.tp_q_head_num * layer.head_dim)


def register_llada2_cfg_flashinfer_backend() -> None:
    """Register DLLM pad masking without changing stock/text-condition backends."""

    def _create_backend(runner):
        if runner.use_mla_backend:
            raise ValueError("LLaDA2 CFG attention does not use an MLA backend")
        return LLaDA2CFGFlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )

    ATTENTION_BACKENDS[CFG_ATTENTION_BACKEND] = _create_backend
    if CFG_ATTENTION_BACKEND not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices([CFG_ATTENTION_BACKEND])
