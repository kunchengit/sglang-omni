# SPDX-License-Identifier: Apache-2.0
"""Single-process adapter over diffusers' ZImage backbone.

This is the semantic-only LLaDA2-Uni checkpoint, not LLaDA-Image's
QueryFormer/text-conditioned transformer. Only the diffusers backend is
supported; this adapter does not implement native SGLang or sequence parallelism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn


def _decoder_config(cfg: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "all_patch_size": (2,),
        "all_f_patch_size": (1,),
        "in_channels": 16,
        "dim": 3840,
        "n_layers": 30,
        "n_refiner_layers": 2,
        "n_heads": 30,
        "n_kv_heads": 30,
        "norm_eps": 1e-5,
        "qk_norm": True,
        "cap_feat_dim": 4096,
        "rope_theta": 256.0,
        "t_scale": 1000.0,
        "axes_dims": (32, 48, 48),
        "axes_lens": (32768, 1024, 1024),
    }
    # HF metadata and the original unused siglip_feat_dim are not architecture.
    unknown = set(cfg) - defaults.keys() - {"siglip_feat_dim"}
    unknown = {key for key in unknown if not key.startswith("_")}
    if unknown:
        raise ValueError(f"Unsupported decoder config keys: {sorted(unknown)}")
    defaults.update({key: cfg[key] for key in defaults.keys() & cfg.keys()})
    for key in ("all_patch_size", "all_f_patch_size", "axes_dims", "axes_lens"):
        defaults[key] = tuple(defaults[key])
    return defaults


def _semantic_checkpoint(weights):
    seen = set()
    for name, value in weights:
        if name.startswith("semantic_embedder."):
            name = "cap_embedder." + name.removeprefix("semantic_embedder.")
        if name in seen:
            raise ValueError(f"Duplicate decoder checkpoint parameter: {name}")
        seen.add(name)
        yield name, value


class ZImageTransformer2DModelWrapper(nn.Module):
    """Load a semantic decoder checkpoint and expose its forward convention.

    Args:
        decoder_dir: Directory containing ``model.safetensors``.
        cfg: Decoder config, with cap_feat_dim/axes_lens overrides applied
            by the image decoder. Unknown architectural options fail closed.
        device, dtype: Weight placement and inference dtype.
        backend: Only ``diffusers`` is supported; there is no fallback.

    Requires diffusers with ZImage support (tested with 0.37.0). Inputs are
    lists of [C, F, H, W] latents and [L, D] semantic features; t is transport
    time in [0, 1]. Diffusers embeds t*t_scale and returns positive velocity,
    so no timestep inversion or output negation is applied.
    """

    def __init__(
        self,
        decoder_dir: str,
        cfg: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        *,
        backend: str = "diffusers",
    ) -> None:
        super().__init__()
        if backend != "diffusers":
            raise ValueError("Image decoding supports only backend='diffusers'")
        from diffusers.models.transformers.transformer_z_image import (
            ZImageTransformer2DModel,
        )

        self.backend = backend
        self.cfg = _decoder_config(cfg)
        with torch.device("meta"):
            model = ZImageTransformer2DModel(**self.cfg)
        checkpoint = str(Path(decoder_dir) / "model.safetensors")
        state = dict(_semantic_checkpoint(load_file(checkpoint, device="cpu").items()))
        model.load_state_dict(state, strict=True, assign=True)
        self.model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    def forward(
        self,
        x,
        t,
        cap_feats,
        return_dict: bool = True,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        if not x or len(x) != len(cap_feats):
            raise ValueError(
                "Decoder requires one semantic feature sequence per latent"
            )
        if (patch_size, f_patch_size) not in set(
            zip(self.cfg["all_patch_size"], self.cfg["all_f_patch_size"])
        ):
            raise ValueError("Unsupported decoder patch-size pair")
        for latent, cap in zip(x, cap_feats):
            if latent.ndim != 4 or latent.shape[0] != self.cfg["in_channels"]:
                raise ValueError("Decoder latents must have shape [C, F, H, W]")
            if any(
                size < 1 or size % patch
                for size, patch in zip(
                    latent.shape[1:], (f_patch_size, patch_size, patch_size)
                )
            ):
                raise ValueError(
                    "Decoder latent dimensions must be divisible by patch sizes"
                )
            if (
                cap.ndim != 2
                or cap.shape[0] < 1
                or cap.shape[1] != self.cfg["cap_feat_dim"]
            ):
                raise ValueError(
                    "Decoder semantic features must have shape [L, cap_feat_dim]"
                )
        t = torch.as_tensor(t, dtype=torch.float32, device=x[0].device)
        if t.ndim > 1 or t.numel() not in {1, len(x)}:
            raise ValueError("Decoder timestep must be a scalar or batch vector")
        t = t.reshape(-1).expand(len(x))
        return self.model(
            x=x,
            t=t,
            cap_feats=cap_feats,
            return_dict=return_dict,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
        )
