# -*- coding: utf-8 -*-
"""Frozen CLIP visual encoder for the SCDV + RLGC framework.

The CLIP ViT-L/14 weights stay outside the student model: all CLIP parameters
are frozen, excluded from the optimizer/checkpoints, and the same encoder runs
during training, validation, and testing. The encoder returns selected patch
grids and CLS descriptors used by SCDV and RLGC.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "The CLIP encoder requires timm. Install it with:\n"
        "  python -m pip install timm"
    ) from exc


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Nested feature representation returned by ``extract_features``.
CLIPFeatureDict = Dict[str, Dict[int, torch.Tensor]]

# Frozen CLIP settings inherited from the supplied baseline project.
PAPER_CLIP_MODEL_NAME = "vit_large_patch14_clip_224.openai"
PAPER_CLIP_SELECTED_LAYERS = (12, 24)
PAPER_CLIP_INPUT_SIZE = 224
PAPER_CLIP_USE_AMP = True
PAPER_CLIP_FEATURE_CHUNK_SIZE = 4
PAPER_CLIP_FP16_WEIGHTS = True


def _unwrap_checkpoint(checkpoint):
    """Extract a model state dict from common timm checkpoints."""
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    cleaned = {}
    for key, value in checkpoint.items():
        new_key = str(key)
        for prefix in ("module.", "model.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


def _load_local_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint does not exist: {path}")

    if path.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading a .safetensors CLIP checkpoint requires safetensors."
            ) from exc
        state_dict = load_file(str(path), device="cpu")
    else:
        try:
            checkpoint = torch.load(
                str(path),
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(str(path), map_location="cpu")
        state_dict = _unwrap_checkpoint(checkpoint)

    incompatible = model.load_state_dict(state_dict, strict=True)
    print(
        "Loaded explicit CLIP checkpoint with "
        f"missing={len(incompatible.missing_keys)}, "
        f"unexpected={len(incompatible.unexpected_keys)}"
    )


class FrozenCLIPVisionEncoder(nn.Module):
    """Return intermediate CLIP patch and CLS features.

    ``selected_layers`` uses one-based transformer block indices. For the
    24-block OpenAI CLIP ViT-L/14, ``(12, 24)`` provides a middle-level local
    representation and a deep semantic representation.
    """

    def __init__(
        self,
        model_name: str = "vit_large_patch14_clip_224.openai",
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        selected_layers: Sequence[int] = (12, 24),
        input_size: Optional[int] = None,
        use_amp: bool = True,
        feature_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.checkpoint_path = (
            str(checkpoint_path).strip() if checkpoint_path else ""
        )

        self.vision_model = timm.create_model(
            self.model_name,
            pretrained=bool(pretrained and not self.checkpoint_path),
            num_classes=0,
            global_pool="",
        )
        if self.checkpoint_path:
            _load_local_checkpoint(self.vision_model, self.checkpoint_path)

        if not hasattr(self.vision_model, "blocks"):
            raise TypeError(
                "The selected timm CLIP model must expose transformer blocks, "
                f"but {self.model_name!r} does not."
            )
        if not hasattr(self.vision_model, "forward_features"):
            raise TypeError(
                "The selected timm CLIP model must implement forward_features."
            )

        self.block_count = len(self.vision_model.blocks)
        self.selected_layers = tuple(
            sorted(set(int(value) for value in selected_layers))
        )
        if len(self.selected_layers) < 2:
            raise ValueError(
                "Encoder requires at least two selected CLIP layers, "
                f"got {self.selected_layers}."
            )
        if any(
            value < 1 or value > self.block_count
            for value in self.selected_layers
        ):
            raise ValueError(
                f"selected_layers must be in [1, {self.block_count}], "
                f"got {self.selected_layers}."
            )

        pretrained_cfg = getattr(self.vision_model, "pretrained_cfg", {}) or {}
        cfg_input_size = pretrained_cfg.get("input_size", (3, 224, 224))
        inferred_size = int(cfg_input_size[-1])
        self.input_size = int(input_size) if input_size else inferred_size
        self.use_amp = bool(use_amp)
        self.feature_chunk_size = max(1, int(feature_chunk_size))
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")

        clip_mean = tuple(
            float(value)
            for value in pretrained_cfg.get("mean", DEFAULT_CLIP_MEAN)
        )
        clip_std = tuple(
            float(value)
            for value in pretrained_cfg.get("std", DEFAULT_CLIP_STD)
        )
        if len(clip_mean) != 3 or len(clip_std) != 3:
            raise ValueError(
                f"Invalid CLIP preprocessing statistics: "
                f"mean={clip_mean}, std={clip_std}."
            )

        self.num_prefix_tokens = int(
            getattr(self.vision_model, "num_prefix_tokens", 1)
        )
        self.embed_dim = int(
            getattr(
                self.vision_model,
                "embed_dim",
                getattr(self.vision_model, "num_features", 0),
            )
        )
        if self.embed_dim <= 0:
            raise RuntimeError(
                "Could not infer CLIP embedding dimension from the timm model."
            )

        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_mean",
            torch.tensor(clip_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor(clip_std).view(1, 3, 1, 1),
            persistent=False,
        )

        self._captured_features: Dict[int, torch.Tensor] = {}
        self._hook_handles = []
        for layer_index in self.selected_layers:
            block = self.vision_model.blocks[layer_index - 1]
            handle = block.register_forward_hook(
                self._make_feature_hook(layer_index)
            )
            self._hook_handles.append(handle)

        self.vision_model.requires_grad_(False)
        self.vision_model.eval()

        print("=" * 100)
        print("FROZEN TIMM CLIP VISUAL ENCODER")
        print(f"Model name: {self.model_name}")
        print(
            "Weights: "
            + (
                f"explicit checkpoint {self.checkpoint_path}"
                if self.checkpoint_path
                else ("timm pretrained/cache" if pretrained else "random")
            )
        )
        print(f"Transformer blocks: {self.block_count}")
        print(f"Selected layers: {self.selected_layers}")
        print(f"Embedding dimension: {self.embed_dim}")
        print(f"Input size: {self.input_size}")
        print(f"Feature chunk size: {self.feature_chunk_size}")
        print("CLIP is frozen and excluded from student checkpoints.")
        print("=" * 100)

    def _make_feature_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                if not output:
                    raise RuntimeError(
                        f"CLIP block {layer_index} returned an empty output."
                    )
                output = output[0]
            if not torch.is_tensor(output):
                raise RuntimeError(
                    f"CLIP block {layer_index} returned unsupported type "
                    f"{type(output)!r}."
                )
            self._captured_features[layer_index] = output
        return hook

    def train(self, mode: bool = True):
        # The encoder always stays deterministic and frozen.
        super().train(False)
        self.vision_model.eval()
        return self

    def _prepare(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise RuntimeError(
                "CLIP encoder expects ImageNet-normalized BCHW RGB input."
            )
        mean = self.imagenet_mean.to(device=image.device, dtype=image.dtype)
        std = self.imagenet_std.to(device=image.device, dtype=image.dtype)
        image_01 = (image * std + mean).clamp(0.0, 1.0)
        image_01 = F.interpolate(
            image_01,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        clip_mean = self.clip_mean.to(device=image.device, dtype=image.dtype)
        clip_std = self.clip_std.to(device=image.device, dtype=image.dtype)
        return (image_01 - clip_mean) / clip_std

    def _forward_selected(self, image: torch.Tensor) -> Dict[int, torch.Tensor]:
        self._captured_features.clear()
        _ = self.vision_model.forward_features(image)
        missing = [
            index
            for index in self.selected_layers
            if index not in self._captured_features
        ]
        if missing:
            raise RuntimeError(
                "Failed to capture outputs from CLIP blocks: "
                f"{missing}."
            )
        captured = {
            index: self._captured_features[index]
            for index in self.selected_layers
        }
        self._captured_features.clear()
        return captured

    def _tokens_to_patch_and_cls(
        self,
        feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert BNC/BCHW/BHWC output to a patch grid and global vector."""
        if feature.ndim == 4:
            if feature.shape[1] == self.embed_dim:
                patch = feature.contiguous()
            elif feature.shape[-1] == self.embed_dim:
                patch = feature.permute(0, 3, 1, 2).contiguous()
            else:
                raise RuntimeError(
                    "Cannot identify the channel dimension in CLIP feature "
                    f"shape {tuple(feature.shape)}."
                )
            # Hierarchical or explicitly reshaped features may not retain CLS.
            cls = patch.mean(dim=(2, 3))
            return patch, cls

        if feature.ndim != 3:
            raise RuntimeError(
                "A CLIP block must return BNC/BCHW/BHWC features, "
                f"got {tuple(feature.shape)}."
            )

        if feature.shape[-1] != self.embed_dim:
            if feature.shape[1] == self.embed_dim:
                feature = feature.transpose(1, 2)
            else:
                raise RuntimeError(
                    f"Unexpected CLIP token shape {tuple(feature.shape)}; "
                    f"expected embedding dimension {self.embed_dim}."
                )

        token_count = int(feature.shape[1])
        prefix_count = min(self.num_prefix_tokens, token_count)
        patch_tokens = feature[:, prefix_count:]
        patch_count = int(patch_tokens.shape[1])
        side = int(round(math.sqrt(patch_count)))

        if side * side != patch_count:
            found = False
            for candidate_prefix in (0, 1, 2):
                candidate_count = token_count - candidate_prefix
                candidate_side = int(round(math.sqrt(max(0, candidate_count))))
                if candidate_side * candidate_side == candidate_count:
                    prefix_count = candidate_prefix
                    patch_tokens = feature[:, candidate_prefix:]
                    side = candidate_side
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    "Cannot reshape CLIP tokens into a square patch grid: "
                    f"shape={tuple(feature.shape)}, "
                    f"num_prefix_tokens={self.num_prefix_tokens}."
                )

        if prefix_count > 0:
            cls = feature[:, 0]
        else:
            cls = patch_tokens.mean(dim=1)

        patch = patch_tokens.transpose(1, 2).reshape(
            feature.shape[0],
            feature.shape[2],
            side,
            side,
        ).contiguous()
        return patch, cls.contiguous()

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> CLIPFeatureDict:
        """Extract frozen intermediate features for a full image batch."""
        if images.ndim != 4:
            raise RuntimeError(
                f"Expected BCHW images, got shape {tuple(images.shape)}."
            )
        if images.shape[0] == 0:
            raise ValueError("CLIP feature extraction received an empty batch.")

        chunk_size = max(
            1,
            int(chunk_size or self.feature_chunk_size),
        )
        patch_chunks: Dict[int, list[torch.Tensor]] = {
            layer: [] for layer in self.selected_layers
        }
        cls_chunks: Dict[int, list[torch.Tensor]] = {
            layer: [] for layer in self.selected_layers
        }

        for start in range(0, int(images.shape[0]), chunk_size):
            chunk = self._prepare(images[start:start + chunk_size])
            device_type = chunk.device.type
            amp_enabled = bool(self.use_amp and device_type == "cuda")
            with torch.autocast(
                device_type=device_type,
                enabled=amp_enabled,
            ):
                captured = self._forward_selected(chunk)

            for layer in self.selected_layers:
                patch, cls = self._tokens_to_patch_and_cls(captured[layer])
                patch_chunks[layer].append(patch.detach())
                cls_chunks[layer].append(cls.detach())

            del captured, chunk

        return {
            "patch_features": {
                layer: torch.cat(patch_chunks[layer], dim=0)
                for layer in self.selected_layers
            },
            "cls_features": {
                layer: torch.cat(cls_chunks[layer], dim=0)
                for layer in self.selected_layers
            },
        }

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> CLIPFeatureDict:
        return self.extract_features(images, chunk_size=chunk_size)



def build_paper_clip_encoder(device: torch.device) -> FrozenCLIPVisionEncoder:
    """Build the single frozen CLIP encoder used by train/val/test."""
    from config import config
    local_checkpoint = config.get("clip_checkpoint", "")
    if not local_checkpoint or not Path(local_checkpoint).is_file():
        raise FileNotFoundError("Frozen CLIP weights are required. See README.md; "
                                "set CLIP_CHECKPOINT or populate weights/clip_visual.pth.")
    encoder = FrozenCLIPVisionEncoder(
        model_name=PAPER_CLIP_MODEL_NAME,
        pretrained=False,
        checkpoint_path=local_checkpoint,
        selected_layers=PAPER_CLIP_SELECTED_LAYERS,
        input_size=PAPER_CLIP_INPUT_SIZE,
        use_amp=PAPER_CLIP_USE_AMP,
        feature_chunk_size=PAPER_CLIP_FEATURE_CHUNK_SIZE,
    ).to(device)
    if PAPER_CLIP_FP16_WEIGHTS and device.type == "cuda":
        encoder.vision_model.half()
    encoder.eval()

    missing = [
        layer
        for layer in PAPER_CLIP_SELECTED_LAYERS
        if layer not in encoder.selected_layers
    ]
    if missing:
        raise ValueError(
            "Requested layers must be included in frozen CLIP features; "
            f"missing={missing}."
        )
    return encoder
