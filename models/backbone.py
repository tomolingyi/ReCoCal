# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

"""Fixed MiT-B2 backbone used by the paper model."""

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

from config import config

try:
    from transformers import SegformerConfig, SegformerModel
except ImportError as exc:
    raise ImportError(
        "MiT-B2 requires transformers. Install it with:\n"
        "  python -m pip install transformers"
    ) from exc

DEFAULT_MIT_B2_DIR = Path(config["mit_b2_dir"])
STUDENT_CHANNELS = (64, 128, 320, 512)
MIT_B2_DEPTHS = (3, 4, 6, 3)
STUDENT_REDUCTIONS = (4, 8, 16, 32)
BACKBONE_DROP_PATH = 0.10
BACKBONE_GRADIENT_CHECKPOINTING = True

def _check_local_mit_b2(
    local_dir: Path,
    require_weights: bool,
) -> None:
    if not local_dir.is_dir():
        raise FileNotFoundError(
            f"MiT-B2 local folder does not exist: {local_dir}"
        )
    if not (local_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"Missing MiT-B2 config: {local_dir / 'config.json'}"
        )
    if require_weights:
        expected = (
            local_dir / "model.safetensors",
            local_dir / "pytorch_model.bin",
        )
        if not any(path.is_file() for path in expected):
            raise FileNotFoundError(
                "Missing MiT-B2 weights. Expected one of:\n"
                f"  {expected[0]}\n"
                f"  {expected[1]}"
            )


def build_local_mit_b2(
    local_dir: str,
    pretrained: bool = True,
    drop_path_rate: float = 0.1,
) -> SegformerModel:
    local_path = Path(local_dir)
    _check_local_mit_b2(
        local_dir=local_path,
        require_weights=bool(pretrained),
    )
    config = SegformerConfig.from_pretrained(
        str(local_path),
        local_files_only=True,
    )
    hidden_sizes = tuple(int(value) for value in config.hidden_sizes)
    depths = tuple(int(value) for value in config.depths)
    if hidden_sizes != STUDENT_CHANNELS or depths != MIT_B2_DEPTHS:
        raise ValueError(
            "The local Hugging Face folder is not MiT-B2.\n"
            f"Expected hidden_sizes={STUDENT_CHANNELS}, "
            f"depths={MIT_B2_DEPTHS}\n"
            f"Got hidden_sizes={hidden_sizes}, depths={depths}"
        )

    config.drop_path_rate = float(drop_path_rate)
    config.output_hidden_states = True
    config.return_dict = True
    config.reshape_last_stage = True

    if pretrained:
        model = SegformerModel.from_pretrained(
            str(local_path),
            config=config,
            local_files_only=True,
        )
        text = "local pretrained weights"
    else:
        model = SegformerModel(config)
        text = "random initialization"

    print("=" * 100)
    print("MiT-B2 BACKBONE")
    print(f"Local folder: {local_path}")
    print(f"Weights: {text}")
    print("=" * 100)
    return model


class MiTB2Pyramid(nn.Module):
    def __init__(
        self,
        local_dir: str,
        pretrained: bool = True,
        drop_path_rate: float = 0.1,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.model = build_local_mit_b2(
            local_dir=local_dir,
            pretrained=bool(pretrained),
            drop_path_rate=float(drop_path_rate),
        )
        if gradient_checkpointing:
            if not hasattr(self.model, "gradient_checkpointing_enable"):
                raise RuntimeError(
                    "The installed transformers version does not support "
                    "SegFormer gradient checkpointing."
                )
            self.model.gradient_checkpointing_enable()

    @staticmethod
    def _to_bchw(
        feature: torch.Tensor,
        channels: int,
        reduction: int,
        input_size: Tuple[int, int],
    ) -> torch.Tensor:
        if feature.ndim == 4:
            if feature.shape[1] == int(channels):
                return feature.contiguous()
            if feature.shape[-1] == int(channels):
                return feature.permute(0, 3, 1, 2).contiguous()
            raise RuntimeError(
                f"Stage feature {tuple(feature.shape)} does not contain "
                f"the expected channel count {channels}."
            )

        if feature.ndim != 3:
            raise RuntimeError(
                "MiT hidden state must be BCHW or BNC, "
                f"got {tuple(feature.shape)}."
            )
        if feature.shape[-1] == int(channels):
            feature = feature.transpose(1, 2)
        elif feature.shape[1] != int(channels):
            raise RuntimeError(
                f"Stage token feature {tuple(feature.shape)} has an "
                f"unexpected channel count; expected {channels}."
            )

        batch, channel_count, token_count = feature.shape
        height = (int(input_size[0]) + int(reduction) - 1) // int(reduction)
        width = (int(input_size[1]) + int(reduction) - 1) // int(reduction)
        if height * width != token_count:
            raise RuntimeError(
                f"Cannot reshape {token_count} tokens into {height}x{width} "
                f"for input_size={input_size}, reduction={reduction}."
            )
        return feature.reshape(
            batch,
            channel_count,
            height,
            width,
        ).contiguous()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) < 4:
            raise RuntimeError(
                "SegformerModel did not return four stage hidden states."
            )
        stage_states = hidden_states[-4:]
        input_size = (int(x.shape[-2]), int(x.shape[-1]))
        return tuple(
            self._to_bchw(
                feature=feature,
                channels=channels,
                reduction=reduction,
                input_size=input_size,
            )
            for feature, channels, reduction in zip(
                stage_states,
                STUDENT_CHANNELS,
                STUDENT_REDUCTIONS,
            )
        )
