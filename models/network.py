# -*- coding: utf-8 -*-
"""Full SCDV -> RLGC -> F3-assisted progressive decoder model."""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from .backbone import (BACKBONE_DROP_PATH, BACKBONE_GRADIENT_CHECKPOINTING,
                       DEFAULT_MIT_B2_DIR, STUDENT_CHANNELS, MiTB2Pyramid)
from .decoder import F3AssistedProgressiveDecoder
from .scdv import SemanticContextualDiscrepancyVerification
from .rlgc import ReliabilityAwareLocalGlobalCalibration


class ForgeryLocalizationNet(nn.Module):
    def __init__(self, clip_feature_dim: int = 1024, *, feature_dim: int = 160,
                 semantic_reference_layer: int = 24, context_topk: int = 8,
                 reference_temperature: float = 0.10, reference_seed: int = 42,
                 semantic_temperature: float = 0.10, exclusion_radius: int = 1,
                 **kwargs) -> None:
        super().__init__()
        del kwargs
        self.semantic_reference_layer = int(semantic_reference_layer)
        self.backbone_name = "nvidia/mit-b2"
        self.feature_channels = list(STUDENT_CHANNELS)
        self.backbone = MiTB2Pyramid(
            local_dir=str(DEFAULT_MIT_B2_DIR),
            pretrained=bool(config["backbone_pretrained"]),
            drop_path_rate=float(BACKBONE_DROP_PATH),
            gradient_checkpointing=bool(BACKBONE_GRADIENT_CHECKPOINTING),
        )
        dim = int(feature_dim)
        self.scdv = SemanticContextualDiscrepancyVerification(
            in_channels=int(STUDENT_CHANNELS[3]), feature_dim=dim,
            clip_dim=int(clip_feature_dim), topk=int(context_topk),
            reference_temperature=float(reference_temperature),
            reference_seed=int(reference_seed),
            semantic_temperature=float(semantic_temperature),
            exclusion_radius=int(exclusion_radius),
            use_semantic_conditioning=True,
        )
        self.rlgc = ReliabilityAwareLocalGlobalCalibration(
            clip_dim=int(clip_feature_dim), feature_dim=dim, selected_layers=(12, 24),
            global_hidden_dim=512, dropout_p=0.10, global_scale_limit=0.25, global_bias_limit=0.25,
        )
        self.decoder = F3AssistedProgressiveDecoder(
            reason_channels=dim, f3_channels=int(STUDENT_CHANNELS[2]), decoder_dim=160, dropout_p=0.10,
        )

    @staticmethod
    def max_image_score(mask_logit: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(mask_logit.float()).flatten(1).amax(dim=1)

    def forward_backbone(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        features = self.backbone(x)
        if len(features) != 4:
            raise RuntimeError(f"MiT-B2 must return four levels, got {len(features)}")
        return features

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None,
                clip_features: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        del labels
        if not isinstance(clip_features, dict):
            raise ValueError("Expected frozen CLIP feature dictionary")
        patches = clip_features.get("patch_features", {})
        reference_patch = patches.get(self.semantic_reference_layer)
        if reference_patch is None:
            raise KeyError(f"Missing CLIP layer {self.semantic_reference_layer} Patch tokens")
        clip_mid = patches.get(12)
        clip_deep = patches.get(24)
        features = self.forward_backbone(x)
        f3 = features[2]
        f4 = features[3]
        state = self.scdv(f4, reference_patch, clip_mid, clip_deep)
        calibration = self.rlgc(
            state["verified_feature"], state["verification_evidence"], state["reference_variance"], clip_features
        )
        feature = calibration["calibrated_feature"]
        _, native_logit = self.decoder(feature, f3)
        mask_logit = F.interpolate(native_logit, size=x.shape[-2:], mode="bilinear", align_corners=False)
        outputs: Dict[str, torch.Tensor] = {
            "mask_logit": mask_logit,
            "image_score": self.max_image_score(mask_logit),
            "contextual_discrepancy": state["contextual_discrepancy"],
            "verification_evidence": state["verification_evidence"],
            "reference_variance": state["reference_variance"],
        }
        outputs["f_ctx"] = state["contextual_feature"]
        outputs["f_ver"] = state["verified_feature"]
        outputs["f_rec"] = calibration["rec_feature"]
        outputs["f_cal"] = feature
        outputs["reference_reliability_gate"] = calibration["reference_reliability_gate"]
        outputs["global_scale_mean"] = calibration["global_scale_mean"]
        outputs["global_bias_mean"] = calibration["global_bias_mean"]
        return outputs
