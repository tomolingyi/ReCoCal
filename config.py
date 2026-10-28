"""Inference configuration; all included assets resolve relative to this package."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
config = {
    "mit_b2_dir": str(ROOT / "weights" / "mit_b2"),
    "clip_checkpoint": str(ROOT / "weights" / "clip_visual.pth"),
    "backbone_pretrained": False,
    "clip_feature_dim": 1024, "feature_dim": 160,
    "semantic_reference_layer": 24, "context_topk": 8,
    "reference_temperature": 0.10, "reference_seed": 42,
    "semantic_temperature": 0.10, "exclusion_radius": 1,
    "seed": 42,
}
