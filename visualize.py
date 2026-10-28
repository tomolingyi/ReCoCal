"""Visualize a trained localization model on an image or a directory of images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Image or directory; directories are scanned recursively.")
    source.add_argument("--smoke-test", action="store_true", help="Use a generated synthetic image to check execution.")
    parser.add_argument("--mask", type=Path, help="Optional GT image, or directory matching input relative stems.")
    parser.add_argument("--resolution", type=int, choices=(256, 512), default=512)
    parser.add_argument("--checkpoint", type=Path, help="Override the included trained student checkpoint.")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision, including CLIP fp16 weights.")
    parser.add_argument("--max-images", type=int, default=0, help="0 processes every image supplied.")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be in [0, 1]")
    if args.max_images < 0:
        parser.error("--max-images must be nonnegative")
    if args.smoke_test and args.mask:
        parser.error("--mask is not used with --smoke-test")
    return args


def seed_everything(seed=42):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def preprocess(image, size):
    """Match the original Albumentations 1.3.0 eval resize and normalization."""
    import cv2
    import numpy as np
    import torch
    rgb = cv2.resize(np.asarray(image, dtype=np.uint8), (size, size), interpolation=cv2.INTER_LINEAR)
    mean = np.array((0.485, 0.456, 0.406), dtype=np.float32) * np.float32(255.0)
    inv_std = np.float32(1.0) / (np.array((0.229, 0.224, 0.225), dtype=np.float32) * np.float32(255.0))
    normalized = (rgb.astype(np.float32) - mean) * inv_std
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)


def synthetic_image():
    import numpy as np
    from PIL import Image
    y, x = np.mgrid[:320, :480]
    array = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    array[100:210, 180:330] = array[100:210, 180:330, ::-1]
    return Image.fromarray(array)


def collect_inputs(args):
    if args.smoke_test:
        return [(None, Path("synthetic.png"))]
    source = args.input.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in EXTENSIONS:
            raise ValueError("Unsupported input image extension")
        files = [(source, Path(source.name))]
    elif source.is_dir():
        files = [(p, p.relative_to(source)) for p in sorted(source.rglob("*"))
                 if p.is_file() and p.suffix.lower() in EXTENSIONS
                 and not p.is_relative_to(args.output.expanduser().resolve())]
    else:
        raise FileNotFoundError(f"Input does not exist: {source}")
    if not files:
        raise ValueError("No input images found")
    return files[:args.max_images] if args.max_images else files


def find_mask(args, relative_path, count):
    if args.mask is None:
        return None
    root = args.mask.expanduser().resolve()
    if root.is_file():
        if count != 1:
            raise ValueError("A single --mask file requires exactly one input image")
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {root}")
    candidate = root / relative_path
    if candidate.is_file():
        return candidate
    matches = [p for p in candidate.parent.glob(candidate.stem + ".*") if p.suffix.lower() in EXTENSIONS]
    if len(matches) != 1:
        raise ValueError(f"Expected one matching mask for {relative_path.as_posix()}, found {len(matches)}")
    return matches[0]


def load_models(args, device):
    import torch
    from config import config
    from models import ForgeryLocalizationNet, build_paper_clip_encoder
    from models import clip_encoder as clip_module
    checkpoint = args.checkpoint or ROOT / "weights" / f"c2d_{args.resolution}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing student checkpoint: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved = payload.get("config", {})
    if saved.get("input_size") != args.resolution:
        raise ValueError("Checkpoint input_size differs from --resolution")
    for key in ("feature_dim", "semantic_reference_layer", "context_topk", "reference_temperature",
                "reference_seed", "semantic_temperature", "exclusion_radius"):
        if key in saved and saved[key] != config[key]:
            raise ValueError(f"Checkpoint/config mismatch for {key}")
    clip_module.PAPER_CLIP_USE_AMP = not args.no_amp
    clip_module.PAPER_CLIP_FP16_WEIGHTS = not args.no_amp
    # A strict full student checkpoint already contains every MiT-B2 backbone tensor.
    config["backbone_pretrained"] = False
    clip = build_paper_clip_encoder(device)
    model = ForgeryLocalizationNet(
        **{key: config[key] for key in ("clip_feature_dim", "feature_dim", "semantic_reference_layer",
            "context_topk", "reference_temperature", "reference_seed", "semantic_temperature", "exclusion_radius")}
    )
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()
    del payload
    return model, clip


def render_outputs(image, mask_path, outputs, sample_dir, threshold, plt):
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image
    sample_dir.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(image)
    hw = (image.height, image.width)
    probability = F.interpolate(outputs["mask_logit"].float().sigmoid(), size=hw,
                                mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    prediction = probability >= threshold
    overlay = rgb.astype(np.float32).copy()
    overlay[prediction] = 0.55 * overlay[prediction] + 0.45 * np.array((255, 0, 0))
    overlay = overlay.clip(0, 255).astype(np.uint8)
    image.save(sample_dir / "input.png")
    Image.fromarray(prediction.astype(np.uint8) * 255).save(sample_dir / "prediction.png")
    Image.fromarray(overlay).save(sample_dir / "overlay.png")
    np.save(sample_dir / "probability.npy", probability)
    plt.imsave(sample_dir / "probability.png", probability, cmap="turbo", vmin=0, vmax=1)
    panels = [(rgb, "Input", None), (prediction, "Prediction", "gray"),
              (overlay, "Overlay", None), (probability, "Probability", "turbo")]
    metrics = {}
    if mask_path:
        with Image.open(mask_path) as mask_image:
            gt = np.asarray(mask_image.convert("L").resize(image.size, Image.Resampling.NEAREST)) > 127
        Image.fromarray(gt.astype(np.uint8) * 255).save(sample_dir / "gt.png")
        panels.insert(1, (gt, "GT mask", "gray"))
        tp = int((gt & prediction).sum())
        fp = int((~gt & prediction).sum())
        fn = int((gt & ~prediction).sum())
        metrics = {"pixel_f1": 2 * tp / max(2 * tp + fp + fn, 1),
                   "pixel_iou": tp / max(tp + fp + fn, 1),
                   "metric_scope": "single_image_original_resolution"}

    def figure(items, name):
        fig, axes = plt.subplots(1, len(items), figsize=(2.6 * len(items), 2.8), squeeze=False)
        for ax, (data, title, cmap) in zip(axes[0], items):
            ax.imshow(data, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
            ax.set_title(title)
            ax.axis("off")
        fig.tight_layout(pad=0.15)
        fig.savefig(sample_dir / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(sample_dir / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    figure(panels, "localization")
    mechanism = [(rgb, "Input", None)]
    for key, title in (("f_ctx", "F_ctx"), ("f_ver", "F_ver"), ("f_rec", "F_rec"), ("f_cal", "F_cal")):
        # Same mean absolute channel response used in the original ranked visualizer.
        response = F.interpolate(outputs[key].float().abs().mean(1, keepdim=True), size=hw,
                                 mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        response = (response - response.min()) / max(float(response.max() - response.min()), 1e-8)
        plt.imsave(sample_dir / f"{key}.png", response, cmap="turbo", vmin=0, vmax=1)
        mechanism.append((response, title, "turbo"))
    figure(mechanism, "mechanism")
    return metrics


def main():
    args = parse_args()
    inputs = collect_inputs(args)
    masks = [find_mask(args, relative, len(inputs)) for _, relative in inputs]
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams["svg.fonttype"] = "none"
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    device = torch.device(device_name)
    seed_everything()
    model, clip = load_models(args, device)
    rows = []
    with torch.inference_mode():
        for (path, relative), mask in zip(inputs, masks):
            if path:
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
            else:
                image = synthetic_image()
            batch = preprocess(image, args.resolution).to(device)
            clip_features = clip.extract_features(batch)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda" and not args.no_amp):
                outputs = model(batch, clip_features=clip_features)
            if not torch.isfinite(outputs["mask_logit"]).all():
                raise FloatingPointError("Model produced nonfinite logits")
            identifier = relative.with_suffix("")
            # Include the extension in the hash to avoid same-stem image collisions.
            digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:8]
            sample_dir = args.output / identifier.parent / (identifier.name + "_" + digest)
            metrics = render_outputs(image, mask, outputs, sample_dir, args.threshold, plt)
            rows.append({"input": relative.as_posix(), "output": sample_dir.relative_to(args.output).as_posix(),
                         "image_score": float(outputs["image_score"].item()), **metrics})
            print(f"Saved {relative.as_posix()} -> {sample_dir}")
    metadata = {"resolution": args.resolution, "device": device.type, "threshold": args.threshold,
                "mixed_precision": device.type == "cuda" and not args.no_amp,
                "synthetic_smoke_test": args.smoke_test, "images": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
