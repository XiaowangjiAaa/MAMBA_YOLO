#!/usr/bin/env python3
"""Paired feature-map comparison for a baseline and a crack-path model.

Both models receive the exact same 640x640 letterboxed tensor. The script
captures semantically corresponding layers (P2/P3/P4/P5 by default), saves
spatial-energy heatmaps and their difference, and optionally exports the
proposed model's crack probability/orientation/connectivity/path caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from visualize_seg_results import (
    is_dataset_yaml,
    load_yolo_segmentation_gt,
    prepare_device_environment,
    read_image,
    resolve_source,
    write_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare corresponding intermediate features with and without the crack-path module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-yaml", required=True)
    parser.add_argument("--baseline-weights", required=True)
    parser.add_argument("--proposed-yaml", required=True)
    parser.add_argument("--proposed-weights", required=True)
    parser.add_argument("--source", required=True, help="Image, directory, TXT list, or dataset YAML")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--layers", default="2,4,6,8", help="Corresponding top-level YAML layer indices")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-images", type=int, default=1, help="0 means all resolved images")
    parser.add_argument("--image-index", type=int, default=0, help="Start index in the resolved source")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--reduce", choices=("l2", "mean_abs", "max_abs"), default="l2")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--alpha", type=float, default=0.46)
    parser.add_argument("--save-npy", action="store_true")
    parser.add_argument("--no-internal", action="store_true", help="Do not export proposed p/o/c/path maps")
    parser.add_argument("--output-dir", default="feature_comparison")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.imgsz <= 0 or args.max_images < 0 or args.image_index < 0:
        raise ValueError("--imgsz must be positive; image indices/counts cannot be negative")
    if not 50.0 < args.percentile <= 100.0:
        raise ValueError("--percentile must be in (50, 100]")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")


def sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")


def add_label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (18, 18, 18), -1)
    cv2.putText(output, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def extract_spatial_tensor(value: Any):
    """Return the first BCHW tensor from a nested layer output."""
    import torch

    if torch.is_tensor(value):
        return value if value.ndim == 4 and min(value.shape[-2:]) > 1 else None
    if isinstance(value, dict):
        for item in value.values():
            found = extract_spatial_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = extract_spatial_tensor(item)
            if found is not None:
                return found
    return None


def capture_hook(storage: Dict[str, Any], name: str):
    def hook(_module, _inputs, output):
        tensor = extract_spatial_tensor(output)
        if tensor is not None:
            storage[name] = tensor.detach().float().cpu()

    return hook


def reduce_feature(feature, method: str) -> np.ndarray:
    feature = feature[0]
    if method == "l2":
        heat = feature.square().mean(dim=0).sqrt()
    elif method == "max_abs":
        heat = feature.abs().amax(dim=0)
    else:
        heat = feature.abs().mean(dim=0)
    return heat.numpy().astype(np.float32)


def robust_normalize(array: np.ndarray, percentile: float) -> np.ndarray:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    tail = max(0.0, 100.0 - percentile)
    low, high = np.percentile(finite, [tail, percentile])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def shared_normalize(first: np.ndarray, second: np.ndarray, percentile: float) -> Tuple[np.ndarray, np.ndarray]:
    joined = np.concatenate((first.reshape(-1), second.reshape(-1)))
    finite = joined[np.isfinite(joined)]
    if finite.size == 0:
        return np.zeros_like(first), np.zeros_like(second)
    tail = max(0.0, 100.0 - percentile)
    low, high = np.percentile(finite, [tail, percentile])
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    scale = max(high - low, 1e-12)
    return np.clip((first - low) / scale, 0, 1), np.clip((second - low) / scale, 0, 1)


def color_heat(normalized: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(normalized, size, interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(np.round(resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def heat_overlay(image: np.ndarray, normalized: np.ndarray, alpha: float) -> np.ndarray:
    color = color_heat(normalized, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0.0)


def signed_difference_image(difference: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    difference = cv2.resize(difference.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
    maximum = max(float(np.percentile(np.abs(difference), 99.0)), 1e-8)
    normalized = np.clip(difference / maximum, -1.0, 1.0)
    output = np.full((*normalized.shape, 3), 235, dtype=np.uint8)
    positive = normalized > 0
    negative = normalized < 0
    strength = np.abs(normalized)
    output[..., 0] = np.where(positive, 235 * (1 - strength), 235).astype(np.uint8)
    output[..., 1] = (235 * (1 - strength)).astype(np.uint8)
    output[..., 2] = np.where(negative, 235 * (1 - strength), 235).astype(np.uint8)
    return output


def preprocess(image: np.ndarray, imgsz: int, device, torch):
    """Create one shared centered-letterbox tensor and masks for valid (non-padding) pixels."""
    height, width = image.shape[:2]
    ratio = min(imgsz / height, imgsz / width)
    resized_width, resized_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (imgsz - resized_width) // 2
    top = (imgsz - resized_height) // 2
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    valid = np.zeros((imgsz, imgsz), dtype=bool)
    valid[top:top + resized_height, left:left + resized_width] = True
    rgb = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1))
    tensor = torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    meta = (ratio, left, top, resized_width, resized_height)
    return canvas, tensor, valid, meta


def letterbox_mask(mask: np.ndarray, imgsz: int, meta) -> np.ndarray:
    _, left, top, width, height = meta
    resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    output = np.zeros((imgsz, imgsz), dtype=bool)
    output[top:top + height, left:left + width] = resized
    return output


def focus_statistics(heat: np.ndarray, gt_mask: Optional[np.ndarray], valid: np.ndarray) -> Dict[str, float]:
    heat = cv2.resize(heat, (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_LINEAR)
    if gt_mask is None or not gt_mask.any():
        return {"foreground_mean": math.nan, "background_mean": math.nan, "focus_ratio": math.nan}
    foreground = gt_mask & valid
    background = (~gt_mask) & valid
    fg_mean = float(heat[foreground].mean()) if foreground.any() else math.nan
    bg_mean = float(heat[background].mean()) if background.any() else math.nan
    ratio = fg_mean / max(bg_mean, 1e-12) if math.isfinite(fg_mean) and math.isfinite(bg_mean) else math.nan
    return {"foreground_mean": fg_mean, "background_mean": bg_mean, "focus_ratio": ratio}


def correlation(first: np.ndarray, second: np.ndarray, valid: np.ndarray) -> float:
    first = cv2.resize(first, (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_LINEAR)[valid]
    second = cv2.resize(second, (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_LINEAR)[valid]
    if first.size < 2 or float(first.std()) < 1e-12 or float(second.std()) < 1e-12:
        return math.nan
    return float(np.corrcoef(first, second)[0, 1])


def resolve_layer_names(core_model, selectors: str) -> List[str]:
    modules = dict(core_model.named_modules())
    selected = []
    for token in selectors.split(","):
        token = token.strip()
        if not token:
            continue
        name = f"model.{token}" if token.isdigit() else token
        if name not in modules:
            raise ValueError(f"layer '{token}' was not found; expected names such as model.2/model.4")
        selected.append(name)
    if not selected:
        raise ValueError("--layers selected no modules")
    return selected


def build_model(yaml_path: Path, weights_path: Path, device):
    from ultralytics import YOLO
    from visualize_model_features import load_matching_weights

    wrapper = YOLO(str(yaml_path), task="segment")
    matched, total = load_matching_weights(wrapper.model, weights_path)
    if matched / max(total, 1) < 0.90:
        print(f"[WARN] Only {matched}/{total} state tensors matched for {yaml_path.name}.")
    wrapper.model.to(device).eval()
    return wrapper.model, matched, total


def run_with_hooks(torch, model, layer_names: Sequence[str], tensor) -> Dict[str, Any]:
    storage: Dict[str, Any] = {}
    modules = dict(model.named_modules())
    handles = [modules[name].register_forward_hook(capture_hook(storage, name)) for name in layer_names]
    try:
        with torch.inference_mode():
            model(tensor)
    finally:
        for handle in handles:
            handle.remove()
    return storage


def write_metrics(path: Path, rows: Sequence[Dict]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    validate_args(args)
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required. Activate the training environment or install opencv-python>=4.6.0."
        )
    prepare_device_environment(args.device)

    import torch
    from ultralytics.utils.torch_utils import select_device
    import visualize_model_features as single_visualizer

    paths = {
        "baseline_yaml": Path(args.baseline_yaml).expanduser().resolve(),
        "baseline_weights": Path(args.baseline_weights).expanduser().resolve(),
        "proposed_yaml": Path(args.proposed_yaml).expanduser().resolve(),
        "proposed_weights": Path(args.proposed_weights).expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} not found: {path}")

    source = Path(args.source).expanduser()
    dataset_mode = is_dataset_yaml(source)
    images, dataset_root, _ = resolve_source(source, args.split, args.recursive)
    if args.image_index >= len(images):
        raise IndexError(f"--image-index {args.image_index} is outside {len(images)} resolved images")
    images = images[args.image_index:]
    if args.max_images > 0:
        images = images[:args.max_images]
    if not images:
        raise FileNotFoundError(f"no images resolved from {source}")

    device = select_device(args.device, batch=1, verbose=True)
    baseline, baseline_matched, baseline_total = build_model(
        paths["baseline_yaml"], paths["baseline_weights"], device
    )
    proposed, proposed_matched, proposed_total = build_model(
        paths["proposed_yaml"], paths["proposed_weights"], device
    )
    baseline_layers = resolve_layer_names(baseline, args.layers)
    proposed_layers = resolve_layer_names(proposed, args.layers)
    if len(baseline_layers) != len(proposed_layers):
        raise RuntimeError("baseline and proposed layer selections have different lengths")

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    for image_number, image_path in enumerate(images, start=args.image_index):
        image = read_image(image_path)
        if image is None:
            print(f"[WARN] unreadable image: {image_path}")
            continue
        input_bgr, tensor, valid_mask, letterbox_meta = preprocess(image, args.imgsz, device, torch)
        gt_mask = None
        if dataset_mode:
            gt = load_yolo_segmentation_gt(image_path, image.shape[0], image.shape[1], dataset_root)
            union = gt.masks.any(axis=0) if len(gt.masks) else np.zeros(image.shape[:2], dtype=bool)
            gt_mask = letterbox_mask(union, args.imgsz, letterbox_meta)

        baseline_capture = run_with_hooks(torch, baseline, baseline_layers, tensor)
        proposed_capture = run_with_hooks(torch, proposed, proposed_layers, tensor)
        image_dir = output_root / f"{image_number:05d}_{sanitize(image_path.stem)}"
        image_dir.mkdir(parents=True, exist_ok=True)
        write_image(image_dir / "input_letterbox.png", input_bgr)
        if gt_mask is not None:
            gt_view = input_bgr.copy()
            green = np.zeros_like(gt_view)
            green[:] = (0, 220, 0)
            gt_view[gt_mask] = cv2.addWeighted(gt_view, 0.55, green, 0.45, 0)[gt_mask]
            write_image(image_dir / "ground_truth.png", gt_view)

        for baseline_name, proposed_name in zip(baseline_layers, proposed_layers):
            if baseline_name not in baseline_capture or proposed_name not in proposed_capture:
                print(f"[WARN] missing captured output for {baseline_name}/{proposed_name}")
                continue
            baseline_feature = baseline_capture[baseline_name]
            proposed_feature = proposed_capture[proposed_name]
            baseline_heat = reduce_feature(baseline_feature, args.reduce)
            proposed_heat = reduce_feature(proposed_feature, args.reduce)
            baseline_norm = robust_normalize(baseline_heat, args.percentile)
            proposed_norm = robust_normalize(proposed_heat, args.percentile)
            baseline_shared, proposed_shared = shared_normalize(
                baseline_heat, proposed_heat, args.percentile
            )
            target_size = (input_bgr.shape[1], input_bgr.shape[0])
            base_resized = cv2.resize(baseline_norm, target_size, interpolation=cv2.INTER_LINEAR)
            prop_resized = cv2.resize(proposed_norm, target_size, interpolation=cv2.INTER_LINEAR)
            difference = prop_resized - base_resized

            layer_id = sanitize(baseline_name)
            comparison = np.concatenate(
                (
                    add_label(input_bgr, "same input"),
                    add_label(heat_overlay(input_bgr, baseline_norm, args.alpha), f"baseline {baseline_name}"),
                    add_label(heat_overlay(input_bgr, proposed_norm, args.alpha), f"proposed {proposed_name}"),
                    add_label(signed_difference_image(difference, target_size), "difference: red + / blue -"),
                ),
                axis=1,
            )
            write_image(image_dir / f"{layer_id}_normalized_comparison.png", comparison)
            shared_panel = np.concatenate(
                (
                    add_label(input_bgr, "same input"),
                    add_label(heat_overlay(input_bgr, baseline_shared, args.alpha), "baseline | shared scale"),
                    add_label(heat_overlay(input_bgr, proposed_shared, args.alpha), "proposed | shared scale"),
                ),
                axis=1,
            )
            write_image(image_dir / f"{layer_id}_shared_scale.png", shared_panel)
            if args.save_npy:
                np.save(image_dir / f"{layer_id}_baseline_feature.npy", baseline_feature[0].numpy())
                np.save(image_dir / f"{layer_id}_proposed_feature.npy", proposed_feature[0].numpy())

            baseline_focus = focus_statistics(baseline_heat, gt_mask, valid_mask)
            proposed_focus = focus_statistics(proposed_heat, gt_mask, valid_mask)
            metric_rows.append(
                {
                    "image": str(image_path),
                    "layer": baseline_name,
                    "baseline_shape": str(tuple(baseline_feature.shape)),
                    "proposed_shape": str(tuple(proposed_feature.shape)),
                    "baseline_mean_energy": float(baseline_heat.mean()),
                    "proposed_mean_energy": float(proposed_heat.mean()),
                    "baseline_focus_ratio": baseline_focus["focus_ratio"],
                    "proposed_focus_ratio": proposed_focus["focus_ratio"],
                    "focus_ratio_delta": proposed_focus["focus_ratio"] - baseline_focus["focus_ratio"],
                    "normalized_map_correlation": correlation(baseline_norm, proposed_norm, valid_mask),
                    "normalized_map_mae": float(np.abs(difference[valid_mask]).mean()),
                }
            )

        if not args.no_internal:
            internal_dir = image_dir / "proposed_internal"
            internal_dir.mkdir(parents=True, exist_ok=True)
            internal_panels, internal_records = single_visualizer.save_structure_maps(
                proposed, input_bgr, internal_dir, args.alpha, args.save_npy
            )
            single_visualizer.save_contact_sheet(internal_panels, internal_dir / "internal_contact_sheet.png")
            (internal_dir / "internal_manifest.json").write_text(
                json.dumps(internal_records, indent=2), encoding="utf-8"
            )
        print(f"[{len(metric_rows)} feature rows] saved {image_path.name}")

    if not metric_rows:
        raise RuntimeError("no paired feature maps were captured")
    write_metrics(output_root / "feature_focus_metrics.csv", metric_rows)
    metadata = {
        "baseline_yaml": str(paths["baseline_yaml"]),
        "baseline_weights": str(paths["baseline_weights"]),
        "proposed_yaml": str(paths["proposed_yaml"]),
        "proposed_weights": str(paths["proposed_weights"]),
        "source": str(source.resolve()),
        "split": args.split if dataset_mode else "",
        "imgsz": args.imgsz,
        "layers": baseline_layers,
        "reduction": args.reduce,
        "normalization_note": "normalized panels use per-model robust scaling; shared panels use one pairwise scale",
        "baseline_matched_state_tensors": f"{baseline_matched}/{baseline_total}",
        "proposed_matched_state_tensors": f"{proposed_matched}/{proposed_total}",
        "interpretation_warning": (
            "Feature channels from separately trained models are not one-to-one aligned. "
            "Use spatial energy/focus trends and internal p/o/c/path maps, not individual-channel subtraction, for claims."
        ),
    }
    (output_root / "comparison_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[DONE] Paired feature comparison: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
