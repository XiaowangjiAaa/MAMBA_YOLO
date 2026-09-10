#!/usr/bin/env python3
"""Visualize segmentation predictions and calculate per-image metrics.

Supported sources:
  1. One image.
  2. A directory of images.
  3. An Ultralytics dataset YAML containing train/val/test entries.

Examples:
  python visualize_seg_results.py --weights best_map50.pt --source image.jpg
  python visualize_seg_results.py --weights best_map50.pt --source images/ --recursive
  python visualize_seg_results.py --model-yaml model.yaml --weights best_map50.pt \
      --source crack-seg/data.yaml --split val --device 0

The per-image AP values in ``per_image_metrics.csv`` are diagnostic image-level
AP values. The dataset-level AP values in ``summary_metrics.csv`` aggregate all
predictions before computing AP. They are useful for controlled comparisons made
with this script, but can differ slightly from Ultralytics' official validator
because post-processing and interpolation details may differ.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

try:
    import cv2
except ImportError:  # Keep --help usable in an environment missing optional runtime dependencies.
    cv2 = None


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
AP_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


@dataclass
class GroundTruth:
    masks: np.ndarray  # [N, H, W], bool
    classes: np.ndarray  # [N], int64
    label_path: Optional[Path]
    malformed_lines: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize YOLO segmentation predictions and export per-image/dataset metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="Ultralytics .pt or PyTorch .pth checkpoint")
    parser.add_argument(
        "--model-yaml",
        default=None,
        help="Optional model architecture YAML. Recommended for a raw state_dict .pth file.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image, image directory, image-list TXT, or dataset YAML (data.yaml/crack-seg.yaml)",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output-dir", default="visualization_results", help="Output directory")
    parser.add_argument("--device", default="0", help="CUDA device such as 0, 0,1, cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Inference floor used to retain the confidence curve for AP calculation",
    )
    parser.add_argument(
        "--visual-conf",
        type=float,
        default=0.25,
        help="Confidence threshold used for overlays and pixel/clDice metrics",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan a direct image directory")
    parser.add_argument("--alpha", type=float, default=0.42, help="Mask overlay opacity")
    parser.add_argument("--line-width", type=int, default=2)
    parser.add_argument(
        "--visual-style",
        choices=("instance", "diagnostic"),
        default="instance",
        help="instance draws mask/box/class/conf; diagnostic draws GT/prediction errors",
    )
    parser.add_argument("--save-json", action="store_true", help="Also save summary_metrics.json")
    parser.add_argument("--no-images", action="store_true", help="Calculate CSV metrics without saving overlays")
    return parser.parse_args()


def prepare_device_environment(device) -> Optional[str]:
    """Fix CUDA visibility before Ultralytics imports torch.

    PyTorch queues a CUDA capability check during import. Changing
    CUDA_VISIBLE_DEVICES after that import can make the queued check retain a
    stale device count (for example, checking device 1 after only device 0 is
    visible). This script imports Ultralytics lazily, so visibility can safely
    be fixed here first.
    """
    requested = str(device).strip().lower().replace("cuda:", "")
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    if requested == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif requested == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    elif requested and requested not in {"none", "mps", "mps:0"}:
        compact = requested.replace(" ", "")
        if all(part.isdigit() for part in compact.split(",")):
            os.environ["CUDA_VISIBLE_DEVICES"] = compact
    return previous


def is_dataset_yaml(path: Path) -> bool:
    if path.suffix.lower() not in {".yaml", ".yml"} or not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return False
    return isinstance(data, dict) and any(key in data for key in ("train", "val", "test"))


def expand_image_entry(entry: Path, recursive: bool, relative_base: Optional[Path] = None) -> List[Path]:
    """Expand one image, directory, glob, or TXT list into image paths."""
    entry = entry.expanduser()
    if not entry.is_absolute() and relative_base is not None:
        entry = (relative_base / entry).resolve()
    else:
        entry = entry.resolve()

    if entry.is_file() and entry.suffix.lower() in IMAGE_SUFFIXES:
        return [entry]
    if entry.is_file() and entry.suffix.lower() == ".txt":
        images: List[Path] = []
        with entry.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                candidate = Path(line).expanduser()
                if not candidate.is_absolute():
                    candidate = (entry.parent / candidate).resolve()
                images.extend(expand_image_entry(candidate, recursive=True))
        return images
    if entry.is_dir():
        iterator = entry.rglob("*") if recursive else entry.glob("*")
        return sorted(path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

    # Preserve wildcard characters that Path.resolve() may have retained.
    matches = [Path(match).resolve() for match in glob.glob(str(entry), recursive=True)]
    return sorted(path for path in matches if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def resolve_dataset_source(dataset_yaml: Path, split: str) -> Tuple[List[Path], Path, dict]:
    """Resolve an Ultralytics dataset YAML without relying on private loader APIs."""
    dataset_yaml = dataset_yaml.resolve()
    with dataset_yaml.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if split not in data or data[split] in (None, ""):
        raise ValueError(f"dataset YAML has no usable '{split}' entry: {dataset_yaml}")

    yaml_dir = dataset_yaml.parent
    root_value = data.get("path", "")
    dataset_root = Path(str(root_value)).expanduser() if root_value else yaml_dir
    if not dataset_root.is_absolute():
        dataset_root = (yaml_dir / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()

    entries = data[split] if isinstance(data[split], list) else [data[split]]
    images: List[Path] = []
    effective_root = dataset_root
    for value in entries:
        raw = Path(str(value)).expanduser()
        candidates = [raw] if raw.is_absolute() else [dataset_root / raw, yaml_dir / raw]
        selected = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not raw.is_absolute() and len(candidates) > 1 and selected == candidates[1]:
            # Some downloaded dataset YAMLs retain ``path: crack-seg`` even
            # when the YAML itself has already been placed inside crack-seg/.
            effective_root = yaml_dir
        images.extend(expand_image_entry(selected, recursive=True))
    return unique_existing_images(images), effective_root, data


def unique_existing_images(images: Iterable[Path]) -> List[Path]:
    seen = set()
    output = []
    for image in images:
        resolved = image.resolve()
        key = str(resolved).lower()
        if resolved.is_file() and resolved.suffix.lower() in IMAGE_SUFFIXES and key not in seen:
            seen.add(key)
            output.append(resolved)
    return output


def resolve_source(source: Path, split: str, recursive: bool) -> Tuple[List[Path], Optional[Path], Optional[dict]]:
    if is_dataset_yaml(source):
        images, root, data = resolve_dataset_source(source, split)
        return images, root, data
    images = unique_existing_images(expand_image_entry(source, recursive=recursive))
    return images, None, None


def candidate_label_paths(image_path: Path, dataset_root: Optional[Path]) -> List[Path]:
    """Generate label-path candidates for standard Ultralytics layouts."""
    candidates: List[Path] = []
    parts = list(image_path.parts)
    image_indices = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if image_indices:
        index = image_indices[-1]
        replaced = parts.copy()
        replaced[index] = "labels"
        candidates.append(Path(*replaced).with_suffix(".txt"))

    if dataset_root is not None:
        try:
            relative = image_path.relative_to(dataset_root)
            relative_parts = list(relative.parts)
            if relative_parts and relative_parts[0].lower() == "images":
                relative_parts[0] = "labels"
                candidates.append((dataset_root / Path(*relative_parts)).with_suffix(".txt"))
            candidates.append((dataset_root / "labels" / relative).with_suffix(".txt"))
        except ValueError:
            pass
    candidates.append(image_path.with_suffix(".txt"))
    return list(dict.fromkeys(path.resolve() for path in candidates))


def find_label_path(image_path: Path, dataset_root: Optional[Path]) -> Optional[Path]:
    return next((path for path in candidate_label_paths(image_path, dataset_root) if path.is_file()), None)


def load_yolo_segmentation_gt(
    image_path: Path,
    height: int,
    width: int,
    dataset_root: Optional[Path],
) -> GroundTruth:
    """Rasterize YOLO polygon labels; five-value box labels are also accepted."""
    label_path = find_label_path(image_path, dataset_root)
    if label_path is None:
        return GroundTruth(np.zeros((0, height, width), dtype=bool), np.zeros(0, dtype=np.int64), None)

    masks: List[np.ndarray] = []
    classes: List[int] = []
    malformed = 0
    with label_path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for raw_line in stream:
            values = raw_line.strip().split()
            if not values:
                continue
            try:
                class_id = int(float(values[0]))
                coordinates = np.asarray([float(value) for value in values[1:]], dtype=np.float32)
            except ValueError:
                malformed += 1
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            if coordinates.size >= 6 and coordinates.size % 2 == 0:
                polygon = coordinates.reshape(-1, 2)
                polygon[:, 0] = np.clip(polygon[:, 0] * width, 0, width - 1)
                polygon[:, 1] = np.clip(polygon[:, 1] * height, 0, height - 1)
                cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
            elif coordinates.size == 4:
                # Compatibility with box-only labels. Segmentation metrics for
                # these rows describe the rasterized rectangle, not a true mask.
                cx, cy, box_w, box_h = coordinates
                x1 = int(np.clip((cx - box_w / 2) * width, 0, width - 1))
                y1 = int(np.clip((cy - box_h / 2) * height, 0, height - 1))
                x2 = int(np.clip((cx + box_w / 2) * width, 0, width - 1))
                y2 = int(np.clip((cy + box_h / 2) * height, 0, height - 1))
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
            else:
                malformed += 1
                continue
            if mask.any():
                masks.append(mask.astype(bool))
                classes.append(class_id)
            else:
                malformed += 1
    stacked = np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool)
    return GroundTruth(stacked, np.asarray(classes, dtype=np.int64), label_path, malformed)


def load_model(weights: Path, model_yaml: Optional[Path]):
    """Load a full Ultralytics checkpoint or a raw state_dict with an architecture YAML."""
    from ultralytics import YOLO

    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if model_yaml is None:
        return YOLO(str(weights), task="segment"), "checkpoint architecture"

    model_yaml = model_yaml.resolve()
    if not model_yaml.is_file():
        raise FileNotFoundError(f"model YAML not found: {model_yaml}")
    model = YOLO(str(model_yaml), task="segment")
    try:
        model.load(str(weights))
        return model, "model YAML + Ultralytics checkpoint"
    except Exception as ultralytics_error:
        import torch

        try:
            try:
                checkpoint = torch.load(str(weights), map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(str(weights), map_location="cpu")
            if isinstance(checkpoint, dict):
                state = checkpoint.get("state_dict")
                if state is None:
                    state = checkpoint.get("model_state_dict")
                if state is None:
                    module = checkpoint.get("ema")
                    if module is None:
                        module = checkpoint.get("model")
                    state = module.state_dict() if hasattr(module, "state_dict") else checkpoint
            elif hasattr(checkpoint, "state_dict"):
                state = checkpoint.state_dict()
            else:
                raise TypeError(f"unsupported checkpoint object: {type(checkpoint).__name__}")
            # Distributed checkpoints commonly add ``module.``. Raw exports may
            # additionally contain one wrapper-level ``model.`` prefix. Select
            # the key layout that actually matches this architecture.
            cleaned = {
                (key[7:] if key.startswith("module.") else key): value
                for key, value in state.items()
                if hasattr(value, "shape")
            }
            target = model.model
            target_state = target.state_dict()
            variants = [cleaned]
            variants.append({(key[6:] if key.startswith("model.") else key): value for key, value in cleaned.items()})
            variants.append(
                {(key[10:] if key.startswith("_orig_mod.") else key): value for key, value in cleaned.items()}
            )

            def matching_tensors(candidate):
                return {
                    key: value
                    for key, value in candidate.items()
                    if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
                }

            filtered = max((matching_tensors(candidate) for candidate in variants), key=len)
            result = target.load_state_dict(filtered, strict=False)
            matched = len(filtered)
            if matched <= 0:
                raise RuntimeError("raw state_dict did not match any model parameter")
            return model, f"model YAML + raw state_dict ({matched} tensors matched)"
        except Exception as state_error:
            raise RuntimeError(
                f"Ultralytics load failed: {ultralytics_error}\nRaw state_dict load failed: {state_error}"
            ) from state_error


def result_to_predictions(result, height: int, width: int, mask_threshold: float):
    boxes = getattr(result, "boxes", None)
    masks_object = getattr(result, "masks", None)
    if boxes is None or len(boxes) == 0:
        return (
            np.zeros((0, height, width), dtype=bool),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
    classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
    confidences = boxes.conf.detach().cpu().numpy().astype(np.float32)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    if masks_object is None or masks_object.data is None:
        # This is a segmentation evaluator: box-only detections must not be fed
        # into mask matching, otherwise confidence and mask indices become stale.
        return (
            np.zeros((0, height, width), dtype=bool),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )

    # Ultralytics exposes mask polygons in original-image coordinates. Prefer
    # these over naively resizing masks from the letterboxed inference tensor.
    polygons = getattr(masks_object, "xy", None)
    rasterized = []
    if polygons is not None and len(polygons) == len(classes):
        for instance_polygons in polygons:
            mask = np.zeros((height, width), dtype=np.uint8)
            pieces = instance_polygons if isinstance(instance_polygons, (list, tuple)) else [instance_polygons]
            valid_pieces = []
            for piece in pieces:
                polygon = np.asarray(piece, dtype=np.float32).reshape(-1, 2)
                if len(polygon) >= 3:
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
                    valid_pieces.append(np.round(polygon).astype(np.int32))
            if valid_pieces:
                cv2.fillPoly(mask, valid_pieces, 1)
            rasterized.append(mask.astype(bool))

    if rasterized:
        pred_masks = np.stack(rasterized)
    else:
        raw_masks = masks_object.data.detach().float().cpu().numpy()
        resized = []
        for mask in raw_masks:
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            resized.append(mask >= mask_threshold)
        pred_masks = np.stack(resized) if resized else np.zeros((0, height, width), dtype=bool)
    count = min(len(pred_masks), len(classes))
    return pred_masks[:count], classes[:count], confidences[:count], xyxy[:count]


def read_image(path: Path) -> Optional[np.ndarray]:
    """Read paths containing non-ASCII characters on Windows as well as POSIX."""
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, ValueError):
        return None


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image with Unicode-path support and explicit failure reporting."""
    suffix = path.suffix.lower() or ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise OSError(f"OpenCV could not encode visualization as {suffix}: {path}")
    try:
        encoded.tofile(str(path))
    except OSError as error:
        raise OSError(f"failed to write visualization: {path}") from error


def mask_iou_matrix(pred_masks: np.ndarray, gt_masks: np.ndarray) -> np.ndarray:
    output = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    pred_areas = pred_masks.reshape(len(pred_masks), -1).sum(1) if len(pred_masks) else np.zeros(0)
    gt_areas = gt_masks.reshape(len(gt_masks), -1).sum(1) if len(gt_masks) else np.zeros(0)
    for pred_index, pred_mask in enumerate(pred_masks):
        for gt_index, gt_mask in enumerate(gt_masks):
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = pred_areas[pred_index] + gt_areas[gt_index] - intersection
            output[pred_index, gt_index] = intersection / union if union else 0.0
    return output


def match_predictions(
    ious: np.ndarray,
    pred_classes: np.ndarray,
    pred_confidences: np.ndarray,
    gt_classes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Greedy confidence-ordered, class-aware one-to-one mask matching."""
    true_positive = np.zeros(len(pred_confidences), dtype=bool)
    used_gt = set()
    for pred_index in np.argsort(-pred_confidences):
        candidates = [
            gt_index
            for gt_index in range(len(gt_classes))
            if gt_index not in used_gt
            and pred_classes[pred_index] == gt_classes[gt_index]
            and ious[pred_index, gt_index] >= threshold
        ]
        if candidates:
            gt_index = max(candidates, key=lambda index: ious[pred_index, index])
            used_gt.add(gt_index)
            true_positive[pred_index] = True
    return true_positive


def compute_ap(confidences: np.ndarray, true_positive: np.ndarray, gt_count: int) -> float:
    """COCO-style 101-point interpolated AP for one class/IoU threshold."""
    if gt_count <= 0:
        return math.nan
    if len(confidences) == 0:
        return 0.0
    order = np.argsort(-confidences)
    tp = true_positive[order].astype(np.float64)
    fp = 1.0 - tp
    recall = np.cumsum(tp) / gt_count
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp + fp), 1e-12)
    recall_grid = np.linspace(0.0, 1.0, 101)
    interpolated = np.zeros_like(recall_grid)
    for index, recall_level in enumerate(recall_grid):
        values = precision[recall >= recall_level]
        interpolated[index] = values.max() if values.size else 0.0
    return float(interpolated.mean())


def image_mask_ap(
    pred_masks: np.ndarray,
    pred_classes: np.ndarray,
    confidences: np.ndarray,
    gt_masks: np.ndarray,
    gt_classes: np.ndarray,
) -> Tuple[float, float, Dict[float, np.ndarray]]:
    ious = mask_iou_matrix(pred_masks, gt_masks)
    matches = {
        float(threshold): match_predictions(ious, pred_classes, confidences, gt_classes, float(threshold))
        for threshold in AP_THRESHOLDS
    }
    classes = np.unique(gt_classes)
    if len(classes) == 0:
        return math.nan, math.nan, matches
    per_threshold = []
    for threshold in AP_THRESHOLDS:
        class_aps = []
        tp = matches[float(threshold)]
        for class_id in classes:
            pred_select = pred_classes == class_id
            gt_count = int((gt_classes == class_id).sum())
            class_aps.append(compute_ap(confidences[pred_select], tp[pred_select], gt_count))
        per_threshold.append(float(np.nanmean(class_aps)))
    return per_threshold[0], float(np.mean(per_threshold)), matches


def binary_skeleton(mask: np.ndarray) -> np.ndarray:
    """Morphological skeleton with OpenCV only, avoiding a scikit-image dependency."""
    image = mask.astype(np.uint8)
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while image.any():
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
    return skeleton.astype(bool)


def union_metrics(pred_masks: np.ndarray, gt_masks: np.ndarray) -> dict:
    pred = pred_masks.any(axis=0) if len(pred_masks) else np.zeros(gt_masks.shape[1:], dtype=bool)
    gt = gt_masks.any(axis=0) if len(gt_masks) else np.zeros(pred.shape, dtype=bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn
    pred_area, gt_area = int(pred.sum()), int(gt.sum())
    both_empty = pred_area == 0 and gt_area == 0
    iou = 1.0 if both_empty else tp / union if union else 0.0
    dice_denominator = 2 * tp + fp + fn
    dice = 1.0 if both_empty else 2 * tp / dice_denominator if dice_denominator else 0.0
    precision = 1.0 if both_empty else tp / (tp + fp) if tp + fp else 0.0
    recall = 1.0 if both_empty else tp / (tp + fn) if tp + fn else 0.0

    pred_skeleton = binary_skeleton(pred)
    gt_skeleton = binary_skeleton(gt)
    topology_precision = (
        1.0 if not pred_skeleton.any() and not gt.any()
        else float(gt[pred_skeleton].mean()) if pred_skeleton.any()
        else 0.0
    )
    topology_sensitivity = (
        1.0 if not gt_skeleton.any() and not pred.any()
        else float(pred[gt_skeleton].mean()) if gt_skeleton.any()
        else 0.0
    )
    cldice = (
        2 * topology_precision * topology_sensitivity / (topology_precision + topology_sensitivity)
        if topology_precision + topology_sensitivity > 0
        else 0.0
    )
    return {
        "pixel_iou": iou,
        "pixel_dice": dice,
        "pixel_precision": precision,
        "pixel_recall": recall,
        "cldice": cldice,
        "pixel_tp": tp,
        "pixel_fp": fp,
        "pixel_fn": fn,
    }


class DatasetAPAccumulator:
    """Accumulate class-aware predictions for dataset-level mask AP."""

    def __init__(self):
        self.gt_counts: Dict[int, int] = defaultdict(int)
        self.records: Dict[float, Dict[int, List[Tuple[float, bool]]]] = {
            float(threshold): defaultdict(list) for threshold in AP_THRESHOLDS
        }

    def update(self, pred_classes, confidences, gt_classes, matches):
        for class_id in gt_classes:
            self.gt_counts[int(class_id)] += 1
        for threshold, tp in matches.items():
            for class_id, confidence, is_tp in zip(pred_classes, confidences, tp):
                self.records[threshold][int(class_id)].append((float(confidence), bool(is_tp)))

    def compute(self) -> Tuple[float, float, Dict[str, float]]:
        valid_classes = sorted(class_id for class_id, count in self.gt_counts.items() if count > 0)
        if not valid_classes:
            return math.nan, math.nan, {}
        threshold_maps = []
        details = {}
        for threshold in AP_THRESHOLDS:
            threshold = float(threshold)
            class_aps = []
            for class_id in valid_classes:
                entries = self.records[threshold].get(class_id, [])
                confidences = np.asarray([item[0] for item in entries], dtype=np.float32)
                true_positive = np.asarray([item[1] for item in entries], dtype=bool)
                ap = compute_ap(confidences, true_positive, self.gt_counts[class_id])
                class_aps.append(ap)
                details[f"mask_ap{int(round(threshold * 100)):02d}_class{class_id}"] = ap
            threshold_maps.append(float(np.nanmean(class_aps)))
        return threshold_maps[0], float(np.mean(threshold_maps)), details


INSTANCE_COLORS = (
    (255, 42, 4),    # Ultralytics-like blue in OpenCV BGR order
    (235, 219, 11),
    (0, 219, 255),
    (0, 212, 187),
    (255, 111, 221),
    (79, 68, 255),
    (138, 0, 255),
    (255, 178, 29),
)


def class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def instance_visualization(
    image: np.ndarray,
    pred_masks: np.ndarray,
    pred_classes: np.ndarray,
    confidences: np.ndarray,
    boxes: np.ndarray,
    names,
    alpha: float,
    line_width: int,
) -> np.ndarray:
    """Draw standard instance-segmentation masks, boxes, class names, and confidence scores."""
    canvas = image.copy()
    line_width = max(int(line_width), 1)

    # Blend every instance independently so different classes remain visually separable.
    for index, mask in enumerate(pred_masks):
        class_id = int(pred_classes[index])
        color = INSTANCE_COLORS[class_id % len(INSTANCE_COLORS)]
        active = mask.astype(bool)
        if active.any():
            color_layer = np.empty_like(canvas)
            color_layer[:] = color
            blended = cv2.addWeighted(canvas, 1.0 - alpha, color_layer, alpha, 0)
            canvas[active] = blended[active]
            contours, _ = cv2.findContours(active.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, line_width)

    for index, box in enumerate(boxes):
        class_id = int(pred_classes[index])
        color = INSTANCE_COLORS[class_id % len(INSTANCE_COLORS)]
        x1, y1, x2, y2 = box.round().astype(int)
        x1 = int(np.clip(x1, 0, image.shape[1] - 1))
        y1 = int(np.clip(y1, 0, image.shape[0] - 1))
        x2 = int(np.clip(x2, 0, image.shape[1] - 1))
        y2 = int(np.clip(y2, 0, image.shape[0] - 1))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width)

        label = f"{class_name(names, class_id)} {float(confidences[index]):.2f}"
        font_scale = max(0.45, line_width * 0.22)
        text_thickness = max(1, line_width - 1)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        label_top = y1 - text_height - baseline - 4
        if label_top < 0:
            label_top = y1
        label_bottom = min(label_top + text_height + baseline + 4, image.shape[0] - 1)
        label_right = min(x1 + text_width + 6, image.shape[1] - 1)
        cv2.rectangle(canvas, (x1, label_top), (label_right, label_bottom), color, thickness=-1)
        text_y = min(label_top + text_height + 2, label_bottom - baseline)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )
    return canvas


def overlay_visualization(
    image: np.ndarray,
    pred_masks: np.ndarray,
    pred_classes: np.ndarray,
    confidences: np.ndarray,
    boxes: np.ndarray,
    gt_masks: np.ndarray,
    metrics: Optional[dict],
    alpha: float,
    line_width: int,
) -> np.ndarray:
    """Draw GT-only green, prediction-only red, and overlap yellow."""
    canvas = image.copy()
    pred_union = pred_masks.any(axis=0) if len(pred_masks) else np.zeros(image.shape[:2], dtype=bool)
    gt_union = gt_masks.any(axis=0) if len(gt_masks) else np.zeros(image.shape[:2], dtype=bool)
    gt_only = gt_union & ~pred_union
    pred_only = pred_union & ~gt_union
    overlap = pred_union & gt_union
    color_layer = np.zeros_like(canvas)
    color_layer[gt_only] = (0, 210, 0)
    color_layer[pred_only] = (0, 0, 255)
    color_layer[overlap] = (0, 230, 230)
    active = gt_union | pred_union
    canvas[active] = cv2.addWeighted(canvas, 1.0 - alpha, color_layer, alpha, 0)[active]

    for mask in gt_masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 0), line_width)
    for index, mask in enumerate(pred_masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 0, 255), line_width)
        if index < len(boxes):
            x1, y1, x2, y2 = boxes[index].round().astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), line_width)
            text = f"P c{int(pred_classes[index])} {float(confidences[index]):.2f}"
            cv2.putText(canvas, text, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    legend = "GT:green  Pred:red  Overlap:yellow"
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 460), 50), (20, 20, 20), -1)
    cv2.putText(canvas, legend, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if metrics is not None:
        text = (
            f"AP50 {metrics['mask_ap50']:.3f}  IoU {metrics['pixel_iou']:.3f}  "
            f"Dice {metrics['pixel_dice']:.3f}  clDice {metrics['cldice']:.3f}"
        )
        cv2.putText(canvas, text, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
    return canvas


def finite_or_blank(value):
    if value is None:
        return ""
    try:
        return "" if not math.isfinite(float(value)) else float(value)
    except (TypeError, ValueError):
        return value


def nanmean(rows: Sequence[dict], key: str) -> float:
    values = np.asarray([row.get(key, math.nan) for row in rows], dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else math.nan


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: finite_or_blank(row.get(key, "")) for key in fieldnames})


def main() -> int:
    args = parse_args()
    previous_cuda_visible = prepare_device_environment(args.device)
    if cv2 is None:
        raise ModuleNotFoundError(
            "OpenCV is required. Activate the project environment or install opencv-python>=4.6.0."
        )
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.visual_conf <= 1.0:
        raise ValueError("--conf and --visual-conf must be within [0, 1]")
    if args.conf > args.visual_conf:
        print(
            "Warning: --conf is greater than --visual-conf; detections below --conf cannot be visualized.",
            file=sys.stderr,
        )
    source = Path(args.source).expanduser()
    weights = Path(args.weights).expanduser()
    model_yaml = Path(args.model_yaml).expanduser() if args.model_yaml else None
    output_dir = Path(args.output_dir).expanduser().resolve()
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    dataset_mode = is_dataset_yaml(source)
    images, dataset_root, dataset_config = resolve_source(source, args.split, args.recursive)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError(f"no supported images resolved from: {source}")

    print(f"Source:  {source.resolve()}")
    print(f"Mode:    {'dataset YAML / ' + args.split if dataset_mode else 'image source'}")
    print(f"Images:  {len(images)}")
    print(f"Output:  {output_dir}")
    if str(args.device).lower() != "cpu":
        print(
            f"CUDA:    requested={args.device}, "
            f"visible={os.environ.get('CUDA_VISIBLE_DEVICES', '<unchanged>')} "
            f"(previous={previous_cuda_visible or '<unset>'})"
        )
    model, load_description = load_model(weights, model_yaml)
    print(f"Model:   {load_description}")

    predict_kwargs = dict(
        source=[str(path) for path in images],
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        verbose=False,
        save=False,
    )
    results = model.predict(**predict_kwargs)
    rows: List[dict] = []
    accumulator = DatasetAPAccumulator()
    global_tp = global_fp = global_fn = 0
    started = time.perf_counter()

    for index, result in enumerate(results, start=1):
        result_path = getattr(result, "path", "")
        image_path = Path(result_path).expanduser().resolve() if result_path else images[index - 1]
        original = getattr(result, "orig_img", None)
        image = original.copy() if isinstance(original, np.ndarray) else read_image(image_path)
        if image is None:
            print(f"[{index}/{len(images)}] unreadable: {image_path}", file=sys.stderr)
            continue
        height, width = image.shape[:2]
        pred_masks, pred_classes, confidences, boxes = result_to_predictions(
            result, height, width, args.mask_threshold
        )
        visual_select = confidences >= args.visual_conf
        visual_masks = pred_masks[visual_select]
        visual_classes = pred_classes[visual_select]
        visual_confidences = confidences[visual_select]
        visual_boxes = boxes[visual_select]
        speed = getattr(result, "speed", {}) or {}
        inference_ms = float(speed.get("inference", math.nan))
        gt = load_yolo_segmentation_gt(image_path, height, width, dataset_root) if dataset_mode else None

        row = {
            "index": index,
            "image": image_path.name,
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "label_path": str(gt.label_path) if gt and gt.label_path else "",
            "label_found": bool(gt and gt.label_path),
            "malformed_label_lines": gt.malformed_lines if gt else 0,
            "num_gt": len(gt.masks) if gt else "",
            "num_pred": len(pred_masks),
            "num_pred_visual": len(visual_masks),
            "mean_confidence": float(confidences.mean()) if len(confidences) else 0.0,
            "max_confidence": float(confidences.max()) if len(confidences) else 0.0,
            "inference_ms": inference_ms,
        }
        visual_metrics = None
        if gt is not None:
            image_ap50, image_map, matches = image_mask_ap(
                pred_masks, pred_classes, confidences, gt.masks, gt.classes
            )
            pixel = union_metrics(visual_masks, gt.masks)
            row.update(pixel)
            row.update({"mask_ap50": image_ap50, "mask_map50_95": image_map})
            visual_metrics = row
            accumulator.update(pred_classes, confidences, gt.classes, matches)
            global_tp += pixel["pixel_tp"]
            global_fp += pixel["pixel_fp"]
            global_fn += pixel["pixel_fn"]

        if not args.no_images:
            if args.visual_style == "instance":
                names = getattr(result, "names", None)
                if names is None and dataset_config is not None:
                    names = dataset_config.get("names", {})
                visual = instance_visualization(
                    image,
                    visual_masks,
                    visual_classes,
                    visual_confidences,
                    visual_boxes,
                    names,
                    args.alpha,
                    args.line_width,
                )
            else:
                visual = overlay_visualization(
                    image,
                    visual_masks,
                    visual_classes,
                    visual_confidences,
                    visual_boxes,
                    gt.masks if gt else np.zeros((0, height, width), dtype=bool),
                    visual_metrics,
                    args.alpha,
                    args.line_width,
                )
            # PNG preserves thin crack-mask boundaries better than lossy JPEG.
            output_name = f"{index:06d}_{image_path.name}.png"
            output_path = image_dir / output_name
            write_image(output_path, visual)
            row["visualization"] = str(output_path)
        rows.append(row)
        if index == 1 or index % 25 == 0 or index == len(images):
            print(
                f"[{index}/{len(images)}] {image_path.name}: "
                f"pred_ap={len(pred_masks)}, pred_visual={len(visual_masks)}"
            )

    if not rows:
        raise RuntimeError("no image was processed successfully")

    per_image_fields = [
        "index", "image", "image_path", "width", "height", "label_path", "label_found",
        "malformed_label_lines", "num_gt", "num_pred", "num_pred_visual", "mean_confidence", "max_confidence",
        "mask_ap50", "mask_map50_95", "pixel_iou", "pixel_dice", "pixel_precision",
        "pixel_recall", "cldice", "pixel_tp", "pixel_fp", "pixel_fn", "inference_ms",
        "visualization",
    ]
    per_image_csv = output_dir / "per_image_metrics.csv"
    write_csv(per_image_csv, rows, per_image_fields)

    elapsed = time.perf_counter() - started
    summary = {
        "weights": str(weights.resolve()),
        "model_yaml": str(model_yaml.resolve()) if model_yaml else "",
        "source": str(source.resolve()),
        "split": args.split if dataset_mode else "",
        "metric_implementation": "custom class-aware mask AP; use official model.val for paper headline metrics",
        "ap_conf_floor": args.conf,
        "visual_and_pixel_conf": args.visual_conf,
        "seed_note": "metrics are deterministic for fixed weights/input settings",
        "images": len(rows),
        "images_with_labels": sum(bool(row.get("label_found")) for row in rows),
        "images_with_gt": sum(int(row.get("num_gt", 0) or 0) > 0 for row in rows),
        "missing_label_files": sum(not bool(row.get("label_found")) for row in rows) if dataset_mode else "",
        "total_gt_instances": sum(int(row.get("num_gt", 0) or 0) for row in rows),
        "total_pred_instances": sum(int(row.get("num_pred", 0) or 0) for row in rows),
        "mean_inference_ms": nanmean(rows, "inference_ms"),
        "wall_time_seconds": elapsed,
    }
    if dataset_mode:
        dataset_ap50, dataset_map, details = accumulator.compute()
        both_empty = global_tp + global_fp + global_fn == 0
        positive_rows = [row for row in rows if int(row.get("num_gt", 0) or 0) > 0]
        summary.update(
            {
                "dataset_mask_map50": dataset_ap50,
                "dataset_mask_map50_95": dataset_map,
                "macro_pixel_iou": nanmean(rows, "pixel_iou"),
                "macro_pixel_dice": nanmean(rows, "pixel_dice"),
                "macro_pixel_precision": nanmean(rows, "pixel_precision"),
                "macro_pixel_recall": nanmean(rows, "pixel_recall"),
                "macro_cldice": nanmean(rows, "cldice"),
                "positive_macro_pixel_iou": nanmean(positive_rows, "pixel_iou"),
                "positive_macro_pixel_dice": nanmean(positive_rows, "pixel_dice"),
                "positive_macro_cldice": nanmean(positive_rows, "cldice"),
                "micro_pixel_iou": 1.0 if both_empty else global_tp / max(global_tp + global_fp + global_fn, 1),
                "micro_pixel_dice": 1.0 if both_empty else 2 * global_tp / max(2 * global_tp + global_fp + global_fn, 1),
                **details,
            }
        )
        if summary["images_with_labels"] == 0:
            print(
                "Warning: no label TXT was found. Check dataset path/images/labels layout before trusting metrics.",
                file=sys.stderr,
            )
    summary_csv = output_dir / "summary_metrics.csv"
    write_csv(summary_csv, [summary], list(summary.keys()))
    if args.save_json:
        with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as stream:
            json.dump({key: finite_or_blank(value) for key, value in summary.items()}, stream, indent=2, ensure_ascii=False)

    print(f"Per-image CSV: {per_image_csv}")
    print(f"Summary CSV:   {summary_csv}")
    if dataset_mode:
        print(f"Mask mAP50:    {summary['dataset_mask_map50']:.4f}")
        print(f"Mask mAP50-95: {summary['dataset_mask_map50_95']:.4f}")
        print(f"Macro clDice:  {summary['macro_cldice']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
