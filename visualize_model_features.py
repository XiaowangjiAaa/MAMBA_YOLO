"""Visualize intermediate feature and crack-structure maps from any supported YOLO YAML.

Examples:
    python visualize_model_features.py --model path/to/model.yaml --weights path/to/best.pt --source image.jpg --list-layers
    python visualize_model_features.py --model path/to/model.yaml --weights path/to/best.pt --source image.jpg --layers 3,5,7,15,18,21
    python visualize_model_features.py --model path/to/model.yaml --weights path/to/best.pth --source image.jpg --layers "type:VSSBlock"
    python visualize_model_features.py --model path/to/model.yaml --weights path/to/best.pt --source image.jpg --layers "model.5.*"

The heatmap is a channel aggregation of an intermediate tensor, not a class
activation map. It answers "where this layer responds strongly", not by itself
"which pixels the model believes are cracks".
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize arbitrary intermediate features in a YOLO model.")
    parser.add_argument("--model", required=True, help="Model YAML used for training.")
    parser.add_argument("--weights", required=True, help="Trained Ultralytics checkpoint or state dict (.pt/.pth).")
    parser.add_argument("--source", help="Input image. Required unless only inspecting the layer list.")
    parser.add_argument(
        "--layers",
        default="auto",
        help=(
            "Comma-separated selectors: YAML layer indices (3,5), exact module names (model.5.op), "
            "glob patterns (model.5.*), or module types (type:VSSBlock). Default: auto."
        ),
    )
    parser.add_argument("--list-layers", action="store_true", help="Print and save all selectable module names.")
    parser.add_argument("--list-only", action="store_true", help="Exit after listing layers; --source is not required.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference size used by LetterBox.")
    parser.add_argument("--device", default="0", help="Device: cpu, 0, 1, cuda:0, etc.")
    parser.add_argument(
        "--reduce",
        choices=("mean_abs", "l2", "max_abs", "mean"),
        default="mean_abs",
        help="How to reduce channels into one spatial heatmap.",
    )
    parser.add_argument("--percentile", type=float, default=99.0, help="Robust upper percentile for heatmap scaling.")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Colored heatmap opacity in overlays.")
    parser.add_argument("--max-tensors", type=int, default=1, help="Maximum spatial tensors saved per selected module.")
    parser.add_argument("--save-npy", action="store_true", help="Also save raw feature tensors as float32 .npy files.")
    parser.add_argument(
        "--no-structure-maps",
        action="store_true",
        help="Do not auto-save crack probability, tangent, connectivity and dynamic-path maps.",
    )
    parser.add_argument("--output", default="feature_visualization", help="Output directory.")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    value = str(value).strip().lower()
    if value == "cpu" or not torch.cuda.is_available():
        if value != "cpu" and not torch.cuda.is_available():
            print("[WARN] CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    return torch.device(value)


def safe_torch_load(path: Path) -> Any:
    """Load full checkpoints and raw state dicts across supported torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch versions without weights_only
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint: Any) -> OrderedDict[str, torch.Tensor]:
    """Extract a tensor state dict from common Ultralytics and custom checkpoint formats."""
    if isinstance(checkpoint, nn.Module):
        return OrderedDict(checkpoint.float().state_dict())

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint).__name__}")

    # Ultralytics saves EMA/model modules; custom trainers commonly save one of
    # the state-dict keys below. Prefer EMA to match normal best.pt inference.
    for key in ("ema", "model", "state_dict", "model_state_dict", "weights"):
        value = checkpoint.get(key)
        if value is None:
            continue
        if isinstance(value, nn.Module):
            return OrderedDict(value.float().state_dict())
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return OrderedDict(value)

    if checkpoint and all(isinstance(k, str) and torch.is_tensor(v) for k, v in checkpoint.items()):
        return OrderedDict(checkpoint)
    raise TypeError("No model/ema/state_dict tensor weights were found in the checkpoint.")


def key_variants(state_dict: OrderedDict[str, torch.Tensor]) -> list[OrderedDict[str, torch.Tensor]]:
    """Generate common key-prefix variants and let shape matching select the best one."""
    variants = [state_dict]
    prefixes = ("module.", "_orig_mod.", "model.")
    current = state_dict
    for _ in range(3):
        changed = False
        for prefix in prefixes:
            if current and all(k.startswith(prefix) for k in current):
                current = OrderedDict((k[len(prefix):], v) for k, v in current.items())
                variants.append(current)
                changed = True
                break
        if not changed:
            break
    variants.extend(OrderedDict((f"model.{k}", v) for k, v in variant.items()) for variant in list(variants))
    return variants


def load_matching_weights(model: nn.Module, weights: Path) -> tuple[int, int]:
    checkpoint = safe_torch_load(weights)
    source = extract_state_dict(checkpoint)
    target = model.state_dict()
    candidates = key_variants(source)
    matched_candidates = [
        OrderedDict((k, v.float()) for k, v in candidate.items() if k in target and target[k].shape == v.shape)
        for candidate in candidates
    ]
    matched = max(matched_candidates, key=len)
    if not matched:
        sample_source = list(source)[:3]
        sample_target = list(target)[:3]
        raise RuntimeError(
            "No checkpoint tensors matched the YAML model. "
            f"Checkpoint examples: {sample_source}; model examples: {sample_target}"
        )
    model.load_state_dict(matched, strict=False)
    return len(matched), len(target)


def module_type_names(module: nn.Module) -> set[str]:
    return {cls.__name__ for cls in type(module).mro() if issubclass(cls, nn.Module)}


def is_top_level_layer(name: str) -> bool:
    return bool(re.fullmatch(r"model\.\d+", name))


def layer_rows(model: nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        if not name:
            continue
        rows.append(
            {
                "name": name,
                "type": type(module).__name__,
                "top_level": is_top_level_layer(name),
                "parameters": sum(p.numel() for p in module.parameters(recurse=False)),
            }
        )
    return rows


def print_and_save_layers(model: nn.Module, output_dir: Path) -> None:
    rows = layer_rows(model)
    lines = [f"{'name':<58} {'type':<28} {'own_params':>12}"]
    lines.append("-" * 102)
    lines.extend(f"{r['name']:<58} {r['type']:<28} {r['parameters']:>12}" for r in rows)
    text = "\n".join(lines)
    print(text)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_layers.txt").write_text(text + "\n", encoding="utf-8")
    (output_dir / "model_layers.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def auto_layer_names(model: nn.Module) -> list[str]:
    """Choose top-level feature-producing blocks without assuming one exact YAML."""
    preferred = (
        "AdaptiveC3k2CrackPath", "AdaptiveC3k2CASP", "C3k2", "VSSBlock", "XSSBlock",
        "C2PSA", "SPPF", "Bottleneck", "C2f", "C3"
    )
    selected = []
    for name, module in model.named_modules():
        if is_top_level_layer(name) and any(token in type(module).__name__ for token in preferred):
            selected.append(name)
    if selected:
        return selected
    return [name for name, _ in model.named_modules() if is_top_level_layer(name)][:-1]


def resolve_layers(model: nn.Module, selectors: str) -> list[str]:
    modules = dict(model.named_modules())
    if selectors.strip().lower() == "auto":
        return auto_layer_names(model)

    selected = []
    for raw in selectors.split(","):
        selector = raw.strip()
        if not selector:
            continue
        if selector.isdigit():
            selector = f"model.{selector}"
        if selector.startswith("type:"):
            requested_type = selector.split(":", 1)[1]
            matches = [name for name, module in modules.items() if requested_type in module_type_names(module)]
        elif any(char in selector for char in "*?["):
            matches = [name for name in modules if fnmatch.fnmatchcase(name, selector)]
        else:
            matches = [selector] if selector in modules else []
        if not matches:
            raise ValueError(f"Layer selector '{raw}' matched nothing. Run with --list-layers to inspect names.")
        selected.extend(matches)
    return list(dict.fromkeys(selected))


def spatial_tensors(value: Any, prefix: str = "output") -> list[tuple[str, torch.Tensor]]:
    """Recursively collect BCHW tensors from tensor/list/tuple/dict outputs."""
    found = []
    if torch.is_tensor(value):
        if value.ndim == 4 and value.shape[-2] > 1 and value.shape[-1] > 1:
            found.append((prefix, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(spatial_tensors(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(spatial_tensors(item, f"{prefix}.{index}"))
    return found


def make_hook(name: str, storage: dict[str, list[tuple[str, torch.Tensor]]], max_tensors: int):
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tensors = spatial_tensors(output)[:max_tensors]
        storage[name] = [(key, tensor.detach().float().cpu()) for key, tensor in tensors]

    return hook


def reduce_feature(feature: torch.Tensor, method: str) -> np.ndarray:
    feature = feature[0]
    if method == "mean_abs":
        heat = feature.abs().mean(dim=0)
    elif method == "l2":
        heat = feature.square().mean(dim=0).sqrt()
    elif method == "max_abs":
        heat = feature.abs().amax(dim=0)
    else:
        heat = feature.mean(dim=0)
    return heat.numpy()


def normalize_heatmap(heat: np.ndarray, percentile: float) -> np.ndarray:
    finite = np.isfinite(heat)
    if not finite.any():
        return np.zeros_like(heat, dtype=np.uint8)
    values = heat[finite]
    low_percentile = max(0.0, 100.0 - percentile)
    low, high = np.percentile(values, [low_percentile, percentile])
    if high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros_like(heat, dtype=np.uint8)
    normalized = np.clip((heat - low) / (high - low), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), thickness=-1)
    cv2.putText(image, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def save_feature_visuals(
    name: str,
    output_key: str,
    feature: torch.Tensor,
    input_bgr: np.ndarray,
    output_dir: Path,
    reduce: str,
    percentile: float,
    overlay_alpha: float,
    save_npy: bool,
) -> np.ndarray:
    stem = sanitize(f"{name}_{output_key}")
    heat = reduce_feature(feature, reduce)
    heat_u8 = normalize_heatmap(heat, percentile)
    heat_u8 = cv2.resize(heat_u8, (input_bgr.shape[1], input_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(input_bgr, 1.0 - overlay_alpha, colored, overlay_alpha, 0.0)

    cv2.imwrite(str(output_dir / f"{stem}_heatmap.png"), colored)
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)
    if save_npy:
        np.save(output_dir / f"{stem}_feature.npy", feature[0].numpy())

    input_panel = add_label(input_bgr, "input")
    heat_panel = add_label(colored, f"{name} | {tuple(feature.shape)}")
    overlay_panel = add_label(overlay, f"{reduce} overlay")
    return np.concatenate((input_panel, heat_panel, overlay_panel), axis=1)


def save_contact_sheet(rows: list[np.ndarray], path: Path) -> None:
    if not rows:
        return
    target_width = max(row.shape[1] for row in rows)
    resized = []
    for row in rows:
        if row.shape[1] != target_width:
            height = round(row.shape[0] * target_width / row.shape[1])
            row = cv2.resize(row, (target_width, height), interpolation=cv2.INTER_AREA)
        resized.append(row)
    cv2.imwrite(str(path), np.concatenate(resized, axis=0))


def save_orientation_visual(
    name: str, orientation: torch.Tensor, input_bgr: np.ndarray, output_dir: Path, overlay_alpha: float
) -> np.ndarray:
    """Save an undirected (cos(2 theta), sin(2 theta)) field as an HSV orientation image."""
    vector = orientation[0].float().cpu().numpy()
    angle = 0.5 * np.arctan2(vector[1], vector[0])
    magnitude = np.sqrt(np.square(vector).sum(axis=0))
    hue = ((angle % np.pi) / np.pi * 179.0).astype(np.uint8)
    value = normalize_heatmap(magnitude, 99.0)
    hsv = np.stack((hue, np.full_like(hue, 255), value), axis=-1)
    colored = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    colored = cv2.resize(colored, (input_bgr.shape[1], input_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(input_bgr, 1.0 - overlay_alpha, colored, overlay_alpha, 0.0)
    stem = sanitize(f"{name}_orientation")
    cv2.imwrite(str(output_dir / f"{stem}_hsv.png"), colored)
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)
    return np.concatenate(
        (add_label(input_bgr, "input"), add_label(colored, f"{name} | tangent HSV"), add_label(overlay, "orientation overlay")),
        axis=1,
    )


def save_dynamic_paths(name: str, indices: torch.Tensor, valid_mask: torch.Tensor,
                       feature_size: tuple[int, int], input_bgr: np.ndarray,
                       output_dir: Path) -> tuple[np.ndarray, float]:
    """Overlay the first image's sparse image-adaptive token paths."""
    height, width = feature_size
    overlay = input_bgr.copy()
    indices = indices[0].detach().cpu()
    valid_mask = valid_mask[0].detach().cpu().bool()
    palette = ((0, 255, 255), (255, 160, 0), (80, 255, 80), (255, 80, 220))
    visited = set()
    for path_index in range(min(indices.shape[0], 64)):
        token_ids = indices[path_index][valid_mask[path_index]].tolist()
        if len(token_ids) < 2:
            continue
        visited.update(token_ids)
        points = np.asarray([
            (
                round(((token_id % width) + 0.5) * input_bgr.shape[1] / width),
                round(((token_id // width) + 0.5) * input_bgr.shape[0] / height),
            )
            for token_id in token_ids
        ], dtype=np.int32)
        cv2.polylines(overlay, [points], False, palette[path_index % len(palette)], 1, cv2.LINE_AA)
    coverage = len(visited) / float(height * width)
    stem = sanitize(f"{name}_dynamic_paths")
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)
    panel = np.concatenate(
        (add_label(input_bgr, "input"), add_label(overlay, f"dynamic crack paths | coverage={coverage:.3f}")), axis=1
    )
    return panel, coverage


def save_structure_maps(
    model: nn.Module, input_bgr: np.ndarray, output_dir: Path, overlay_alpha: float, save_npy: bool
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Export probability, tangent, connectivity, edge and dynamic-path caches."""
    panels, records = [], []
    for name, module in model.named_modules():
        guidance = getattr(module, "last_guidance", None)
        orientation = getattr(module, "last_orientation", None)
        connectivity = getattr(module, "last_connectivity", None)
        edge_confidence = getattr(module, "last_edge_confidence", None)
        path_indices = getattr(module, "last_path_indices", None)
        path_mask = getattr(module, "last_path_mask", None)
        if torch.is_tensor(guidance):
            guidance = guidance.detach().float().cpu()
            panels.append(
                save_feature_visuals(
                    name, "crack_guidance", guidance, input_bgr, output_dir, "mean", 99.0, overlay_alpha, save_npy
                )
            )
            records.append({"layer": name, "kind": "crack_guidance", "shape": list(guidance.shape)})
        if torch.is_tensor(orientation) and orientation.shape[1] == 2 and getattr(module, "crack_aligned_edges", False):
            orientation = orientation.detach().float().cpu()
            family_probability = torch.softmax(
                float(getattr(module, "orientation_temperature", 1.0)) * orientation, dim=1
            )
            for index, family in enumerate(("horizontal", "vertical")):
                panels.append(save_feature_visuals(
                    name, f"orientation_{family}", family_probability[:, index:index + 1],
                    input_bgr, output_dir, "mean", 99.0, overlay_alpha, save_npy
                ))
            records.append({"layer": name, "kind": "orientation_hv_probability", "shape": list(orientation.shape)})
        elif torch.is_tensor(orientation) and orientation.shape[1] == 2:
            orientation = orientation.detach().float().cpu()
            panels.append(save_orientation_visual(name, orientation, input_bgr, output_dir, overlay_alpha))
            if save_npy:
                np.save(output_dir / f"{sanitize(name)}_orientation.npy", orientation[0].numpy())
            records.append({"layer": name, "kind": "orientation", "shape": list(orientation.shape)})
        if torch.is_tensor(connectivity) and connectivity.shape[1] == 4:
            connectivity = connectivity.detach().float().cpu()
            for index, family in enumerate(("horizontal", "vertical", "main_diagonal", "anti_diagonal")):
                panels.append(save_feature_visuals(
                    name, f"connectivity_{family}", connectivity[:, index:index + 1],
                    input_bgr, output_dir, "mean", 99.0, overlay_alpha, save_npy
                ))
            records.append({"layer": name, "kind": "crack_connectivity_hvda", "shape": list(connectivity.shape)})
        if torch.is_tensor(edge_confidence) and edge_confidence.shape[1] == 2:
            edge_confidence = edge_confidence.detach().float().cpu()
            for index, family in enumerate(("horizontal", "vertical")):
                panels.append(save_feature_visuals(
                    name, f"edge_{family}", edge_confidence[:, index:index + 1],
                    input_bgr, output_dir, "mean", 99.0, overlay_alpha, save_npy
                ))
            records.append({"layer": name, "kind": "crack_edge_hv", "shape": list(edge_confidence.shape)})
        if (torch.is_tensor(path_indices) and torch.is_tensor(path_mask)
                and torch.is_tensor(guidance)):
            panel, coverage = save_dynamic_paths(
                name, path_indices, path_mask, tuple(guidance.shape[-2:]), input_bgr, output_dir
            )
            panels.append(panel)
            records.append({
                "layer": name, "kind": "dynamic_crack_paths", "shape": list(path_indices.shape),
                "coverage_first_image": coverage,
            })
    return panels, records


def preprocess_image(source: Path, imgsz: int, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    image = cv2.imread(str(source))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source}")
    letterboxed = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    rgb = np.ascontiguousarray(letterboxed[:, :, ::-1].transpose(2, 0, 1))
    tensor = torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
    return letterboxed, tensor


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if not 50.0 < args.percentile <= 100.0:
        raise ValueError("--percentile must be in (50, 100]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")
    if args.max_tensors <= 0:
        raise ValueError("--max-tensors must be positive")
    model_path, weights_path = Path(args.model), Path(args.weights)
    output_dir = Path(args.output)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {model_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights do not exist: {weights_path}")

    device = resolve_device(args.device)
    print(f"[INFO] Building model from {model_path}")
    wrapper = YOLO(str(model_path))
    model = wrapper.model
    matched, total = load_matching_weights(model, weights_path)
    print(f"[INFO] Loaded {matched}/{total} matching state tensors from {weights_path}")
    if matched / total < 0.5:
        print("[WARN] Fewer than 50% of model state tensors matched. Verify YAML, scale and class count.")
    model.to(device).eval()

    if args.list_layers or args.list_only:
        print_and_save_layers(model, output_dir)
    if args.list_only:
        return
    if not args.source:
        raise ValueError("--source is required unless --list-only is used")

    selected = resolve_layers(model, args.layers)
    print("[INFO] Selected layers:")
    for name in selected:
        print(f"  {name:<45} {type(dict(model.named_modules())[name]).__name__}")

    captured: dict[str, list[tuple[str, torch.Tensor]]] = {}
    modules = dict(model.named_modules())
    handles = [
        modules[name].register_forward_hook(make_hook(name, captured, args.max_tensors)) for name in selected
    ]
    source = Path(args.source)
    input_bgr, tensor = preprocess_image(source, args.imgsz, device)
    try:
        with torch.inference_mode():
            model(tensor)
    finally:
        for handle in handles:
            handle.remove()

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "input_letterbox.png"), input_bgr)
    panels = []
    manifest = []
    for name in selected:
        tensors = captured.get(name, [])
        if not tensors:
            print(f"[WARN] {name} produced no spatial BCHW tensor.")
            continue
        for output_key, feature in tensors:
            panels.append(
                save_feature_visuals(
                    name,
                    output_key,
                    feature,
                    input_bgr,
                    output_dir,
                    args.reduce,
                    args.percentile,
                    args.overlay_alpha,
                    args.save_npy,
                )
            )
            manifest.append(
                {
                    "layer": name,
                    "type": type(modules[name]).__name__,
                    "output": output_key,
                    "shape": list(feature.shape),
                    "mean": float(feature.mean()),
                    "mean_abs": float(feature.abs().mean()),
                    "rms": float(feature.square().mean().sqrt()),
                    "std": float(feature.std()),
                }
            )
    structure_manifest = []
    if not args.no_structure_maps:
        structure_panels, structure_manifest = save_structure_maps(
            model, input_bgr, output_dir, args.overlay_alpha, args.save_npy
        )
        panels.extend(structure_panels)
    if not manifest and not structure_manifest:
        raise RuntimeError("No spatial feature tensors or structure maps were captured.")

    save_contact_sheet(panels, output_dir / "contact_sheet.png")
    metadata = {
        "model": str(model_path),
        "weights": str(weights_path),
        "source": str(source),
        "imgsz": args.imgsz,
        "device": str(device),
        "reduction": args.reduce,
        "matched_state_tensors": matched,
        "total_state_tensors": total,
        "features": manifest,
        "structure_maps": structure_manifest,
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] Saved {len(manifest)} feature visualization(s) to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
