#!/usr/bin/env python3
"""Benchmark a YAML-defined Ultralytics segmentation model at 640x640.

Reports parameter count, GFLOPs, pure PyTorch forward FPS, and end-to-end
Ultralytics prediction FPS on an actual split from crack-seg.yaml/data.yaml.
The benchmark is repeated for multiple rounds and exports CSV/JSON records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from visualize_seg_results import (
    is_dataset_yaml,
    load_model,
    prepare_device_environment,
    resolve_dataset_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Params, GFLOPs, and repeated 640x640 FPS for a model YAML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-yaml", required=True, help="Model architecture YAML")
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional .pt/.pth weights; omit to benchmark the randomly initialized YAML architecture",
    )
    parser.add_argument("--data", default="crack-seg/crack-seg.yaml", help="Ultralytics dataset YAML")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="0", help="CUDA device such as 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="Square inference resolution")
    parser.add_argument("--batch", type=int, default=1, help="Inference batch size")
    parser.add_argument("--rounds", type=int, default=5, help="Repeated timing rounds")
    parser.add_argument("--forward-iters", type=int, default=100, help="Model-forward iterations per round")
    parser.add_argument("--warmup", type=int, default=20, help="Untimed model-forward warmup iterations")
    parser.add_argument(
        "--images-per-round",
        type=int,
        default=0,
        help="Dataset images used in each end-to-end round; 0 means the complete split",
    )
    parser.add_argument(
        "--pipeline-warmup-images",
        type=int,
        default=16,
        help="Untimed dataset images used to initialize prediction and CUDA kernels",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--half", action="store_true", help="Benchmark FP16 inference on CUDA")
    parser.add_argument("--seed", type=int, default=1, help="Dataset sampling seed")
    parser.add_argument("--output-dir", default="benchmark_results", help="CSV/JSON output directory")
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="After timing, save Ultralytics prediction visualizations for the selected images",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("--imgsz and --batch must be positive")
    if args.rounds <= 0 or args.forward_iters <= 0 or args.warmup < 0:
        raise ValueError("--rounds/--forward-iters must be positive and --warmup cannot be negative")
    if args.images_per_round < 0 or args.pipeline_warmup_images < 0:
        raise ValueError("image counts cannot be negative")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("--conf and --iou must be within [0, 1]")


def synchronize(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean_std(values: Sequence[float]) -> tuple:
    if not values:
        return math.nan, math.nan
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def write_csv(path: Path, rows: Sequence[Dict], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_images(images: Sequence[Path], count: int, seed: int) -> List[Path]:
    if count == 0 or count >= len(images):
        return list(images)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(images)), count))
    return [images[index] for index in indices]


def write_manifest(path: Path, images: Sequence[Path]) -> None:
    """Use a TXT source so Ultralytics opens only one batch instead of every path at once."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for image in images:
            stream.write(f"{image.resolve()}\n")


def load_yaml_model(model_yaml: Path, weights: Optional[Path]):
    if weights is not None:
        return load_model(weights, model_yaml)
    from ultralytics import YOLO

    return YOLO(str(model_yaml.resolve()), task="segment"), "model YAML with random initialization"


def count_parameters(model) -> tuple:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def calculate_gflops(core_model, imgsz: int) -> tuple:
    """Use the same THOP-based method as Ultralytics validation, with profiler fallback."""
    from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler

    method = "Ultralytics THOP estimate"
    try:
        value = float(get_flops(core_model, imgsz=imgsz))
    except Exception:
        value = 0.0
    if value > 0:
        return value, method
    try:
        value = float(get_flops_with_torch_profiler(core_model, imgsz=imgsz))
        if value > 0:
            return value, "torch.profiler fallback"
    except Exception:
        pass
    return math.nan, "unavailable (custom operator was not counted)"


def benchmark_forward(torch, core_model, device, args: argparse.Namespace) -> List[Dict]:
    """Measure raw neural-network forward throughput, excluding all data/post-processing."""
    first_parameter = next(core_model.parameters())
    input_channels = 3
    for parameter in core_model.parameters():
        if parameter.ndim == 4:
            input_channels = int(parameter.shape[1])
            break
    dtype = torch.float16 if args.half and device.type == "cuda" else first_parameter.dtype
    sample = torch.rand(args.batch, input_channels, args.imgsz, args.imgsz, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(args.warmup):
            core_model(sample)
        synchronize(torch, device)

        rows = []
        for round_index in range(1, args.rounds + 1):
            synchronize(torch, device)
            started = time.perf_counter()
            for _ in range(args.forward_iters):
                core_model(sample)
            synchronize(torch, device)
            elapsed = time.perf_counter() - started
            image_count = args.batch * args.forward_iters
            rows.append(
                {
                    "round": round_index,
                    "mode": "model_forward",
                    "images": image_count,
                    "seconds": elapsed,
                    "latency_ms_per_image": elapsed * 1000.0 / image_count,
                    "fps": image_count / elapsed,
                }
            )
    return rows


def prediction_kwargs(args: argparse.Namespace, source: Path, save: bool = False, output_dir: Optional[Path] = None):
    kwargs = dict(
        source=str(source),
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        half=args.half,
        verbose=False,
        save=save,
    )
    if save and output_dir is not None:
        kwargs.update(project=str(output_dir), name="predictions", exist_ok=True)
    return kwargs


def consume_predictions(results) -> int:
    count = 0
    for _ in results:
        count += 1
    return count


def benchmark_pipeline(torch, yolo_model, device, args: argparse.Namespace, manifest: Path, warmup_manifest: Path):
    """Measure image loading, preprocessing, forward, NMS, mask post-processing, and Results creation."""
    if args.pipeline_warmup_images > 0:
        consume_predictions(yolo_model.predict(**prediction_kwargs(args, warmup_manifest)))
        synchronize(torch, device)

    rows = []
    for round_index in range(1, args.rounds + 1):
        synchronize(torch, device)
        started = time.perf_counter()
        count = consume_predictions(yolo_model.predict(**prediction_kwargs(args, manifest)))
        synchronize(torch, device)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "round": round_index,
                "mode": "end_to_end",
                "images": count,
                "seconds": elapsed,
                "latency_ms_per_image": elapsed * 1000.0 / max(count, 1),
                "fps": count / elapsed,
            }
        )
        print(
            f"End-to-end round {round_index}/{args.rounds}: "
            f"{count} images, {elapsed:.3f}s, {count / elapsed:.2f} FPS"
        )
    return rows


def aggregate_rounds(rows: Sequence[Dict], mode: str) -> Dict[str, float]:
    selected = [row for row in rows if row["mode"] == mode]
    fps_values = [float(row["fps"]) for row in selected]
    latency_values = [float(row["latency_ms_per_image"]) for row in selected]
    fps_mean, fps_std = mean_std(fps_values)
    latency_mean, latency_std = mean_std(latency_values)
    return {
        f"{mode}_fps_mean": fps_mean,
        f"{mode}_fps_std": fps_std,
        f"{mode}_fps_min": min(fps_values) if fps_values else math.nan,
        f"{mode}_fps_max": max(fps_values) if fps_values else math.nan,
        f"{mode}_latency_ms_mean": latency_mean,
        f"{mode}_latency_ms_std": latency_std,
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    previous_visible = prepare_device_environment(args.device)

    model_yaml = Path(args.model_yaml).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve() if args.weights else None
    data_yaml = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_yaml.is_file():
        raise FileNotFoundError(f"model YAML not found: {model_yaml}")
    if weights is not None and not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    if not is_dataset_yaml(data_yaml):
        raise ValueError(f"not an Ultralytics dataset YAML: {data_yaml}")

    all_images, dataset_root, _ = resolve_dataset_source(data_yaml, args.split)
    if not all_images:
        raise FileNotFoundError(f"no images resolved for split '{args.split}' from {data_yaml}")
    selected_images = select_images(all_images, args.images_per_round, args.seed)
    warmup_count = min(len(selected_images), args.pipeline_warmup_images)
    warmup_images = selected_images[:warmup_count]
    manifest = output_dir / "benchmark_images.txt"
    warmup_manifest = output_dir / "warmup_images.txt"
    write_manifest(manifest, selected_images)
    write_manifest(warmup_manifest, warmup_images or selected_images[:1])

    yolo_model, load_description = load_yaml_model(model_yaml, weights)
    import torch
    from ultralytics.utils.torch_utils import select_device

    device = select_device(args.device, batch=args.batch, verbose=True)
    core_model = yolo_model.model.to(device).eval()
    use_half = bool(args.half and device.type == "cuda")
    if args.half and not use_half:
        print("Warning: --half is supported only on CUDA; using FP32.", file=sys.stderr)
        args.half = False
    core_model.float()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    params, trainable_params = count_parameters(core_model)
    gflops, flops_method = calculate_gflops(core_model, args.imgsz)
    # Ultralytics prediction uses fused Conv+BN layers. Preserve the original
    # parameter/FLOPs report above, then use the inference-form model for timing.
    fused_for_timing = False
    if hasattr(core_model, "fuse"):
        core_model = core_model.fuse(verbose=False)
        yolo_model.model = core_model
        fused_for_timing = True
    if use_half:
        core_model.half()
    print(f"Model:      {load_description}")
    print(f"YAML:       {model_yaml}")
    print(f"Weights:    {weights or '<random initialization>'}")
    print(f"Dataset:    {data_yaml} / {args.split} ({len(all_images)} available, {len(selected_images)} timed)")
    print(f"Resolution: {args.imgsz}x{args.imgsz}, batch={args.batch}, precision={'FP16' if use_half else 'FP32'}")
    print(f"Params:     {params:,} ({params / 1e6:.3f} M)")
    print(f"GFLOPs:     {gflops:.3f}" if math.isfinite(gflops) else f"GFLOPs:     N/A ({flops_method})")

    forward_rows = benchmark_forward(torch, core_model, device, args)
    forward_stats = aggregate_rounds(forward_rows, "model_forward")
    print(
        f"Forward:    {forward_stats['model_forward_fps_mean']:.2f} ± "
        f"{forward_stats['model_forward_fps_std']:.2f} FPS"
    )

    pipeline_rows = benchmark_pipeline(torch, yolo_model, device, args, manifest, warmup_manifest)
    pipeline_stats = aggregate_rounds(pipeline_rows, "end_to_end")
    print(
        f"End-to-end: {pipeline_stats['end_to_end_fps_mean']:.2f} ± "
        f"{pipeline_stats['end_to_end_fps_std']:.2f} FPS"
    )

    round_rows = forward_rows + pipeline_rows
    write_csv(output_dir / "benchmark_rounds.csv", round_rows)
    summary = {
        "model_yaml": str(model_yaml),
        "weights": str(weights) if weights else "",
        "data": str(data_yaml),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "device_argument": str(args.device),
        "cuda_visible_devices_previous": previous_visible or "",
        "torch_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "precision": "FP16" if use_half else "FP32",
        "rounds": args.rounds,
        "forward_iters_per_round": args.forward_iters,
        "dataset_images_per_round": len(selected_images),
        "params": params,
        "params_million": params / 1e6,
        "trainable_params": trainable_params,
        "gflops": gflops if math.isfinite(gflops) else "",
        "gflops_method": flops_method,
        "fused_for_timing": fused_for_timing,
        **forward_stats,
        **pipeline_stats,
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        ),
    }
    write_csv(output_dir / "benchmark_summary.csv", [summary])
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)

    if args.save_predictions:
        print("Saving prediction visualizations...")
        consume_predictions(
            yolo_model.predict(**prediction_kwargs(args, manifest, save=True, output_dir=output_dir))
        )

    print(f"Rounds CSV:  {output_dir / 'benchmark_rounds.csv'}")
    print(f"Summary CSV: {output_dir / 'benchmark_summary.csv'}")
    print(f"Summary JSON:{output_dir / 'benchmark_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
