"""All-in-one CrackPath Mamba-YOLO with a Crack-Structure Fusion Neck.

This file intentionally keeps the complete experimental definition in one place:

1. ``CrackStructureFusion`` (CSF): target-preserving, structure-gated source injection.
2. ``CrackStructureFusionNeck``: TD4 -> TD3 -> BU4 -> BU5 bidirectional fusion.
3. ``CrackPathCSFSegment``: CSF-Neck followed by the official YOLO11 segmentation head.
4. ``MODEL_YAML``: the complete YOLO11n-Seg + CPSB Backbone configuration.
5. A small CLI to export the embedded YAML, smoke-test, train, validate, or predict.

The CPSB implementation is reused from ``ultralytics.nn.modules.mamba_yolo`` rather
than copied.  This prevents the paper reference implementation from silently
diverging from the CPSB that produced the existing G01/K01 experiments.

Important architectural contract
--------------------------------
* Backbone P2-P5: AdaptiveC3k2CrackPath (CPSB).
* Deepest backbone context: official SPPF -> C2PSA.
* Neck: four CSF calls; no CPSB is added to the Neck.
* Head: official YOLO11 instance-segmentation prediction branches.
* Shared story: p/o/c controls within-scale state propagation in CPSB and
  cross-scale residual injection in CSF.

Example
-------
Smoke test::

    python crackpath_mamba_yolo_csf_all_in_one.py --action check --device cpu

Export the embedded YAML for inspection::

    python crackpath_mamba_yolo_csf_all_in_one.py --action export-yaml

Train (single process)::

    python crackpath_mamba_yolo_csf_all_in_one.py --action train \
        --data crack-seg/data.yaml --device 0 --batch 16 --epochs 300

The single-file runtime registration is performed before model/checkpoint loading.
For distributed training through an external launcher, import this module and call
``register_csf_head()`` in every worker before constructing ``YOLO``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.head import v11Segment as OfficialYOLO11Segment


# Keep a stable import path when this file is executed as a script.  Ultralytics
# checkpoints pickle the model object, so the custom class must not be recorded
# only as ``__main__.CrackPathCSFSegment``.
MODULE_NAME = "crackpath_mamba_yolo_csf_all_in_one"
sys.modules.setdefault(MODULE_NAME, sys.modules[__name__])


def _inverse_tanh(value: float) -> float:
    """Numerically safe inverse tanh used for the bounded residual scale."""
    value = min(max(float(value), -0.999999), 0.999999)
    return 0.5 * math.log((1.0 + value) / (1.0 - value))


class StructureCueHead(nn.Module):
    """Predict crack probability p, undirected orientation o, and connectivity c.

    Output layout is ``[p, o_x, o_y, c_0, c_1, c_2, c_3]``.  Orientation is
    normalized as a double-angle vector, matching the undirected tangent grammar
    used by CPSB.  The head is deliberately depthwise-separable and lightweight.
    """

    def __init__(self, channels: int, hidden_ratio: float = 0.25, kernel_size: int = 3):
        super().__init__()
        hidden = max(8, int(round(channels * hidden_ratio / 8.0)) * 8)
        padding = kernel_size // 2
        self.reduce = Conv(channels, hidden, 1, 1)
        self.context = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size, padding=padding, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.predict = nn.Conv2d(hidden, 7, 1, bias=True)
        nn.init.normal_(self.predict.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.predict.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.predict(self.context(self.reduce(x)))
        probability = raw[:, :1].sigmoid()
        orientation = F.normalize(raw[:, 1:3].float(), dim=1, eps=1e-6).to(raw.dtype)
        connectivity = raw[:, 3:].sigmoid()
        structure = torch.cat((probability, orientation, connectivity), dim=1)
        return structure, probability, orientation, connectivity


class CrackStructureFusion(nn.Module):
    """Fuse one aligned source scale into a target scale using p/o/c agreement.

    Let ``T`` be the target feature and ``S`` the source feature.  The unit keeps
    the target route as the identity-carrying path and injects only a gated source
    residual::

        Y = T_proj + gamma * G(p, o, c) * R_s

    ``gamma`` is bounded and zero-initialized by default, making the initial model
    equal to the target projection instead of an uncontrolled new fusion branch.
    Spatial alignment is performed inside this unit, so the same implementation
    supports both top-down and bottom-up calls.
    """

    def __init__(
        self,
        target_channels: int,
        source_channels: int,
        out_channels: int,
        structure_ratio: float = 0.25,
        gate_hidden: int = 16,
        gamma_init: float = 0.0,
        gamma_max: float = 0.5,
        align_mode: str = "bilinear",
    ):
        super().__init__()
        if out_channels < 1:
            raise ValueError("out_channels must be positive")
        if not 0.0 <= abs(gamma_init) < gamma_max:
            raise ValueError("abs(gamma_init) must be smaller than gamma_max")
        if align_mode not in {"nearest", "bilinear"}:
            raise ValueError("align_mode must be 'nearest' or 'bilinear'")

        self.target_channels = int(target_channels)
        self.source_channels = int(source_channels)
        self.out_channels = int(out_channels)
        self.align_mode = align_mode
        self.gamma_max = float(gamma_max)

        self.target_project = (
            nn.Identity()
            if target_channels == out_channels
            else Conv(target_channels, out_channels, 1, 1)
        )
        self.source_project = Conv(source_channels, out_channels, 1, 1)
        self.source_residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.structure = StructureCueHead(out_channels, structure_ratio)
        gate_hidden = max(8, int(gate_hidden))
        self.gate_refine = nn.Sequential(
            nn.Conv2d(14, gate_hidden, 1, bias=False),
            nn.BatchNorm2d(gate_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, 1, bias=True),
        )
        # sigmoid(0)=0.5 is neutral; the analytic p/o/c agreement still controls
        # the spatial pattern.  The zero gamma protects the pretrained topology.
        nn.init.zeros_(self.gate_refine[-1].weight)
        nn.init.zeros_(self.gate_refine[-1].bias)
        raw_gamma = _inverse_tanh(gamma_init / gamma_max) if gamma_init else 0.0
        self.gamma_raw = nn.Parameter(torch.tensor(raw_gamma, dtype=torch.float32))
        self.gamma_raw._no_weight_decay = True

        # Existing segmentation loss discovers these attributes and can apply the
        # same weak structure supervision already used for CPSB.  Target cues are
        # exposed; the shared cue head also processes the aligned source feature.
        self.semantic_structure = True
        self.last_guidance = None
        self.last_orientation = None
        self.last_connectivity = None
        self.last_gate = None
        self.last_gamma = None

    def __getstate__(self):
        state = self.__dict__.copy()
        for name in (
            "last_guidance",
            "last_orientation",
            "last_connectivity",
            "last_gate",
            "last_gamma",
        ):
            state.pop(name, None)
        return state

    def effective_gamma(self) -> torch.Tensor:
        return self.gamma_max * torch.tanh(self.gamma_raw)

    def _resize_source(self, source: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
        if tuple(source.shape[-2:]) == tuple(size):
            return source
        kwargs = {"size": tuple(size), "mode": self.align_mode}
        if self.align_mode == "bilinear":
            kwargs["align_corners"] = False
        return F.interpolate(source, **kwargs)

    @staticmethod
    def _agreement(
        target_p: torch.Tensor,
        target_o: torch.Tensor,
        target_c: torch.Tensor,
        source_p: torch.Tensor,
        source_o: torch.Tensor,
        source_c: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-6
        probability = torch.sqrt((target_p * source_p).clamp_min(eps))
        # Double-angle vectors make opposite tangent directions equivalent.
        orientation = (target_o * source_o).sum(1, keepdim=True)
        orientation = (0.5 + 0.5 * orientation).clamp(0.0, 1.0)
        connectivity = torch.sqrt((target_c * source_c).clamp_min(eps)).mean(1, keepdim=True)
        # Do not fully kill a route because one early cue is uncertain.
        return probability * (0.5 + 0.5 * orientation) * (0.5 + 0.5 * connectivity)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise TypeError("CrackStructureFusion expects [target, source]")
        target, source = inputs
        target = self.target_project(target)
        # Resize before channel projection.  This is especially important for
        # bottom-up fusion: projecting a large P3 map and only then reducing its
        # resolution would waste most of the Neck computation.
        source = self._resize_source(source, target.shape[-2:])
        source = self.source_project(source)

        target_s, target_p, target_o, target_c = self.structure(target)
        source_s, source_p, source_o, source_c = self.structure(source)
        analytic_gate = self._agreement(
            target_p, target_o, target_c, source_p, source_o, source_c
        )
        learned_gate = self.gate_refine(torch.cat((target_s, source_s), dim=1)).sigmoid()
        gate = analytic_gate * learned_gate
        gamma = self.effective_gamma().to(dtype=target.dtype, device=target.device)
        source_delta = self.source_residual(source)
        output = target + gamma * gate * source_delta

        self.last_guidance = target_p
        self.last_orientation = target_o
        self.last_connectivity = target_c
        self.last_gate = gate.detach()
        self.last_gamma = gamma.detach()
        return output


class CrackStructureFusionNeck(nn.Module):
    """Bidirectional P3/P4/P5 CSF-Neck.

    Input: ``[B3, B4, B5]`` where B5 is the output of SPPF -> C2PSA.
    Output: ``[P3', P4', P5'] = [T3, N4, N5]``.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Sequence[int] | None = None,
        structure_ratio: float = 0.25,
        gate_hidden: int = 16,
        gamma_init: float = 0.0,
        gamma_max: float = 0.5,
        align_mode: str = "bilinear",
    ):
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError(f"CSF-Neck requires three input scales, got {in_channels}")
        b3, b4, b5 = (int(x) for x in in_channels)
        if out_channels is None:
            # Match the ordinary YOLO11n Neck widths: P3 is compressed by 2,
            # while P4 and P5 retain their backbone-stage widths.
            out_channels = (max(16, b3 // 2), b4, b5)
        p3, p4, p5 = (int(x) for x in out_channels)
        common = dict(
            structure_ratio=structure_ratio,
            gate_hidden=gate_hidden,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            align_mode=align_mode,
        )

        # Top-down semantic injection.
        self.td4 = CrackStructureFusion(b4, b5, p4, **common)
        self.td3 = CrackStructureFusion(b3, p4, p3, **common)
        # Bottom-up detail reinforcement.
        self.bu4 = CrackStructureFusion(p4, p3, p4, **common)
        self.bu5 = CrackStructureFusion(b5, p4, p5, **common)
        self.out_channels = (p3, p4, p5)

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not isinstance(features, (tuple, list)) or len(features) != 3:
            raise TypeError("CrackStructureFusionNeck expects [B3, B4, B5]")
        b3, b4, b5 = features
        t4 = self.td4((b4, b5))
        t3 = self.td3((b3, t4))
        n4 = self.bu4((t4, t3))
        n5 = self.bu5((b5, n4))
        return [t3, n4, n5]


class CrackPathCSFSegment(OfficialYOLO11Segment):
    """YOLO11 segmentation head preceded by the proposed CSF-Neck.

    The argument order is designed to remain compatible with Ultralytics'
    existing ``v11Segment`` parser branch.  ``ch`` is appended automatically by
    ``parse_model`` after the configurable CSF arguments.
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        structure_ratio: float = 0.25,
        gate_hidden: int = 16,
        gamma_init: float = 0.0,
        gamma_max: float = 0.5,
        align_mode: str = "bilinear",
        ch: Sequence[int] = (),
    ):
        if len(ch) != 3:
            raise ValueError(f"CrackPathCSFSegment requires B3/B4/B5 channels, got {ch}")
        output_channels = (max(16, int(ch[0]) // 2), int(ch[1]), int(ch[2]))
        # Construct the official prediction branches for the CSF output widths.
        super().__init__(nc=nc, nm=nm, npr=npr, ch=output_channels)
        self.csf_neck = CrackStructureFusionNeck(
            ch,
            output_channels,
            structure_ratio=structure_ratio,
            gate_hidden=gate_hidden,
            gamma_init=gamma_init,
            gamma_max=gamma_max,
            align_mode=align_mode,
        )
        self.backbone_channels = tuple(int(x) for x in ch)
        self.neck_channels = output_channels

    def forward(self, x: Sequence[torch.Tensor]):
        return super().forward(self.csf_neck(x))


# Stable pickle metadata for checkpoints created when the file is run directly.
for _custom_class in (
    StructureCueHead,
    CrackStructureFusion,
    CrackStructureFusionNeck,
    CrackPathCSFSegment,
):
    _custom_class.__module__ = MODULE_NAME


MODEL_YAML = textwrap.dedent(
    """
    # CrackPath Mamba-YOLO: CPSB Backbone + CSF-Neck + YOLO11 Segmentation Head
    # This YAML is embedded in crackpath_mamba_yolo_csf_all_in_one.py.
    nc: 1
    scale: n

    # Weak structure supervision is shared by CPSB and the target-side CSF cues.
    guidance_loss_weight: 0.1
    guidance_dice_weight: 1.0
    orientation_loss_weight: 0.01
    connectivity_loss_weight: 0.03
    gate_regularization_weight: 0.0

    scales:
      n: [0.50, 0.25, 1024]
      s: [0.50, 0.50, 1024]
      m: [0.50, 1.00, 512]
      l: [1.00, 1.00, 512]
      x: [1.00, 1.50, 512]

    backbone:
      - [-1, 1, Conv, [64, 3, 2]]
      - [-1, 1, Conv, [128, 3, 2]]
      # CPSB P2: shallow crack-detail state propagation.
      - [-1, 2, AdaptiveC3k2CrackPath, [256, False, 0.25, 0.25, 0.02, 128, 3, 0.05, 0.02, 0.50, 8, 0.05, 0.50, 0.05, 0.50, 0.05, 0.25, 3, 0.01]]
      - [-1, 1, Conv, [256, 3, 2]]
      # CPSB P3 -> B3 (1/8).
      - [-1, 2, AdaptiveC3k2CrackPath, [512, False, 0.25, 0.25, 0.02, 128, 3, 0.05, 0.02, 0.50, 8, 0.05, 0.50, 0.05, 0.50, 0.05, 0.25, 3, 0.01]]
      - [-1, 1, Conv, [512, 3, 2]]
      # CPSB P4 -> B4 (1/16).
      - [-1, 2, AdaptiveC3k2CrackPath, [512, True, 0.50, 0.25, 0.02, 128, 3, 0.05, 0.02, 0.50, 8, 0.05, 0.50, 0.05, 0.50, 0.05, 0.25, 3, 0.01]]
      - [-1, 1, Conv, [1024, 3, 2]]
      # CPSB P5 followed by the retained official context modules.
      - [-1, 2, AdaptiveC3k2CrackPath, [1024, True, 0.50, 0.25, 0.02, 128, 3, 0.05, 0.02, 0.50, 8, 0.05, 0.50, 0.05, 0.50, 0.05, 0.25, 3, 0.01]]
      - [-1, 1, SPPF, [1024, 5]]
      - [-1, 2, C2PSA, [1024]]  # B5 (1/32)

    head:
      # The registered v11Segment class contains:
      # CSF-TD4 -> CSF-TD3 -> CSF-BU4 -> CSF-BU5 -> official YOLO11 Segment.
      # Args: nc, nm, npr, structure_ratio, gate_hidden, gamma_init,
      #       gamma_max, align_mode.  parse_model appends [B3, B4, B5] channels.
      - [[4, 6, 10], 1, v11Segment, [nc, 32, 256, 0.25, 16, 0.0, 0.5, bilinear]]
    """
).strip() + "\n"


def register_csf_head() -> None:
    """Register the composite head before YAML parsing or checkpoint loading."""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.modules.head as head
    import ultralytics.nn.tasks as tasks

    head.v11Segment = CrackPathCSFSegment
    modules.v11Segment = CrackPathCSFSegment
    # ``parse_model`` resolves YAML strings from the globals of tasks.py.
    tasks.v11Segment = CrackPathCSFSegment


# Register on import as well as in the CLI.  This is important for Ultralytics'
# generated DDP worker script, which imports the custom trainer below in a fresh
# Python process before rebuilding the model from the embedded YAML.
register_csf_head()


from ultralytics.models.yolo.segment.train import SegmentationTrainer  # noqa: E402


class CrackPathSegmentationTrainer(SegmentationTrainer):
    """DDP-safe trainer that registers the composite CSF head in every worker."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        register_csf_head()
        return super().get_model(cfg=cfg, weights=weights, verbose=verbose)


CrackPathSegmentationTrainer.__module__ = MODULE_NAME


def write_embedded_yaml(path: str | os.PathLike) -> Path:
    """Materialize MODEL_YAML only when a path-based Ultralytics API needs it."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MODEL_YAML, encoding="utf-8")
    return path


def _temporary_yaml() -> tuple[tempfile.TemporaryDirectory, Path]:
    holder = tempfile.TemporaryDirectory(prefix="crackpath_csf_")
    path = Path(holder.name) / "crackpath-mamba-yolo11-csf-seg.yaml"
    path.write_text(MODEL_YAML, encoding="utf-8")
    return holder, path


def build_model(weights: str = "") -> YOLO:
    """Build from embedded YAML or load a checkpoint with registration active."""
    register_csf_head()
    if weights:
        return YOLO(str(Path(weights).expanduser()))
    holder, yaml_path = _temporary_yaml()
    model = YOLO(str(yaml_path))
    # Keep the temporary directory alive for the lifetime of the wrapper.  The
    # parsed dictionary is already stored in model.model.yaml, but retaining the
    # source also helps downstream logging code that records the cfg path.
    model._csf_yaml_holder = holder
    model._csf_yaml_path = str(yaml_path)
    return model


def smoke_test(imgsz: int = 640, device: str = "cpu") -> None:
    model = build_model()
    torch_device = torch.device(device if device == "cpu" else f"cuda:{str(device).split(',')[0]}")
    network = model.model.to(torch_device).eval()
    with torch.no_grad():
        prediction = network(torch.zeros(1, 3, imgsz, imgsz, device=torch_device))
    head = network.model[-1]
    params = sum(parameter.numel() for parameter in network.parameters())
    print(f"OK: {network.__class__.__name__}")
    print(f"Head: {head.__class__.__name__}")
    print(f"Backbone channels: {head.backbone_channels}")
    print(f"CSF-Neck channels: {head.neck_channels}")
    print(f"Parameters: {params:,}")
    print(f"Forward output type: {type(prediction).__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("check", "export-yaml", "train", "val", "predict"),
        default="check",
    )
    parser.add_argument(
        "--yaml-out",
        default="ultralytics/cfg/models/11/10.24-experiments/CrackPath-Mamba-YOLO11-CSF-Seg.yaml",
    )
    parser.add_argument("--weights", default="")
    parser.add_argument("--data", default="crack-seg/data.yaml")
    parser.add_argument("--source", default="")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--project", default="output_dir/crack-seg-csf")
    parser.add_argument("--name", default="CSF00_crack-seg_seed1")
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_csf_head()

    if args.action == "export-yaml":
        print(write_embedded_yaml(args.yaml_out))
        return
    if args.action == "check":
        smoke_test(args.imgsz, args.device)
        return

    model = build_model(args.weights)
    common = dict(device=args.device, imgsz=args.imgsz, batch=args.batch, workers=args.workers)
    if args.action == "train":
        model.train(
            trainer=CrackPathSegmentationTrainer,
            data=args.data,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            optimizer=args.optimizer,
            amp=args.amp,
            project=args.project,
            name=args.name,
            **common,
        )
    elif args.action == "val":
        if not args.weights:
            raise ValueError("--action val requires --weights")
        model.val(data=args.data, split=args.split, **common)
    elif args.action == "predict":
        if not args.weights:
            raise ValueError("--action predict requires --weights")
        if not args.source:
            raise ValueError("--action predict requires --source")
        model.predict(source=args.source, save=True, **common)


if __name__ == "__main__":
    main()
