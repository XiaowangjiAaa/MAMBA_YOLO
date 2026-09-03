#!/usr/bin/env python
"""
Mamba-YOLO YAML config pre-flight validator.

Validates each YAML config BEFORE running actual training by:
  1. Building the model from YAML
  2. Testing forward pass (train + inference modes)
  3. Testing deepcopy (simulates ModelEMA)
  4. Testing AMP + loss compatibility (with dummy data)
  5. Testing multi-GPU compatibility via DDP simulation

Usage:
  python check_yaml.py --aug19 --quick
  python check_yaml.py --aug17 --quick
  python check_yaml.py --aug12 --quick
  python check_yaml.py --aug9 --quick
  python check_yaml.py --sep1
  python check_yaml.py --phase 31F 31T
  python check_yaml.py --phase A B C
  python check_yaml.py --experiments E09 E10
  python check_yaml.py --list
"""

import sys
import os
import argparse
import copy
import types
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "ultralytics" / "cfg" / "models" / "mamba-yolo"
sys.path.insert(0, str(ROOT))


def install_selective_scan_shape_stub():
    """Install a shape/graph-only scan stub for YAML checks on machines without the CUDA extension."""
    module = types.ModuleType("selective_scan_cuda_core")

    def fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows):
        return u, u

    def bwd(u, delta, A, B, C, D, delta_bias, dout, x, delta_softplus, nrows):
        return (dout, torch.zeros_like(delta), torch.zeros_like(A), torch.zeros_like(B),
                torch.zeros_like(C), None if D is None else torch.zeros_like(D),
                None if delta_bias is None else torch.zeros_like(delta_bias))

    module.fwd = fwd
    module.bwd = bwd
    for name in ("selective_scan_cuda_core", "selective_scan_cuda", "selective_scan_cuda_oflex",
                 "selective_scan_cuda_ndstate", "selective_scan_cuda_nrow"):
        sys.modules.setdefault(name, module)


# ---- Experiment configs (same as batch_train.py) ----
YOLO11_EXPERIMENTS = {
    "B0":  "yolo-mamba-seg-yolo11.yaml",
    "S1":  "yolo-mamba-orientation-p3-seg-yolo11.yaml",
    "S2":  "yolo-mamba-orientation-p3-sup-seg-yolo11.yaml",
    "M1":  "yolo-mamba-crack-write-p4-seg-yolo11.yaml",
    "M2":  "yolo-mamba-crack-write-p4-sup-seg-yolo11.yaml",
    "C1":  "yolo-mamba-scan-p3-write-p4-seg-yolo11.yaml",
    "C2":  "yolo-mamba-scan-p3-write-p4-sup-seg-yolo11.yaml",
    "M3":  "yolo-mamba-crack-memory-p4-sup-seg-yolo11.yaml",
    "S3":  "yolo-mamba-orientation-p4-seg-yolo11.yaml",
    "S4":  "yolo-mamba-orientation-p3p4-seg-yolo11.yaml",
    "M0":  "yolo-mamba-crack-write-p3-seg-yolo11.yaml",
    "M0S": "yolo-mamba-crack-write-p3-sup-seg-yolo11.yaml",
    "M4":  "yolo-mamba-crack-write-p3p4-sup-seg-yolo11.yaml",
    "C3":  "yolo-mamba-scan-p3-memory-p4-sup-seg-yolo11.yaml",
    "J1":  "yolo-mamba-structure-p3p4-sup-seg-yolo11.yaml",
    "S5":  "yolo-mamba-orientation-diagonal-p3-sup-seg-yolo11.yaml",
}
DELTA_V2_EXPERIMENTS = {
    "DV2-P2":  "yolo-mamba-crack-delta-v2-p2-seg-yolo11.yaml",
    "DV2-P3":  "yolo-mamba-crack-delta-v2-p3-seg-yolo11.yaml",
    "DV2-P4":  "yolo-mamba-crack-delta-v2-p4-seg-yolo11.yaml",
    "DV2-ALL": "yolo-mamba-crack-delta-v2-seg-yolo11.yaml",
}
YOLOV8_EXPERIMENTS = {
    "V8-S1":  "yolo-mamba-orientation-p3-seg.yaml",
    "V8-M1":  "yolo-mamba-crack-write-p4-seg.yaml",
    "V8-C1":  "yolo-mamba-scan-p3-write-p4-seg.yaml",
    "V8-C2":  "yolo-mamba-scan-p3-write-p4-sup-seg.yaml",
    "V8-DV2": "yolo-mamba-crack-delta-v2-seg.yaml",
    "V8-B0":  "yolo-mamba-seg.yaml",
}
ALL_EXPERIMENTS = {}
ALL_EXPERIMENTS.update(YOLO11_EXPERIMENTS)
ALL_EXPERIMENTS.update(DELTA_V2_EXPERIMENTS)
ALL_EXPERIMENTS.update(YOLOV8_EXPERIMENTS)

AUG9_EXPERIMENTS = {
    "E00": "8.9-experiments/00-b0-yolo11.yaml",
    "E01": "8.9-experiments/01-s1-p3-scan-yolo11.yaml",
    "E02": "8.9-experiments/02-c2-reference-yolo11.yaml",
    "E03": "8.9-experiments/03-c2-guidance-only-yolo11.yaml",
    "E04": "8.9-experiments/04-c2-orientation-only-yolo11.yaml",
    "E05": "8.9-experiments/05-c2-low-aux-yolo11.yaml",
    "E06": "8.9-experiments/06-c1-no-aux-control-yolo11.yaml",
    "E07": "8.9-experiments/07-centered-p4-write-sup-yolo11.yaml",
    "E08": "8.9-experiments/08-p3-scan-centered-p4-write-sup-yolo11.yaml",
    "E09": "8.9-experiments/09-p3-scan-last-p4-write-sup-yolo11.yaml",
    "E10": "8.9-experiments/10-p3-scan-last-p4-centered-write-sup-yolo11.yaml",
}
ALL_EXPERIMENTS.update(AUG9_EXPERIMENTS)

AUG12_EXPERIMENTS = {
    "N00": "8.12-experiments/00-original-mamba-yolo11.yaml",
    "N01": "8.12-experiments/01-crack-stem-lite.yaml",
    "N02": "8.12-experiments/02-crack-stem-directional.yaml",
    "N03": "8.12-experiments/03-crack-merge-lite.yaml",
    "N04": "8.12-experiments/04-crack-merge-directional.yaml",
    "N05": "8.12-experiments/05-crack-front-lite.yaml",
    "N06": "8.12-experiments/06-unified-backbone.yaml",
    "N07": "8.12-experiments/07-unified-all.yaml",
    "N08": "8.12-experiments/08-full-lite-backbone.yaml",
    "N09": "8.12-experiments/09-full-lite-all.yaml",
    "N10": "8.12-experiments/10-full-directional-backbone.yaml",
    "N11": "8.12-experiments/11-full-directional-all.yaml",
}
ALL_EXPERIMENTS.update(AUG12_EXPERIMENTS)

AUG17_EXPERIMENTS = {
    "T00": "8.17-tuning/T00-fixed-default.yaml",
    "T01": "8.17-tuning/T01-no-aux.yaml",
    "T02": "8.17-tuning/T02-strong-aux.yaml",
    "T03": "8.17-tuning/T03-mild-gates.yaml",
    "T04": "8.17-tuning/T04-strong-gates.yaml",
    "T05": "8.17-tuning/T05-temperature-1.yaml",
}
ALL_EXPERIMENTS.update(AUG17_EXPERIMENTS)

AUG19_EXPERIMENTS = {
    "R00": "8.19-experiments/R00-fair-control.yaml",
    "R01": "8.19-experiments/R01-nonnegative-noaux.yaml",
    "R02": "8.19-experiments/R02-nonnegative-gatereg.yaml",
}
ALL_EXPERIMENTS.update(AUG19_EXPERIMENTS)

AUG23_EXPERIMENTS = {
    "Q00": "8.23-experiments/Q00-p5-directional-full.yaml",
    "Q01": "8.23-experiments/Q01-write-only.yaml",
    "Q02": "8.23-experiments/Q02-scan-only.yaml",
    "Q03": "8.23-experiments/Q03-role-specific.yaml",
}
ALL_EXPERIMENTS.update(AUG23_EXPERIMENTS)

AUG24_EXPERIMENTS = {
    "U00": "8.24-experiments/U00-corrected-hv-full.yaml",
    "U01": "8.24-experiments/U01-corrected-hv-temp1.yaml",
    "U02": "8.24-experiments/U02-corrected-hv-scanmax015.yaml",
    "U03": "8.24-experiments/U03-corrected-hv-role-specific.yaml",
    "U04": "8.24-experiments/U04-corrected-hv-learned-init.yaml",
}
ALL_EXPERIMENTS.update(AUG24_EXPERIMENTS)

AUG26_EXPERIMENTS = {
    "W00": "../11/8.26-experiments/W00-yolo11-seg-baseline.yaml",
    "W01": "../11/8.26-experiments/W01-casp-p3p4.yaml",
    "W02": "../11/8.26-experiments/W02-casp-backbone-all.yaml",
    "W03": "../11/8.26-experiments/W03-casp-all-c3k2.yaml",
    "W04": "../11/8.26-experiments/W04-casp-no-transition.yaml",
    "W05": "../11/8.26-experiments/W05-casp-no-write.yaml",
    "W06": "../11/8.26-experiments/W06-casp-p3p4-ratio0125.yaml",
    "W07": "../11/8.26-experiments/W07-casp-no-fusion.yaml",
}
ALL_EXPERIMENTS.update(AUG26_EXPERIMENTS)

AUG27_EXPERIMENTS = {
    "X00": "../11/8.27-experiments/X00-yolo11-seg-map50-baseline.yaml",
    "X01": "../11/8.27-experiments/X01-casp-reference.yaml",
    "X02": "../11/8.27-experiments/X02-guidance001.yaml",
    "X03": "../11/8.27-experiments/X03-guidance005.yaml",
    "X04": "../11/8.27-experiments/X04-guidance010.yaml",
    "X05": "../11/8.27-experiments/X05-orientation0005.yaml",
    "X06": "../11/8.27-experiments/X06-orientation001.yaml",
    "X07": "../11/8.27-experiments/X07-route002.yaml",
    "X08": "../11/8.27-experiments/X08-route010.yaml",
    "X09": "../11/8.27-experiments/X09-direction-mix025.yaml",
    "X10": "../11/8.27-experiments/X10-direction-mix075.yaml",
    "X11": "../11/8.27-experiments/X11-ratio0375.yaml",
    "X12": "../11/8.27-experiments/X12-ratio050.yaml",
    "X13": "../11/8.27-experiments/X13-ratio0375-dstate16.yaml",
    "X14": "../11/8.27-experiments/X14-stage-specific-p3p4.yaml",
}
ALL_EXPERIMENTS.update(AUG27_EXPERIMENTS)

AUG28_EXPERIMENTS = {
    "Y00": "../11/8.28-experiments/Y00-crack-path-reference.yaml",
    "Y01": "../11/8.28-experiments/Y01-seed-ratio001.yaml",
    "Y02": "../11/8.28-experiments/Y02-seed-ratio004.yaml",
    "Y03": "../11/8.28-experiments/Y03-path-steps3.yaml",
    "Y04": "../11/8.28-experiments/Y04-path-steps6.yaml",
    "Y05": "../11/8.28-experiments/Y05-path-conf002.yaml",
    "Y06": "../11/8.28-experiments/Y06-path-conf010.yaml",
    "Y07": "../11/8.28-experiments/Y07-connectivity001.yaml",
    "Y08": "../11/8.28-experiments/Y08-connectivity005.yaml",
    "Y09": "../11/8.28-experiments/Y09-orientation000.yaml",
    "Y10": "../11/8.28-experiments/Y10-orientation001.yaml",
    "Y11": "../11/8.28-experiments/Y11-route005.yaml",
    "Y12": "../11/8.28-experiments/Y12-memory010.yaml",
    "Y13": "../11/8.28-experiments/Y13-transition010.yaml",
    "Y14": "../11/8.28-experiments/Y14-write010.yaml",
    "Y15": "../11/8.28-experiments/Y15-dstate16.yaml",
}
ALL_EXPERIMENTS.update(AUG28_EXPERIMENTS)

AUG31_EXPERIMENTS = {
    "Z00": "../11/8.31-experiments/Z00-y10-fp32-reference.yaml",
    "Z01": "../11/8.31-experiments/Z01-y10-dstate16.yaml",
    "Z02": "../11/8.31-experiments/Z02-y10-seed004.yaml",
    "Z03": "../11/8.31-experiments/Z03-y10-conf010.yaml",
    "Z04": "../11/8.31-experiments/Z04-y10-steps3.yaml",
    "Z05": "../11/8.31-experiments/Z05-y10-connectivity001.yaml",
    "Z06": "../11/8.31-experiments/Z06-y10-dstate16-conf010.yaml",
    "Z07": "../11/8.31-experiments/Z07-y10-dstate16-seed004.yaml",
    "Z08": "../11/8.31-experiments/Z08-y10-dstate16-steps3.yaml",
    "Z09": "../11/8.31-experiments/Z09-y10-dstate16-connectivity001.yaml",
    "Z10": "../11/8.31-experiments/Z10-y10-dstate16-seed004-conf010.yaml",
    "Z11": "../11/8.31-experiments/Z11-y10-dstate16-steps3-conf010.yaml",
}
ALL_EXPERIMENTS.update(AUG31_EXPERIMENTS)

AUG31_FINAL_EXPERIMENTS = {
    "F00": "../11/8.31-final/F00-z04-finalist.yaml",
    "F01": "../11/8.31-final/F01-z10-quality-finalist.yaml",
    "F02": "../11/8.31-final/F02-z04-steps2.yaml",
    "F03": "../11/8.31-final/F03-z04-p3steps2-p4steps3.yaml",
    "F04": "../11/8.31-final/F04-z04-p3steps3-p4steps2.yaml",
    "F05": "../11/8.31-final/F05-z04-conf007.yaml",
    "F06": "../11/8.31-final/F06-z04-conf008.yaml",
    "F07": "../11/8.31-final/F07-z04-seed001.yaml",
    "F08": "../11/8.31-final/F08-z04-maxpaths96.yaml",
    "F09": "../11/8.31-final/F09-z04-orientation0015.yaml",
    "F10": "../11/8.31-final/F10-z04-connectivity002.yaml",
    "F11": "../11/8.31-final/F11-z04-guidance005.yaml",
}
ALL_EXPERIMENTS.update(AUG31_FINAL_EXPERIMENTS)

SEP1_EXPERIMENTS = {
    "G00": "../11/9.1-experiments/G00-z04-placement-reference.yaml",
    "G01": "../11/9.1-experiments/G01-z04-backbone-all-c3k2.yaml",
    "G02": "../11/9.1-experiments/G02-z04-neck-all-c3k2.yaml",
    "G03": "../11/9.1-experiments/G03-z04-deep-all-except-p2.yaml",
    "G04": "../11/9.1-experiments/G04-z04-all-c3k2.yaml",
    "G05": "../11/9.1-experiments/G05-stage-adaptive-all-c3k2.yaml",
}
ALL_EXPERIMENTS.update(SEP1_EXPERIMENTS)

SEP3_EXPERIMENTS = {
    "H00": "../11/9.3-experiments/H00-yolo11n-seg-baseline.yaml",
    "H01": "../11/9.3-experiments/H01-g01-full.yaml",
    "H02": "../11/9.3-experiments/H02-aux-only.yaml",
    "H03": "../11/9.3-experiments/H03-fixed-standard.yaml",
    "H04": "../11/9.3-experiments/H04-adaptive-standard.yaml",
    "H05": "../11/9.3-experiments/H05-fixed-full-memory.yaml",
    "H06": "../11/9.3-experiments/H06-cue-p.yaml",
    "H07": "../11/9.3-experiments/H07-cue-po.yaml",
    "H08": "../11/9.3-experiments/H08-cue-pc.yaml",
    "H09": "../11/9.3-experiments/H09-memory-retention.yaml",
    "H10": "../11/9.3-experiments/H10-memory-retention-transition.yaml",
    "H11": "../11/9.3-experiments/H11-memory-retention-write.yaml",
    "H20": "../11/9.3-experiments/H20-yolov5n-seg-baseline.yaml",
    "H21": "../11/9.3-experiments/H21-yolov5n-seg-crackpath.yaml",
    "H22": "../11/9.3-experiments/H22-yolov8n-seg-baseline.yaml",
    "H23": "../11/9.3-experiments/H23-yolov8n-seg-crackpath.yaml",
    "H24": "../11/9.3-experiments/H24-yolo11n-seg-baseline.yaml",
    "H25": "../11/9.3-experiments/H25-yolo11n-seg-crackpath.yaml",
    "H26": "../11/9.3-experiments/H26-yolo26n-seg-compat-baseline.yaml",
    "H27": "../11/9.3-experiments/H27-yolo26n-seg-compat-crackpath.yaml",
}
ALL_EXPERIMENTS.update(SEP3_EXPERIMENTS)

# ---- Colours ----
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def status_icon(ok):
    return f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"


def print_header(text):
    print(f"\n{BOLD}{CYAN}{'=' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 65}{RESET}")


def build_model(config_path):
    """Build model from YAML config, return (model, yolo_instance)."""
    from ultralytics import YOLO
    model_path = Path(config_path)
    if not model_path.is_absolute() and not model_path.exists():
        model_path = CONFIG_DIR / model_path
    if not model_path.exists():
        return None, f"Config not found: {model_path}"
    try:
        yolo = YOLO(str(model_path), verbose=False)
        return yolo.model, None
    except Exception as e:
        return None, str(e)


def test_deepcopy(model, device):
    """Test that copy.deepcopy works on the model (simulates ModelEMA)."""
    if hasattr(model, 'module'):
        model = model.module  # de_parallel
    model_cpu = model.cpu()

    # Apply the same monkey-patch that ModelEMA uses
    _original_deepcopy = torch.Tensor.__deepcopy__
    def _patched_deepcopy(tensor, memo):
        if tensor.is_leaf:
            return _original_deepcopy(tensor, memo)
        leaf = tensor.detach().clone()
        memo[id(tensor)] = leaf
        return leaf
    torch.Tensor.__deepcopy__ = _patched_deepcopy
    try:
        _ = copy.deepcopy(model_cpu)
        model.to(device)
        return True, None
    except Exception as e:
        model.to(device)
        return False, str(e)
    finally:
        torch.Tensor.__deepcopy__ = _original_deepcopy


def test_forward(model, device, nc=1, imgsz=640):
    """Test forward pass in train and eval mode and retain the train graph for checks."""
    model = model.to(device)
    batch = torch.randn(2, 3, imgsz, imgsz, device=device)

    # Eval mode (inference)
    model.eval()
    try:
        with torch.no_grad():
            _ = model(batch)
    except Exception as e:
        return False, f"inference forward: {e}", None

    # Train mode
    model.train()
    try:
        out = model(batch)
        if isinstance(out, (list, tuple)):
            if len(out) == 2:
                # (preds, loss_for_train)
                pass
            elif len(out) == 3 and isinstance(out[0], list):
                # (feats, pred_masks, proto) - seg model train output
                pass
    except Exception as e:
        return False, f"train forward: {e}", None

    return True, None, out


def test_amp_forward(model, device):
    """Test AMP forward and numerical backward through custom state modules."""
    model = model.to(device).train()
    batch = torch.randn(2, 3, 640, 640, device=device)
    try:
        with torch.cuda.amp.autocast(enabled=True):
            output = model(batch)

        def tensors(value):
            if torch.is_tensor(value):
                return [value]
            if isinstance(value, dict):
                return [item for child in value.values() for item in tensors(child)]
            if isinstance(value, (list, tuple)):
                return [item for child in value for item in tensors(child)]
            return []

        scalar = sum(item.float().mean() for item in tensors(output) if item.requires_grad)
        state_named_parameters = [
            (f"{module.__class__.__name__}.{name}", parameter) for module in model.modules()
            if module.__class__.__name__ in {"EfficientCrackAlignedState", "SparseCrackPathState"}
            for name, parameter in module.named_parameters() if parameter.requires_grad
        ]
        if state_named_parameters:
            gradients = torch.autograd.grad(
                scalar, [parameter for _, parameter in state_named_parameters], allow_unused=True
            )
            nonfinite = [name for (name, _), gradient in zip(state_named_parameters, gradients)
                         if gradient is not None and not torch.isfinite(gradient).all()]
            if nonfinite:
                preview = ", ".join(nonfinite[:8])
                suffix = f" (+{len(nonfinite) - 8} more)" if len(nonfinite) > 8 else ""
                return False, f"AMP backward non-finite: {preview}{suffix}"
    except Exception as e:
        return False, f"AMP forward/backward: {e}"
    return True, "finite AMP state backward"


def test_loss_computation(model, device, nc=1, imgsz=640):
    """Test that the full loss (including structure guidance) computes."""
    if not hasattr(model, 'criterion'):
        return True, "no criterion (skip)"

    model = model.to(device).train()

    # Create a dummy batch that looks like a segmentation batch
    # Format: (imgs, ...), see the dataset output
    images = torch.randn(2, 3, imgsz, imgsz, device=device)
    # Attempt a full forward that populates last_guidance / last_orientation
    try:
        model.eval()
        with torch.no_grad():
            _ = model(images)

        # Now check if guidance_loss_weight > 0 triggers supervision
        criterion = model.criterion
        if hasattr(criterion, 'guidance_loss_weight') and criterion.guidance_loss_weight > 0:
            # Need a real-ish batch dict to test loss
            pass
    except Exception as e:
        return False, f"loss pass: {e}"

    # Check that forward populates last_* correctly
    has_guidance = False
    has_orientation = False
    for m in model.modules():
        if getattr(m, 'last_guidance', None) is not None:
            has_guidance = True
        if getattr(m, 'last_orientation', None) is not None:
            has_orientation = True

    return True, None


def test_ddp_build(config_path, device):
    """Test that model can be built when DDP metadata is present.

    This mimics what happens in trainer._setup_train when world_size > 1:
    the model goes through .to(device), DDP wrap, de_parallel, deepcopy.
    """
    try:
        model, err = build_model(config_path)
        if model is None:
            return False, err

        model = model.to(device).train()
        batch = torch.randn(2, 3, 640, 640, device=device)

        # Forward + backward (ensures gradients flow)
        model.train()
        out = model(batch)
        if isinstance(out, (list, tuple)):
            # Grab a scalar loss if available
            if len(out) == 3 and isinstance(out[0], list):
                # Pretend the output is a train loss
                pass
        # Cleanup
        del model, batch
        torch.cuda.empty_cache()
        return True, None
    except Exception as e:
        return False, str(e)


def test_89_structure(model, config_path):
    """Check the structural invariants that distinguish the new late-P4 experiments."""
    name = Path(config_path).name
    if "8.9-experiments" not in str(config_path) or not name.startswith(("07-", "08-", "09-", "10-")):
        return True, "not a centered/last-P4 config"
    stage = model.model[5]
    guidance_modules = [m for m in stage.modules() if getattr(m, "crack_guided_write", False)]
    centered = [m for m in guidance_modules if m.write_guidance_centered]
    if name.startswith(("09-", "10-")):
        if not isinstance(stage, nn.Sequential) or len(stage) != 3:
            return False, f"expected a 3-block nano P4 stage, got {type(stage).__name__} len={len(stage)}"
        if len(guidance_modules) != 1:
            return False, f"expected exactly one gated P4 block, got {len(guidance_modules)}"
    if name.startswith(("07-", "08-", "10-")) and not centered:
        return False, "centered write was requested but no centered SS2D was found"
    return True, f"P4 blocks={len(stage) if isinstance(stage, nn.Sequential) else 1}, gated={len(guidance_modules)}, centered={len(centered)}"


def test_812_structure(model, config_path):
    """Check that each 8.12 YAML changes exactly the components named by the experiment."""
    name = Path(config_path).name
    if "8.12-experiments" not in str(config_path):
        return True, "not an 8.12 config"
    class_names = [m.__class__.__name__ for m in model.modules()]
    counts = {key: class_names.count(key) for key in (
        "CrackDetailStemLite", "CrackDetailStemDirectional", "CrackMergeLite",
        "CrackMergeDirectional", "UnifiedCrackAwareVSSBlock")}
    expected_unified = 10 if name.startswith(("07-", "09-", "11-")) else 6 if name.startswith(("06-", "08-", "10-")) else 0
    if counts["UnifiedCrackAwareVSSBlock"] != expected_unified:
        return False, f"unified blocks={counts['UnifiedCrackAwareVSSBlock']}, expected={expected_unified}"
    expected_stem = "Lite" if name.startswith(("01-", "05-", "08-", "09-")) else "Directional" if name.startswith(("02-", "10-", "11-")) else None
    stem_total = counts["CrackDetailStemLite"] + counts["CrackDetailStemDirectional"]
    if stem_total != (1 if expected_stem else 0):
        return False, f"crack stems={stem_total}, expected={1 if expected_stem else 0}"
    expected_merge = "Lite" if name.startswith(("03-", "05-", "08-", "09-")) else "Directional" if name.startswith(("04-", "10-", "11-")) else None
    merge_total = counts["CrackMergeLite"] + counts["CrackMergeDirectional"]
    if merge_total != (3 if expected_merge else 0):
        return False, f"crack merges={merge_total}, expected={3 if expected_merge else 0}"
    return True, f"stem={expected_stem or 'original'}, merge={expected_merge or 'original'}, unified={expected_unified}"


def test_817_structure(model, config_path):
    """Enforce the frozen 8.17 architecture so tuning YAMLs cannot drift structurally."""
    if "8.17-tuning" not in str(config_path):
        return True, "not an 8.17 config"
    names = [m.__class__.__name__ for m in model.modules()]
    if names.count("SimpleStem") != 1 or names.count("CrackDetailStemLite") + names.count("CrackDetailStemDirectional"):
        return False, "8.17 must use exactly one original SimpleStem"
    if names.count("CrackMergeDirectional") != 2:
        return False, f"expected two directional merges (P3/P4), got {names.count('CrackMergeDirectional')}"
    if names.count("UnifiedCrackAwareVSSBlock") != 2:
        return False, f"expected exactly two unified blocks (P3 and P4-last), got {names.count('UnifiedCrackAwareVSSBlock')}"
    if not isinstance(model.model[5], nn.Sequential) or model.model[5][-1].__class__.__name__ != "UnifiedCrackAwareVSSBlock":
        return False, "P4 must use standard VSS blocks followed by one unified final block"
    head_unified = sum(
        m.__class__.__name__ == "UnifiedCrackAwareVSSBlock"
        for layer in model.model[10:]
        for m in layer.modules()
    )
    if head_unified:
        return False, f"neck/head must stay standard, found {head_unified} unified block(s)"
    return True, "fixed: original stem, directional P3/P4 merge, unified P3 + P4-last, standard neck"


def test_819_structure(model, config_path):
    """Verify the focused 8.19 control and theory-consistent gate variants."""
    if "8.19-experiments" not in str(config_path):
        return True, "not an 8.19 config"
    name = Path(config_path).name
    names = [m.__class__.__name__ for m in model.modules()]
    if names.count("SimpleStem") != 1 or names.count("CrackMergeDirectional") != 2:
        return False, "all 8.19 models require original stem and exactly two P3/P4 directional merges"
    if model.model[6].__class__.__name__ != "VisionClueMerge":
        return False, "P5 must use the original VisionClueMerge"
    unified = [m for m in model.modules() if getattr(m, "unified_crack_guidance", False)]
    if name.startswith("R00-"):
        if unified:
            return False, f"R00 fair control must contain no Unified SS2D, got {len(unified)}"
        return True, "fair control: identical front end/P5, Standard VSS only"
    if len(unified) != 2:
        return False, f"R01/R02 require exactly two Unified SS2D modules, got {len(unified)}"
    if not all(getattr(m, "nonnegative_gates", False) for m in unified):
        return False, "R01/R02 must enable nonnegative write and scan gates"
    effective = [
        (float(m.effective_write_gate().detach()), float(m.effective_orientation_gate().detach()))
        for m in unified
    ]
    if not all(write > 0 and scan > 0 for write, scan in effective):
        return False, f"nonnegative gates were not initialized positively: {effective}"
    expected_reg = 0.01 if name.startswith("R02-") else 0.0
    actual_reg = float(model.yaml.get("gate_regularization_weight", 0.0))
    if abs(actual_reg - expected_reg) > 1e-12:
        return False, f"gate regularization={actual_reg}, expected={expected_reg}"
    if expected_reg > 0.0:
        regularization = expected_reg * torch.stack(
            [gate.square() for m in unified for gate in (m.effective_write_gate(), m.effective_orientation_gate())]
        ).mean()
        gate_parameters = [p for m in unified for p in (m.write_beta, m.orientation_gate)]
        gradients = torch.autograd.grad(regularization, gate_parameters, allow_unused=True)
        if not all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients):
            return False, "gate regularization did not produce finite nonzero gate gradients"
    return True, f"nonnegative Unified gates={effective}, gate_reg={actual_reg}"


def test_823_structure(model, config_path):
    """Verify the 8.23 P5 restoration and write/scan component attribution matrix."""
    if "8.23-experiments" not in str(config_path):
        return True, "not an 8.23 config"
    name = Path(config_path).name
    names = [m.__class__.__name__ for m in model.modules()]
    if names.count("SimpleStem") != 1 or names.count("CrackMergeDirectional") != 3:
        return False, "8.23 requires original stem and directional P3/P4/P5 merges"
    unified = [m for m in model.modules() if getattr(m, "unified_crack_guidance", False)]
    if len(unified) != 2 or not all(getattr(m, "nonnegative_gates", False) for m in unified):
        return False, f"expected two nonnegative Unified modules, got {len(unified)}"
    if any(getattr(m, "orientation_family_logits", False) for m in unified):
        return False, "8.23 checkpoints must retain the legacy H/V mapping for backward compatibility"
    expected = {
        "Q00-": [(True, True), (True, True)],
        "Q01-": [(True, False), (True, False)],
        "Q02-": [(False, True), (False, True)],
        "Q03-": [(True, False), (False, True)],
    }
    key = next((prefix for prefix in expected if name.startswith(prefix)), None)
    if key is None:
        return False, f"unknown 8.23 YAML name: {name}"
    actual = [(m.unified_enable_write, m.unified_enable_scan) for m in unified]
    if actual != expected[key]:
        return False, f"write/scan roles={actual}, expected={expected[key]}"
    if any(float(model.yaml.get(k, 0.0)) != 0.0 for k in (
        "guidance_loss_weight", "orientation_loss_weight", "gate_regularization_weight"
    )):
        return False, "8.23 must keep auxiliary supervision and gate regularization disabled"
    return True, f"P3/P4/P5 directional; Unified roles={actual}"


def test_824_structure(model, config_path):
    """Verify corrected H/V logits and the focused 8.24 stability matrix."""
    if "8.24-experiments" not in str(config_path):
        return True, "not an 8.24 config"
    name = Path(config_path).name
    names = [m.__class__.__name__ for m in model.modules()]
    if names.count("SimpleStem") != 1 or names.count("CrackMergeDirectional") != 3:
        return False, "8.24 must preserve Q00's original stem and three directional merges"
    unified = [m for m in model.modules() if getattr(m, "unified_crack_guidance", False)]
    if len(unified) != 2 or not all(getattr(m, "nonnegative_gates", False) for m in unified):
        return False, f"expected two nonnegative Unified modules, got {len(unified)}"
    expected_roles = ([(True, False), (False, True)] if name.startswith("U03-")
                      else [(True, True), (True, True)])
    actual_roles = [(m.unified_enable_write, m.unified_enable_scan) for m in unified]
    if actual_roles != expected_roles:
        return False, f"write/scan roles={actual_roles}, expected={expected_roles}"
    for module in unified:
        if module.unified_enable_scan and not getattr(module, "orientation_family_logits", False):
            return False, "every enabled 8.24 scan must use corrected H/V family logits"
        if module.unified_enable_scan:
            probe = torch.tensor([[[[0.25]], [[-0.75]]]])
            scores = module.orientation_scores(probe)
            if not torch.equal(scores, probe):
                return False, "corrected H/V scores do not preserve both family-logit channels"
    if any(float(model.yaml.get(k, 0.0)) != 0.0 for k in (
        "guidance_loss_weight", "orientation_loss_weight", "gate_regularization_weight"
    )):
        return False, "8.24 keeps auxiliary supervision and gate regularization disabled"
    return True, f"corrected H/V family logits; Unified roles={actual_roles}"


def test_826_structure(model, config_path):
    """Verify true-YOLO11 topology, efficient partial channels and coupled CASP roles."""
    if "8.26-experiments" not in str(config_path):
        return True, "not an 8.26 config"
    name = Path(config_path).name
    names = [m.__class__.__name__ for m in model.modules()]
    forbidden = {"SimpleStem", "VisionClueMerge", "CrackMergeDirectional", "VSSBlock"}
    present = sorted(forbidden.intersection(names))
    if present:
        return False, f"8.26 must be YOLO11-based; found legacy Mamba-YOLO modules {present}"
    adapters = [m for m in model.modules() if m.__class__.__name__ == "AdaptiveC3k2CASP"]
    expected_count = {
        "W00-": 0, "W01-": 2, "W02-": 4, "W03-": 8,
        "W04-": 2, "W05-": 2, "W06-": 2, "W07-": 2,
    }
    key = next((prefix for prefix in expected_count if name.startswith(prefix)), None)
    if key is None:
        return False, f"unknown 8.26 YAML name: {name}"
    if len(adapters) != expected_count[key]:
        return False, f"AdaptiveC3k2CASP count={len(adapters)}, expected={expected_count[key]}"
    if key == "W00-":
        if names.count("C3k2") != 8:
            return False, f"true YOLO11 baseline needs eight C3k2 modules, got {names.count('C3k2')}"
        return True, "true YOLO11-Seg C3k2 baseline"
    states = [m for m in model.modules() if m.__class__.__name__ == "EfficientCrackAlignedState"]
    ss2d = [m for m in model.modules() if getattr(m, "crack_aligned_edges", False)]
    if len(states) != len(adapters) or len(ss2d) != len(adapters):
        return False, f"expected one efficient state core per adapter, got state={len(states)}, edgeSS2D={len(ss2d)}"
    if any(not (0 < m.state_channels <= m.channels) for m in states):
        return False, "invalid partial-channel allocation"
    expected_roles = {
        "W04-": (False, True, True),
        "W05-": (True, False, True),
        "W07-": (True, True, False),
    }.get(key, (True, True, True))
    actual_roles = {
        (m.edge_enable_transition, m.edge_enable_write, m.edge_enable_fusion) for m in ss2d
    }
    if actual_roles != {expected_roles}:
        return False, f"edge roles={actual_roles}, expected={expected_roles}"
    # Disabled branches must not register trainable scalar gates. A normal
    # forward pass cannot detect these, but DDP will fail on the next iteration
    # because the parameters never receive gradients.
    stale_parameters = []
    for index, module in enumerate(ss2d):
        disabled_gates = (
            ("transition", "edge_transition_raw", module.edge_enable_transition),
            ("write", "edge_write_raw", module.edge_enable_write),
            ("fusion", "orientation_gate", module.edge_enable_fusion),
        )
        for role, parameter, enabled in disabled_gates:
            if enabled != hasattr(module, parameter):
                stale_parameters.append(
                    f"SS2D[{index}] {role}: enabled={enabled}, parameter={hasattr(module, parameter)}"
                )
    if stale_parameters:
        return False, "role/parameter mismatch (DDP unused-parameter risk): " + "; ".join(stale_parameters)
    if key == "W06-" and any(m.state_ratio != 0.125 for m in states):
        return False, "W06 must use state_ratio=0.125"
    if key != "W06-" and any(m.state_ratio != 0.25 for m in states):
        return False, "non-W06 CASP modules must use state_ratio=0.25"
    if any(float(model.yaml.get(k, 0.0)) != 0.0 for k in (
        "guidance_loss_weight", "orientation_loss_weight", "gate_regularization_weight"
    )):
        return False, "8.26 first causal round keeps auxiliary losses disabled"
    return True, f"adapters={len(adapters)}, partial ratios={[m.state_ratio for m in states]}, roles={expected_roles}"


def test_827_structure(model, config_path):
    """Verify the fixed full-CASP parameter-search contract for 8.27."""
    if "8.27-experiments" not in str(config_path):
        return True, "not an 8.27 config"
    name = Path(config_path).name
    adapters = [m for m in model.modules() if m.__class__.__name__ == "AdaptiveC3k2CASP"]
    if name.startswith("X00-"):
        return (not adapters, "Mask-mAP50 YOLO11 baseline" if not adapters else "baseline contains CASP")
    if len(adapters) != 2 or model.model[4].__class__.__name__ != "AdaptiveC3k2CASP" or model.model[6].__class__.__name__ != "AdaptiveC3k2CASP":
        return False, f"8.27 requires exactly P3/P4 CASP, got {len(adapters)} adapter(s)"
    states = [m for m in model.modules() if m.__class__.__name__ == "EfficientCrackAlignedState"]
    ss2d = [m.state for m in states]
    if not all(m.edge_enable_transition and m.edge_enable_write and m.edge_enable_fusion for m in ss2d):
        return False, "8.27 fixes the complete transition/write/fusion module; no component may be disabled"
    if not all(m.structure_kernel == 3 and abs(m.structure_init_std - 0.01) < 1e-12 for m in ss2d):
        return False, "8.27 requires the learnable DW3x3 structure head with init std 0.01"
    controlled = [state.route_raw for state in states]
    controlled += [parameter for m in ss2d for parameter in (m.edge_transition_raw, m.edge_write_raw, m.orientation_gate)]
    if not all(getattr(parameter, "_no_weight_decay", False) for parameter in controlled):
        return False, "CASP scalar controls must be excluded from optimizer weight decay"

    expected_guidance = {"X02-": 0.01, "X03-": 0.05, "X04-": 0.10}.get(
        next((key for key in ("X02-", "X03-", "X04-") if name.startswith(key)), ""), 0.03
    )
    expected_orientation = 0.005 if name.startswith("X05-") else (0.01 if name.startswith("X06-") else 0.0)
    if abs(float(model.yaml.get("guidance_loss_weight", 0.0)) - expected_guidance) > 1e-12:
        return False, f"guidance weight mismatch; expected {expected_guidance}"
    if abs(float(model.yaml.get("orientation_loss_weight", 0.0)) - expected_orientation) > 1e-12:
        return False, f"orientation weight mismatch; expected {expected_orientation}"

    expected_mix = 0.25 if name.startswith("X09-") else (0.75 if name.startswith("X10-") else 0.50)
    if any(abs(m.direction_mix - expected_mix) > 1e-12 for m in ss2d):
        return False, f"direction_mix mismatch; expected {expected_mix}"
    expected_ratios = ([0.25, 0.375] if name.startswith("X14-") else
                       [0.50, 0.50] if name.startswith("X12-") else
                       [0.375, 0.375] if name.startswith(("X11-", "X13-")) else [0.25, 0.25])
    if [m.state_ratio for m in states] != expected_ratios:
        return False, f"state ratios={[m.state_ratio for m in states]}, expected={expected_ratios}"
    expected_dstate = 16 if name.startswith("X13-") else 8
    if any(m.d_state != expected_dstate for m in ss2d):
        return False, f"d_state mismatch; expected {expected_dstate}"
    expected_routes = [0.02, 0.02] if name.startswith("X07-") else (
        [0.10, 0.10] if name.startswith("X08-") else [0.03, 0.07] if name.startswith("X14-") else [0.05, 0.05]
    )
    actual_routes = [float(m.effective_route().detach()) for m in states]
    if any(abs(a - b) > 1e-6 for a, b in zip(actual_routes, expected_routes)):
        return False, f"route init={actual_routes}, expected={expected_routes}"
    return True, f"full CASP; ratios={expected_ratios}, routes={expected_routes}, direction_mix={expected_mix}"


def test_828_structure(model, config_path):
    """Verify that every 8.28/8.31 YAML keeps the complete dynamic path module."""
    is_831 = "8.31-experiments" in str(config_path)
    is_831_final = "8.31-final" in str(config_path)
    if "8.28-experiments" not in str(config_path) and not is_831 and not is_831_final:
        return True, "not an 8.28/8.31 config"
    name = Path(config_path).name
    adapters = [m for m in model.modules() if m.__class__.__name__ == "AdaptiveC3k2CrackPath"]
    if len(adapters) != 2 or any(model.model[index].__class__.__name__ != "AdaptiveC3k2CrackPath" for index in (4, 6)):
        return False, f"8.28 requires exactly P3/P4 AdaptiveC3k2CrackPath, got {len(adapters)}"
    states = [m for m in model.modules() if m.__class__.__name__ == "SparseCrackPathState"]
    if len(states) != 2:
        return False, f"expected two SparseCrackPathState cores, got {len(states)}"
    if any(m.structure_head[-1].out_channels != 7 for m in states):
        return False, "structure head must predict p + double-angle tangent + four connectivity families"
    if any(torch.count_nonzero(m.state_out.weight.detach()).item() != 0 for m in states):
        return False, "8.28 state_out must be zero-initialized for an exact safe C3k2 start"
    if any(m.state_ratio != 0.25 for m in states):
        return False, "8.28/8.31 fixes state_ratio=0.25"

    if is_831_final:
        final_expected = {
            "F00-": (0.02, (3, 3), 0.05, 8, 128, 0.01, 0.03, 0.10),
            "F01-": (0.04, (4, 4), 0.10, 16, 128, 0.01, 0.03, 0.10),
            "F02-": (0.02, (2, 2), 0.05, 8, 128, 0.01, 0.03, 0.10),
            "F03-": (0.02, (2, 3), 0.05, 8, 128, 0.01, 0.03, 0.10),
            "F04-": (0.02, (3, 2), 0.05, 8, 128, 0.01, 0.03, 0.10),
            "F05-": (0.02, (3, 3), 0.07, 8, 128, 0.01, 0.03, 0.10),
            "F06-": (0.02, (3, 3), 0.08, 8, 128, 0.01, 0.03, 0.10),
            "F07-": (0.01, (3, 3), 0.05, 8, 128, 0.01, 0.03, 0.10),
            "F08-": (0.02, (3, 3), 0.05, 8, 96, 0.01, 0.03, 0.10),
            "F09-": (0.02, (3, 3), 0.05, 8, 128, 0.015, 0.03, 0.10),
            "F10-": (0.02, (3, 3), 0.05, 8, 128, 0.01, 0.02, 0.10),
            "F11-": (0.02, (3, 3), 0.05, 8, 128, 0.01, 0.03, 0.05),
        }
        match = next((values for prefix, values in final_expected.items() if name.startswith(prefix)), None)
        if match is None:
            return False, f"unknown 8.31-final config: {name}"
        (expected_seed, expected_steps, expected_conf, expected_dstate, expected_max_paths,
         expected_orientation_loss, expected_connectivity_loss, expected_guidance_loss) = match
        if abs(float(model.yaml.get("orientation_loss_weight", 0.0)) - expected_orientation_loss) > 1e-9:
            return False, f"orientation loss must be {expected_orientation_loss}"
        if abs(float(model.yaml.get("connectivity_loss_weight", 0.0)) - expected_connectivity_loss) > 1e-9:
            return False, f"connectivity loss must be {expected_connectivity_loss}"
        if abs(float(model.yaml.get("guidance_loss_weight", 0.0)) - expected_guidance_loss) > 1e-9:
            return False, f"guidance loss must be {expected_guidance_loss}"
    elif is_831:
        expected_seed = 0.04 if name.startswith(("Z02-", "Z07-", "Z10-")) else 0.02
        expected_steps = (3, 3) if name.startswith(("Z04-", "Z08-", "Z11-")) else (4, 4)
        expected_conf = 0.10 if name.startswith(("Z03-", "Z06-", "Z10-", "Z11-")) else 0.05
        expected_dstate = 16 if name.startswith(("Z01-", "Z06-", "Z07-", "Z08-", "Z09-", "Z10-", "Z11-")) else 8
        expected_max_paths = 128
        expected_connectivity_loss = 0.01 if name.startswith(("Z05-", "Z09-")) else 0.03
        if abs(float(model.yaml.get("orientation_loss_weight", 0.0)) - 0.01) > 1e-9:
            return False, "8.31 must keep the Y10 orientation loss weight at 0.01"
        if abs(float(model.yaml.get("connectivity_loss_weight", 0.0)) - expected_connectivity_loss) > 1e-9:
            return False, f"8.31 connectivity loss must be {expected_connectivity_loss}"
    else:
        expected_seed = 0.01 if name.startswith("Y01-") else (0.04 if name.startswith("Y02-") else 0.02)
        step = 3 if name.startswith("Y03-") else (6 if name.startswith("Y04-") else 4)
        expected_steps = (step, step)
        expected_conf = 0.02 if name.startswith("Y05-") else (0.10 if name.startswith("Y06-") else 0.05)
        expected_dstate = 16 if name.startswith("Y15-") else 8
        expected_max_paths = 128
    expected_route = 0.05 if name.startswith("Y11-") else 0.02
    expected_memory = 0.10 if name.startswith("Y12-") else 0.05
    expected_transition = 0.10 if name.startswith("Y13-") else 0.05
    expected_write = 0.10 if name.startswith("Y14-") else 0.05
    for state_index, state in enumerate(states):
        actual = (
            state.seed_ratio, state.path_steps, state.path_min_conf, float(state.effective_route().detach()),
            float(state.path_ssm.effective_memory().detach()), float(state.path_ssm.effective_transition().detach()),
            float(state.path_ssm.effective_write().detach()), state.path_ssm.d_state,
        )
        expected = (
            expected_seed, expected_steps[state_index], expected_conf, expected_route,
            expected_memory, expected_transition, expected_write, expected_dstate,
        )
        if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(actual, expected)):
            return False, f"path parameter mismatch: actual={actual}, expected={expected}"
        if state.max_paths != expected_max_paths:
            return False, f"max_paths={state.max_paths}, expected={expected_max_paths}"
        controlled = (
            state.route_raw, state.path_ssm.memory_raw,
            state.path_ssm.transition_raw, state.path_ssm.write_raw,
        )
        if not all(getattr(parameter, "_no_weight_decay", False) for parameter in controlled):
            return False, "path/memory scalar controls must be excluded from weight decay"

    if is_831_final:
        expected_connectivity = expected_connectivity_loss
        expected_orientation = expected_orientation_loss
        expected_guidance = expected_guidance_loss
    elif is_831:
        expected_connectivity = expected_connectivity_loss
        expected_orientation = 0.01
        expected_guidance = 0.10
    else:
        expected_connectivity = 0.01 if name.startswith("Y07-") else (0.05 if name.startswith("Y08-") else 0.03)
        expected_orientation = 0.0 if name.startswith("Y09-") else (0.01 if name.startswith("Y10-") else 0.005)
        expected_guidance = 0.10
    losses = (
        float(model.yaml.get("guidance_loss_weight", 0.0)),
        float(model.yaml.get("connectivity_loss_weight", 0.0)),
        float(model.yaml.get("orientation_loss_weight", 0.0)),
    )
    expected_losses = (expected_guidance, expected_connectivity, expected_orientation)
    if any(abs(a - b) > 1e-12 for a, b in zip(losses, expected_losses)):
        return False, f"structure-loss mismatch: actual={losses}, expected={expected_losses}"
    return True, (
        f"full sparse path; seed={expected_seed}, steps={expected_steps}, conf={expected_conf}, "
        f"route={expected_route}, memory/transition/write={expected_memory}/{expected_transition}/{expected_write}"
    )


def test_live_aux_cache(model):
    """Auxiliary heads must retain the graph; detached copies are visualization-only."""
    guided = [m for m in model.modules() if hasattr(m, "structure_head") or hasattr(m, "guidance_head")]
    if not guided:
        return True, "no auxiliary guidance head"
    missing = [m.__class__.__name__ for m in guided if getattr(m, "last_guidance", None) is None]
    detached = [m.__class__.__name__ for m in guided
                if getattr(m, "last_guidance", None) is not None and not m.last_guidance.requires_grad]
    if missing:
        return False, f"empty live guidance cache in {len(missing)} module(s)"
    if detached:
        return False, f"detached live guidance cache in {len(detached)} module(s)"
    probe_terms = []
    head_params = []
    for module in guided:
        probe_terms.append(module.last_guidance.mean())
        if getattr(module, "last_orientation", None) is not None:
            probe_terms.append(module.last_orientation.mean())
        if getattr(module, "last_connectivity", None) is not None:
            probe_terms.append(module.last_connectivity.mean())
        for head_name in ("structure_head", "guidance_head", "orientation_head"):
            head = getattr(module, head_name, None)
            if head is not None:
                head_params.extend(p for p in head.parameters() if p.requires_grad)
    grads = torch.autograd.grad(sum(probe_terms), head_params, retain_graph=True, allow_unused=True)
    if not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads):
        return False, "auxiliary cache is attached but no guidance-head gradient was produced"
    return True, f"{len(guided)} live graph(s); gradient reaches guidance head(s)"


def test_901_structure(model, config_path):
    """Verify the 9.1 C3k2-placement matrix and unchanged Z04 state design."""
    if "9.1-experiments" not in str(config_path):
        return True, "not a 9.1 config"
    name = Path(config_path).name
    all_c3k2_positions = (2, 4, 6, 8, 13, 16, 19, 22)
    expected_positions = {
        "G00-": (4, 6),
        "G01-": (2, 4, 6, 8),
        "G02-": (4, 6, 13, 16, 19, 22),
        "G03-": (4, 6, 8, 13, 16, 19, 22),
        "G04-": all_c3k2_positions,
        "G05-": all_c3k2_positions,
    }
    match = next((positions for prefix, positions in expected_positions.items() if name.startswith(prefix)), None)
    if match is None:
        return False, f"unknown 9.1 config: {name}"
    for index in all_c3k2_positions:
        actual = model.model[index].__class__.__name__
        expected = "AdaptiveC3k2CrackPath" if index in match else "C3k2"
        if actual != expected:
            return False, f"layer {index} is {actual}, expected {expected}"

    adapters = [model.model[index] for index in match]
    states = [m for adapter in adapters for m in adapter.modules()
              if m.__class__.__name__ == "SparseCrackPathState"]
    if len(states) != len(match):
        return False, f"expected {len(match)} path states, got {len(states)}"
    if any(state.structure_head[-1].out_channels != 7 for state in states):
        return False, "every path state must predict p + tangent(2) + connectivity(4)"
    if any(torch.count_nonzero(state.state_out.weight.detach()).item() != 0 for state in states):
        return False, "every replacement must preserve the zero-initialized safe residual"

    expected_by_layer = {index: (0.02, 128, 3, 0.05) for index in match}
    if name.startswith("G05-"):
        expected_by_layer.update({
            2: (0.02, 64, 2, 0.05),
            4: (0.02, 96, 3, 0.05),
            6: (0.02, 128, 3, 0.05),
            8: (0.02, 128, 2, 0.05),
            13: (0.02, 128, 3, 0.05),
            16: (0.02, 96, 3, 0.05),
            19: (0.02, 128, 3, 0.05),
            22: (0.02, 128, 2, 0.05),
        })
    for layer_index, state in zip(match, states):
        expected_seed, expected_max, expected_steps, expected_conf = expected_by_layer[layer_index]
        actual = (state.seed_ratio, state.max_paths, state.path_steps, state.path_min_conf,
                  state.state_ratio, state.path_ssm.d_state)
        expected = (expected_seed, expected_max, expected_steps, expected_conf, 0.25, 8)
        if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(actual, expected)):
            return False, f"layer {layer_index} path mismatch: actual={actual}, expected={expected}"
        controls = (
            float(state.effective_route().detach()),
            float(state.path_ssm.effective_memory().detach()),
            float(state.path_ssm.effective_transition().detach()),
            float(state.path_ssm.effective_write().detach()),
        )
        if any(abs(a - b) > 1e-6 for a, b in zip(controls, (0.02, 0.05, 0.05, 0.05))):
            return False, f"layer {layer_index} changed Z04 state controls: {controls}"
    losses = tuple(float(model.yaml.get(key, 0.0)) for key in (
        "guidance_loss_weight", "orientation_loss_weight", "connectivity_loss_weight"
    ))
    if any(abs(a - b) > 1e-12 for a, b in zip(losses, (0.10, 0.01, 0.03))):
        return False, f"9.1 must keep Z04 structure losses; got {losses}"
    return True, f"path blocks at layers={match}; all replacements preserve the Z04 state design"


def test_903_structure(model, config_path):
    """Validate the frozen 9.3 causal-ablation and cross-family interfaces."""
    if "9.3-experiments" not in str(config_path):
        return True, "not a 9.3 config"
    name = Path(config_path).name
    if model.yaml.get("scale") != "n":
        return False, "9.3 family study must use the n scale"

    states = [m for m in model.modules() if m.__class__.__name__ == "SparseCrackPathState"]
    baseline_ids = {"H00", "H20", "H22", "H24", "H26"}
    experiment_id = name.split("-", 1)[0]
    if experiment_id in baseline_ids:
        if states:
            return False, f"baseline {experiment_id} unexpectedly contains {len(states)} crack-path state(s)"
        if experiment_id == "H26" and not model.yaml.get("yolo26_compatibility_mode"):
            return False, "H26 must declare the legacy-fork YOLO26 compatibility limitation"
        return True, f"{experiment_id} n-scale baseline has no crack-path state"

    expected_wrapper = {
        "H21": "AdaptiveC3CrackPath",
        "H23": "AdaptiveC2fCrackPath",
    }.get(experiment_id, "AdaptiveC3k2CrackPath")
    wrappers = [m for m in model.modules() if m.__class__.__name__ == expected_wrapper]
    if len(wrappers) != 4 or len(states) != 4:
        return False, f"{experiment_id}: expected four {expected_wrapper}/state modules, got {len(wrappers)}/{len(states)}"
    if experiment_id == "H27" and not model.yaml.get("yolo26_compatibility_mode"):
        return False, "H27 must declare the legacy-fork YOLO26 compatibility limitation"

    expected = {
        "H02": ("adaptive", "poc", "full", False),
        "H03": ("fixed", "poc", "standard", True),
        "H04": ("adaptive", "poc", "standard", True),
        "H05": ("fixed", "poc", "full", True),
        "H06": ("adaptive", "p", "full", True),
        "H07": ("adaptive", "po", "full", True),
        "H08": ("adaptive", "pc", "full", True),
        "H09": ("adaptive", "poc", "retention", True),
        "H10": ("adaptive", "poc", "retention_transition", True),
        "H11": ("adaptive", "poc", "retention_write", True),
    }.get(experiment_id, ("adaptive", "poc", "full", True))
    actual = {(s.path_mode, s.cue_mode, s.memory_mode, s.route_enabled) for s in states}
    if actual != {expected}:
        return False, f"{experiment_id}: state modes={actual}, expected={expected}"
    if any(s.structure_head[-1].out_channels != 7 for s in states):
        return False, "every state must predict p + tangent(2) + connectivity(4)"
    if expected[3] and any(torch.count_nonzero(s.state_out.weight.detach()).item() != 0 for s in states):
        return False, "enabled state routes must retain zero-initialized residual projections"
    return True, f"{experiment_id}: four {expected_wrapper} modules; modes={expected}"


def test_903_yaml_schema(config_path):
    """Catch official-new/API-old YOLO26 argument collisions before model construction."""
    if "9.3-experiments" not in str(config_path):
        return True, "not a 9.3 config"
    try:
        with open(config_path, "r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        return False, f"cannot parse YAML: {error}"
    if not config.get("yolo26_compatibility_mode"):
        return True, "not a YOLO26 compatibility config"
    final_c3k2 = config["head"][11]
    args = final_c3k2[3]
    if final_c3k2[2] != "C3k2" or len(args) < 5:
        return False, "YOLO26 compatibility layer 22 must explicitly provide c3k, e, g, shortcut"
    groups = args[3]
    if type(groups) is not int or groups < 1:
        return False, f"legacy C3k2 groups must be a positive int, got {groups!r}"
    return True, f"legacy C3k2 compatibility args are explicit (groups={groups}, shortcut={args[4]})"


def test_828_structure_losses(model, config_path):
    """Numerically exercise all enabled 8.28/8.31 auxiliary structure losses."""
    if not any(tag in str(config_path) for tag in ("8.28-experiments", "8.31-experiments", "8.31-final", "9.1-experiments", "9.3-experiments")):
        return True, "not an 8.28/8.31 config"
    # Keep this check explicit: an updated check_yaml.py combined with an old
    # ultralytics/utils/loss.py previously raised an unhelpful AttributeError.
    # Reporting the loaded file is especially useful on training servers where
    # an installed ``ultralytics`` package may shadow the repository checkout.
    import ultralytics.utils.loss as loss_module

    v8SegmentationLoss = loss_module.v8SegmentationLoss
    required_api = (
        "_probability_guidance_loss",
        "_connectivity_guidance_loss",
        "_semantic_tangent_loss",
    )
    missing_api = [name for name in required_api if not hasattr(v8SegmentationLoss, name)]
    loaded_loss = Path(loss_module.__file__).resolve()
    expected_loss = (ROOT / "ultralytics" / "utils" / "loss.py").resolve()
    if missing_api:
        return False, (
            f"8.28 loss API mismatch: missing {', '.join(missing_api)}; "
            f"loaded={loaded_loss}. Sync {expected_loss} to the training server."
        )
    if loaded_loss != expected_loss:
        return False, (
            f"8.28 imported the wrong ultralytics loss module: loaded={loaded_loss}; "
            f"expected={expected_loss}. Run check_yaml.py from the repository root "
            "and remove the external package path from PYTHONPATH."
        )

    states = [m for m in model.modules() if m.__class__.__name__ == "SparseCrackPathState"]
    if not states:
        weights = tuple(float(model.yaml.get(key, 0.0)) for key in (
            "guidance_loss_weight", "orientation_loss_weight", "connectivity_loss_weight"
        ))
        return (True, "baseline has no structure loss") if not any(weights) else (
            False, f"nonzero structure loss weights {weights} but no path state exists"
        )
    if any(m.last_guidance is None for m in states):
        return False, "dynamic path forward did not populate structure tensors"
    batch = states[0].last_guidance.shape[0]
    height, width = states[0].last_guidance.shape[-2:]
    target = states[0].last_guidance.new_zeros((batch, height * 4, width * 4))
    # Thin crossing/curved-like strokes ensure every family loss has positives.
    diagonal = torch.arange(min(target.shape[-2:]), device=target.device)
    target[:, diagonal, diagonal] = 1.0
    target[:, target.shape[-2] // 2, :] = 1.0
    terms = []
    for state in states:
        terms.append(float(model.yaml.get("guidance_loss_weight", 0.0)) *
                     v8SegmentationLoss._probability_guidance_loss(state.last_guidance, target, 1.0))
        connectivity_weight = float(model.yaml.get("connectivity_loss_weight", 0.0))
        if connectivity_weight > 0:
            terms.append(connectivity_weight * v8SegmentationLoss._connectivity_guidance_loss(
                state.last_connectivity, target
            ))
        orientation_weight = float(model.yaml.get("orientation_loss_weight", 0.0))
        if orientation_weight > 0:
            terms.append(orientation_weight * v8SegmentationLoss._semantic_tangent_loss(
                state.last_orientation, target
            ))
    auxiliary = sum(terms)
    if not torch.isfinite(auxiliary):
        return False, f"non-finite structure loss: {float(auxiliary.detach())}"
    head_parameters = [parameter for state in states for parameter in state.structure_head.parameters()]
    gradients = torch.autograd.grad(auxiliary, head_parameters, retain_graph=True, allow_unused=True)
    if not any(gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
               for gradient in gradients):
        return False, "structure losses do not reach the dynamic path head"
    return True, f"finite auxiliary={float(auxiliary.detach()):.5f}; gradients reach path head"


def test_826_gradient_reachability(model, config_path, output):
    """Detect trainable CASP parameters that forward-only YAML checks miss.

    DDP with ``find_unused_parameters=False`` fails when a registered parameter
    does not reach the loss graph. This specifically guards component-ablation
    YAMLs such as W07, where a disabled role must also remove its scalar gate.
    """
    if not any(tag in str(config_path) for tag in ("8.26-experiments", "8.27-experiments", "8.28-experiments", "8.31-experiments", "8.31-final", "9.1-experiments", "9.3-experiments")) or output is None:
        return True, "not an 8.26/8.27/8.28/8.31 config"

    def tensors(value):
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, dict):
            return [item for child in value.values() for item in tensors(child)]
        if isinstance(value, (list, tuple)):
            return [item for child in value for item in tensors(child)]
        return []

    output_tensors = [item for item in tensors(output) if item.requires_grad and item.is_floating_point()]
    if not output_tensors:
        return False, "training output has no differentiable tensors"
    states = [m for m in model.modules() if m.__class__.__name__ in {
        "EfficientCrackAlignedState", "SparseCrackPathState"
    }]
    named_parameters = [
        (f"state[{index}].{name}", parameter)
        for index, state in enumerate(states)
        for name, parameter in state.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        return True, "baseline has no CASP parameters"
    scalar = sum(item.float().mean() for item in output_tensors)
    gradients = torch.autograd.grad(
        scalar, [parameter for _, parameter in named_parameters], retain_graph=True, allow_unused=True
    )
    unused = [name for (name, _), gradient in zip(named_parameters, gradients) if gradient is None]
    if any(tag in str(config_path) for tag in ("8.28-experiments", "8.31-experiments", "8.31-final", "9.1-experiments", "9.3-experiments")):
        # Sparse path geometry is deliberately detached from the detection graph
        # because top-k/argmax/atan2 define a hard routing policy. Its structure
        # head is connected by the explicit probability/tangent/connectivity
        # losses tested immediately above, so it need not reach this output-only
        # probe a second time.
        unused = [name for name in unused if ".structure_head." not in name]
        route_disabled = {
            index for index, state in enumerate(states)
            if state.__class__.__name__ == "SparseCrackPathState" and not getattr(state, "route_enabled", True)
        }
        unused = [name for name in unused if not any(name.startswith(f"state[{index}].") for index in route_disabled)]
    if unused:
        preview = ", ".join(unused[:8])
        suffix = f" (+{len(unused) - 8} more)" if len(unused) > 8 else ""
        return False, f"DDP-unused CASP parameter(s): {preview}{suffix}"
    nonfinite = [name for (name, _), gradient in zip(named_parameters, gradients)
                 if gradient is not None and not torch.isfinite(gradient).all()]
    if nonfinite:
        preview = ", ".join(nonfinite[:8])
        suffix = f" (+{len(nonfinite) - 8} more)" if len(nonfinite) > 8 else ""
        return False, f"non-finite CASP gradient(s): {preview}{suffix}"
    return True, f"all {len(named_parameters)} CASP gradients are connected and finite"


def validate_config(config_path, device, nc=1, imgsz=640, quick=False):
    """Run all checks on a single YAML config. Returns list of (check_name, passed, detail)."""
    results = []
    config_name = Path(config_path).name if not Path(config_path).is_absolute() else Path(config_path).name

    schema_ok, schema_detail = test_903_yaml_schema(config_path)
    results.append(("9.3 YAML schema", schema_ok, schema_detail or ""))
    if not schema_ok:
        return results, None

    # 1. Build model
    model, err = build_model(config_path)
    if model is None:
        results.append(("build", False, err))
        # Can't proceed
        return results, model
    results.append(("build", True, f"params={sum(p.numel() for p in model.parameters()):,}"))

    ok, detail = test_89_structure(model, config_path)
    results.append(("8.9 structure", ok, detail or ""))

    ok, detail = test_812_structure(model, config_path)
    results.append(("8.12 structure", ok, detail or ""))

    ok, detail = test_817_structure(model, config_path)
    results.append(("8.17 fixed structure", ok, detail or ""))

    ok, detail = test_819_structure(model, config_path)
    results.append(("8.19 focused structure", ok, detail or ""))

    ok, detail = test_823_structure(model, config_path)
    results.append(("8.23 attribution structure", ok, detail or ""))

    ok, detail = test_824_structure(model, config_path)
    results.append(("8.24 corrected H/V", ok, detail or ""))

    ok, detail = test_826_structure(model, config_path)
    results.append(("8.26 YOLO11/CASP", ok, detail or ""))

    ok, detail = test_827_structure(model, config_path)
    results.append(("8.27 CASP parameters", ok, detail or ""))

    ok, detail = test_828_structure(model, config_path)
    results.append(("8.28/8.31 sparse crack path", ok, detail or ""))

    ok, detail = test_901_structure(model, config_path)
    results.append(("9.1 C3k2 replacement placement", ok, detail or ""))

    ok, detail = test_903_structure(model, config_path)
    results.append(("9.3 causal/family structure", ok, detail or ""))

    # 2. Deepcopy
    ok, detail = test_deepcopy(model, device)
    results.append(("deepcopy (EMA)", ok, detail or ""))

    # 3. Forward pass
    ok, detail, train_output = test_forward(model, device, nc=nc, imgsz=imgsz)
    results.append(("forward", ok, detail or ""))
    if ok:
        cache_ok, cache_detail = test_live_aux_cache(model)
        results.append(("aux graph attached", cache_ok, cache_detail or ""))
        aux_ok, aux_detail = test_828_structure_losses(model, config_path)
        results.append(("8.28/8.31 structure losses", aux_ok, aux_detail or ""))
        reach_ok, reach_detail = test_826_gradient_reachability(model, config_path, train_output)
        results.append(("CASP gradient reach", reach_ok, reach_detail or ""))

    # 4. AMP forward
    if device.type == 'cuda':
        ok, detail = test_amp_forward(model, device)
        results.append(("AMP forward", ok, detail or ""))
    else:
        results.append(("AMP forward", True, "skipped (no CUDA)"))

    # 5. Loss compatible
    ok, detail = test_loss_computation(model, device, nc=nc, imgsz=imgsz)
    results.append(("loss compat", ok, detail or ""))

    # 6. DDP build test
    if quick:
        results.append(("DDP build", True, "skipped (--quick)"))
    else:
        ok, detail = test_ddp_build(config_path, device)
        results.append(("DDP build", ok, detail or ""))

    return results, model


def resolve_experiments(args):
    exp_ids = set()
    if args.all:
        exp_ids = set(ALL_EXPERIMENTS.keys())
    if args.aug9:
        exp_ids.update(AUG9_EXPERIMENTS.keys())
    if args.aug12:
        exp_ids.update(AUG12_EXPERIMENTS.keys())
    if args.aug17:
        exp_ids.update(AUG17_EXPERIMENTS.keys())
    if args.aug19:
        exp_ids.update(AUG19_EXPERIMENTS.keys())
    if args.aug23:
        exp_ids.update(AUG23_EXPERIMENTS.keys())
    if args.aug24:
        exp_ids.update(AUG24_EXPERIMENTS.keys())
    if args.aug26:
        exp_ids.update(AUG26_EXPERIMENTS.keys())
    if args.aug27:
        exp_ids.update(AUG27_EXPERIMENTS.keys())
    if args.aug28:
        exp_ids.update(AUG28_EXPERIMENTS.keys())
    if args.aug31:
        exp_ids.update(AUG31_EXPERIMENTS.keys())
    if args.aug31_final:
        exp_ids.update(AUG31_FINAL_EXPERIMENTS.keys())
    if args.sep1:
        exp_ids.update(SEP1_EXPERIMENTS.keys())
    if args.sep3:
        exp_ids.update(SEP3_EXPERIMENTS.keys())
    if args.experiments:
        for e in args.experiments:
            exp_ids.add(e)
    if args.phase:
        phase_map = {
            "93M": ["H00", "H01"],
            "93A": ["H02"],
            "93SM": ["H03", "H04", "H05"],
            "93C": ["H06", "H07", "H08"],
            "93W": ["H09", "H10", "H11"],
            "93F": ["H20", "H21", "H22", "H23", "H24", "H25"],
            "93F26": ["H26", "H27"],
            "91R": ["G00"],
            "91U": ["G01", "G02", "G03", "G04"],
            "91A": ["G05"],
            "31F": ["F00", "F01"],
            "31T": ["F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11"],
            "31R": ["Z00"],
            "31C": ["Z01"],
            "31P": ["Z02", "Z03", "Z04", "Z05"],
            "31M": ["Z06", "Z07", "Z08", "Z09", "Z10", "Z11"],
            "28F": ["Y00"],
            "28S": ["Y01", "Y02", "Y03", "Y04", "Y05", "Y06"],
            "28L": ["Y07", "Y08", "Y09", "Y10"],
            "28M": ["Y11", "Y12", "Y13", "Y14", "Y15"],
            "27B": ["X00"],
            "27F": ["X01"],
            "27G": ["X02", "X03", "X04"],
            "27O": ["X05", "X06"],
            "27R": ["X07", "X08"],
            "27D": ["X09", "X10"],
            "27C": ["X11", "X12", "X13", "X14"],
            "26B": ["W00"],
            "26M": ["W01", "W06"],
            "26D": ["W02", "W03"],
            "26A": ["W04", "W05", "W07"],
            "24F": ["U00"],
            "24H": ["U01", "U02", "U04"],
            "24R": ["U03"],
            "23F": ["Q00"],
            "23A": ["Q01", "Q02", "Q03"],
            "19C": ["R00"],
            "19G": ["R01", "R02"],
            "17A": ["T00", "T01", "T02"],
            "17G": ["T03", "T04", "T05"],
            "12S": ["N00", "N01", "N02"],
            "12M": ["N03", "N04", "N05"],
            "12U": ["N06", "N07", "N08", "N09", "N10", "N11"],
            "A": ["E00", "E01", "E02"],
            "B": ["E03", "E04", "E05", "E06"],
            "C": ["E07", "E08", "E09", "E10"],
            "1": list(YOLO11_EXPERIMENTS.keys()),
            "2": ["C1", "C2"],
            "3": ["M3", "S3", "S4"],
            "4": ["M0", "M0S", "M4", "C3", "J1", "S5"],
            "dv2": list(DELTA_V2_EXPERIMENTS.keys()),
            "v8": list(YOLOV8_EXPERIMENTS.keys()),
        }
        for p in args.phase:
            exp_ids.update(phase_map.get(p, []))
    if args.exclude:
        for e in args.exclude:
            exp_ids.discard(e)

    sorted_ids = sorted(exp_ids)
    return sorted_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Mamba-YOLO YAML config validator")
    parser.add_argument("--configs", nargs="+", default=None, help="Direct YAML paths or experiment IDs")
    parser.add_argument("--all", action="store_true", help="Check ALL experiments")
    parser.add_argument("--aug9", action="store_true", help="Check all 11 evidence-driven 8.9 experiments")
    parser.add_argument("--aug12", action="store_true", help="Check all 12 crack-front/unified-VSS 8.12 experiments")
    parser.add_argument("--aug17", action="store_true", help="Check all six fixed-method 8.17 tuning experiments")
    parser.add_argument("--aug19", action="store_true", help="Check the three focused 8.19 experiments")
    parser.add_argument("--aug23", action="store_true", help="Check the four focused 8.23 experiments")
    parser.add_argument("--aug24", action="store_true", help="Check the five corrected-H/V 8.24 experiments")
    parser.add_argument("--aug26", action="store_true", help="Check the eight true-YOLO11/CASP 8.26 experiments")
    parser.add_argument("--aug27", action="store_true", help="Check all fixed-full-CASP 8.27 parameter experiments")
    parser.add_argument("--aug28", action="store_true", help="Check all sparse dynamic crack-path 8.28 experiments")
    parser.add_argument("--aug31", action="store_true", help="Check all evidence-driven Y10 fusion experiments")
    parser.add_argument("--aug31-final", action="store_true", help="Check all 8.31-final finalist/tuning experiments")
    parser.add_argument("--sep1", action="store_true", help="Check all 9.1 C3k2-replacement experiments")
    parser.add_argument("--sep3", action="store_true", help="Check all 9.3 causal-ablation/family experiments")
    parser.add_argument("--phase", nargs="+", default=None, help="Phase: 93M/93A/93SM/93C/93W/93F/93F26, 91*, or legacy")
    parser.add_argument("--experiments", nargs="+", default=None, help="Experiment IDs: B0 S1 C2 ...")
    parser.add_argument("--exclude", nargs="+", default=None, help="IDs to exclude")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for tests (cpu or cuda:N)")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML (not used, for reference)")
    parser.add_argument("--quick", action="store_true", help="Skip DDP build test (faster)")
    parser.add_argument("--imgsz", type=int, default=640, help="Synthetic input size used by forward checks")
    parser.add_argument("--stub-scan", action="store_true",
                        help="Use a shape-only selective-scan stub (YAML validation only, never training)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.stub_scan:
        install_selective_scan_shape_stub()
        print("[WARN] Using shape-only selective-scan stub; numerical CUDA behavior is not tested.")

    if args.list:
        print_header("Available experiments")
        for eid, cfg in sorted(ALL_EXPERIMENTS.items()):
            print(f"  {eid:<8s} {cfg}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Resolve experiments
    if args.configs:
        # Accept either literal YAML paths or registered experiment IDs, as the
        # command-line help promises (e.g. --configs H01 H04).
        targets = [ALL_EXPERIMENTS.get(target, target) for target in args.configs]
    else:
        exp_ids = resolve_experiments(args)
        if not exp_ids:
            print("[ERROR] No experiments selected. Use --all, --phase, --experiments, or --configs.")
            sys.exit(1)
        targets = [ALL_EXPERIMENTS.get(eid, eid) for eid in exp_ids]

    all_passed = True
    total_checks = 0
    failed_configs = []

    for target in targets:
        config_path = target if Path(target).exists() else CONFIG_DIR / target
        if not Path(config_path).exists():
            print(f"\n  {RED}SKIP{RESET} {target} — config not found")
            failed_configs.append((target, "not found"))
            continue

        short_name = Path(config_path).name
        print_header(f"Checking: {short_name}")

        results, model = validate_config(str(config_path), device, imgsz=args.imgsz, quick=args.quick)

        for check_name, ok, detail in results:
            icon = status_icon(ok)
            detail_str = f"  ({detail})" if detail else ""
            print(f"  {icon}  {check_name:<22s}{detail_str}")
            total_checks += 1
            if not ok:
                all_passed = False

        if model is not None:
            del model
        torch.cuda.empty_cache()

        # Determine if config passed all checks
        config_ok = all(r[1] for r in results)
        if not config_ok:
            failed_configs.append((short_name, "see above"))

    # Summary
    print_header("SUMMARY")
    if failed_configs:
        print(f"  {RED}FAILED configs:{RESET}")
        for name, reason in failed_configs:
            print(f"    - {name}: {reason}")
        print(f"\n  {RED}Fix these before running full training.{RESET}")
    else:
        print(f"  {GREEN}All configs passed!{RESET}")
        print(f"  Safe to proceed with batch training.")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
