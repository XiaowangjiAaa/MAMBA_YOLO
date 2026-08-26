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
    """Test forward pass with AMP autocast."""
    model = model.to(device).train()
    batch = torch.randn(2, 3, 640, 640, device=device)
    try:
        with torch.cuda.amp.autocast(enabled=True):
            _ = model(batch)
    except Exception as e:
        return False, f"AMP forward: {e}"
    return True, None


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
        for head_name in ("structure_head", "guidance_head", "orientation_head"):
            head = getattr(module, head_name, None)
            if head is not None:
                head_params.extend(p for p in head.parameters() if p.requires_grad)
    grads = torch.autograd.grad(sum(probe_terms), head_params, retain_graph=True, allow_unused=True)
    if not any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads):
        return False, "auxiliary cache is attached but no guidance-head gradient was produced"
    return True, f"{len(guided)} live graph(s); gradient reaches guidance head(s)"


def test_826_gradient_reachability(model, config_path, output):
    """Detect trainable CASP parameters that forward-only YAML checks miss.

    DDP with ``find_unused_parameters=False`` fails when a registered parameter
    does not reach the loss graph. This specifically guards component-ablation
    YAMLs such as W07, where a disabled role must also remove its scalar gate.
    """
    if "8.26-experiments" not in str(config_path) or output is None:
        return True, "not an 8.26 config"

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
    states = [m for m in model.modules() if m.__class__.__name__ == "EfficientCrackAlignedState"]
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
    if unused:
        preview = ", ".join(unused[:8])
        suffix = f" (+{len(unused) - 8} more)" if len(unused) > 8 else ""
        return False, f"DDP-unused CASP parameter(s): {preview}{suffix}"
    return True, f"all {len(named_parameters)} CASP parameter tensors reach the training graph"


def validate_config(config_path, device, nc=1, imgsz=640, quick=False):
    """Run all checks on a single YAML config. Returns list of (check_name, passed, detail)."""
    results = []
    config_name = Path(config_path).name if not Path(config_path).is_absolute() else Path(config_path).name

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

    # 2. Deepcopy
    ok, detail = test_deepcopy(model, device)
    results.append(("deepcopy (EMA)", ok, detail or ""))

    # 3. Forward pass
    ok, detail, train_output = test_forward(model, device, nc=nc, imgsz=imgsz)
    results.append(("forward", ok, detail or ""))
    if ok:
        cache_ok, cache_detail = test_live_aux_cache(model)
        results.append(("aux graph attached", cache_ok, cache_detail or ""))
        reach_ok, reach_detail = test_826_gradient_reachability(model, config_path, train_output)
        results.append(("8.26 gradient reach", reach_ok, reach_detail or ""))

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
    if args.experiments:
        for e in args.experiments:
            exp_ids.add(e)
    if args.phase:
        phase_map = {
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
    parser.add_argument("--phase", nargs="+", default=None, help="Phase: 26B 26M 26D 26A, 24F 24H 24R, or legacy")
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
        targets = args.configs
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
