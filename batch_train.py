import subprocess
import sys
import os
import argparse
import time
import json
import re
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "ultralytics" / "cfg" / "models" / "mamba-yolo"
EXPERIMENT_89_DIR = CONFIG_DIR / "8.9-experiments"
STATUS_FILE = ROOT / "batch_train_status.json"
LOG_DIR = ROOT / "batch_logs"

# ============================================================
# Experiment definitions derived from README_CRACK_STRUCTURE_EXPERIMENTS.md
# Order follows recommended training sequence.
# ============================================================

YOLO11_EXPERIMENTS = {
    # --- Phase 1: Baseline ---
    "B0": {
        "config": "yolo-mamba-seg-yolo11.yaml",
        "desc": "Baseline (YOLO11 head, no modifications)",
        "phase": 1,
    },
    # --- Phase 1: Scan ablations ---
    "S1": {
        "config": "yolo-mamba-orientation-p3-seg-yolo11.yaml",
        "desc": "P3 H/V orientation scan (no supervision)",
        "phase": 1,
    },
    "S2": {
        "config": "yolo-mamba-orientation-p3-sup-seg-yolo11.yaml",
        "desc": "P3 H/V orientation scan + direction supervision",
        "phase": 1,
    },
    # --- Phase 1: Memory (write gate) ablations ---
    "M1": {
        "config": "yolo-mamba-crack-write-p4-seg-yolo11.yaml",
        "desc": "P4 crack probability write gate (no supervision)",
        "phase": 1,
    },
    "M2": {
        "config": "yolo-mamba-crack-write-p4-sup-seg-yolo11.yaml",
        "desc": "P4 crack probability write gate + probability supervision",
        "phase": 1,
    },
    # --- Phase 2: Combinations ---
    "C1": {
        "config": "yolo-mamba-scan-p3-write-p4-seg-yolo11.yaml",
        "desc": "P3 scan + P4 write (no supervision)",
        "phase": 2,
    },
    "C2": {
        "config": "yolo-mamba-scan-p3-write-p4-sup-seg-yolo11.yaml",
        "desc": "P3 scan + P4 write + dual supervision (RECOMMENDED)",
        "phase": 2,
    },
    # --- Phase 3: Layer position & Delta checks ---
    "M3": {
        "config": "yolo-mamba-crack-memory-p4-sup-seg-yolo11.yaml",
        "desc": "P4 Delta + B write joint control",
        "phase": 3,
    },
    "S3": {
        "config": "yolo-mamba-orientation-p4-seg-yolo11.yaml",
        "desc": "P4 H/V scan (layer position control)",
        "phase": 3,
    },
    "S4": {
        "config": "yolo-mamba-orientation-p3p4-seg-yolo11.yaml",
        "desc": "P3+P4 H/V scan (multi-scale)",
        "phase": 3,
    },
    # --- Phase 4: Extended ---
    "M0": {
        "config": "yolo-mamba-crack-write-p3-seg-yolo11.yaml",
        "desc": "P3 write gate (layer position control for memory)",
        "phase": 4,
    },
    "M0S": {
        "config": "yolo-mamba-crack-write-p3-sup-seg-yolo11.yaml",
        "desc": "P3 write gate + supervision (layer position control)",
        "phase": 4,
    },
    "M4": {
        "config": "yolo-mamba-crack-write-p3p4-sup-seg-yolo11.yaml",
        "desc": "P3+P4 write gate multi-scale (supervised)",
        "phase": 4,
    },
    "C3": {
        "config": "yolo-mamba-scan-p3-memory-p4-sup-seg-yolo11.yaml",
        "desc": "P3 scan + P4 Delta+B (strong memory control)",
        "phase": 4,
    },
    "J1": {
        "config": "yolo-mamba-structure-p3p4-sup-seg-yolo11.yaml",
        "desc": "P3/P4 same-scale Scan+Write coupling control",
        "phase": 4,
    },
    "S5": {
        "config": "yolo-mamba-orientation-diagonal-p3-sup-seg-yolo11.yaml",
        "desc": "P3 8-direction scan (diagonal enhanced, high cost)",
        "phase": 4,
    },
}

# Cracks Delta-v2 position ablation experiments
DELTA_V2_EXPERIMENTS = {
    "DV2-P2": {
        "config": "yolo-mamba-crack-delta-v2-p2-seg-yolo11.yaml",
        "desc": "Crack Delta-v2 at P2/4",
        "phase": "dv2",
    },
    "DV2-P3": {
        "config": "yolo-mamba-crack-delta-v2-p3-seg-yolo11.yaml",
        "desc": "Crack Delta-v2 at P3/8",
        "phase": "dv2",
    },
    "DV2-P4": {
        "config": "yolo-mamba-crack-delta-v2-p4-seg-yolo11.yaml",
        "desc": "Crack Delta-v2 at P4/16",
        "phase": "dv2",
    },
    "DV2-ALL": {
        "config": "yolo-mamba-crack-delta-v2-seg-yolo11.yaml",
        "desc": "Crack Delta-v2 on all layers",
        "phase": "dv2",
    },
}

# YOLOv8-equivalent control experiments
YOLOV8_EXPERIMENTS = {
    "V8-S1": {
        "config": "yolo-mamba-orientation-p3-seg.yaml",
        "desc": "YOLOv8: P3 H/V orientation scan",
        "phase": "v8",
    },
    "V8-M1": {
        "config": "yolo-mamba-crack-write-p4-seg.yaml",
        "desc": "YOLOv8: P4 crack write gate",
        "phase": "v8",
    },
    "V8-C1": {
        "config": "yolo-mamba-scan-p3-write-p4-seg.yaml",
        "desc": "YOLOv8: P3 scan + P4 write (no sup)",
        "phase": "v8",
    },
    "V8-C2": {
        "config": "yolo-mamba-scan-p3-write-p4-sup-seg.yaml",
        "desc": "YOLOv8: P3 scan + P4 write + sup",
        "phase": "v8",
    },
    "V8-DV2": {
        "config": "yolo-mamba-crack-delta-v2-seg.yaml",
        "desc": "YOLOv8: Crack Delta-v2",
        "phase": "v8",
    },
    "V8-B0": {
        "config": "yolo-mamba-seg.yaml",
        "desc": "YOLOv8: baseline Mamba seg",
        "phase": "v8",
    },
}

# Mapping of experiment sets
ALL_EXPERIMENTS = {}
ALL_EXPERIMENTS.update(YOLO11_EXPERIMENTS)
ALL_EXPERIMENTS.update(DELTA_V2_EXPERIMENTS)
ALL_EXPERIMENTS.update(YOLOV8_EXPERIMENTS)

# 2026-08-09 evidence-driven experiments. These are the default batch set.
# Phases follow 8.9_CRACK_STRUCTURE_RESULTS_AND_NEXT_EXPERIMENTS.md.
AUG9_EXPERIMENTS = {
    "E00": {"config": "8.9-experiments/00-b0-yolo11.yaml", "desc": "Baseline seed replication", "phase": "A"},
    "E01": {"config": "8.9-experiments/01-s1-p3-scan-yolo11.yaml", "desc": "S1 P3 scan seed replication", "phase": "A"},
    "E02": {"config": "8.9-experiments/02-c2-reference-yolo11.yaml", "desc": "C2 seed replication", "phase": "A"},
    "E03": {"config": "8.9-experiments/03-c2-guidance-only-yolo11.yaml", "desc": "C2 with guidance supervision only", "phase": "B"},
    "E04": {"config": "8.9-experiments/04-c2-orientation-only-yolo11.yaml", "desc": "C2 with orientation supervision only", "phase": "B"},
    "E05": {"config": "8.9-experiments/05-c2-low-aux-yolo11.yaml", "desc": "C2 with lower dual auxiliary weights", "phase": "B"},
    "E06": {"config": "8.9-experiments/06-c1-no-aux-control-yolo11.yaml", "desc": "C1 full-300-epoch control", "phase": "B", "patience": 300},
    "E07": {"config": "8.9-experiments/07-centered-p4-write-sup-yolo11.yaml", "desc": "Centered write at all P4 blocks", "phase": "C"},
    "E08": {"config": "8.9-experiments/08-p3-scan-centered-p4-write-sup-yolo11.yaml", "desc": "P3 scan + centered write at all P4 blocks", "phase": "C"},
    "E09": {"config": "8.9-experiments/09-p3-scan-last-p4-write-sup-yolo11.yaml", "desc": "P3 scan + old write at last P4 block", "phase": "C"},
    "E10": {"config": "8.9-experiments/10-p3-scan-last-p4-centered-write-sup-yolo11.yaml", "desc": "P3 scan + centered write at last P4 block (main candidate)", "phase": "C"},
}
ALL_EXPERIMENTS.update(AUG9_EXPERIMENTS)

# 2026-08-12: crack-specific front end and unified scan/write state update.
# 12S and 12M isolate the new front-end operators; 12U evaluates the unified VSS design.
AUG12_EXPERIMENTS = {
    "N00": {"config": "8.12-experiments/00-original-mamba-yolo11.yaml", "desc": "8.12 untouched Mamba-YOLO reference", "phase": "12S"},
    "N01": {"config": "8.12-experiments/01-crack-stem-lite.yaml", "desc": "Crack Detail Preserving Stem Lite only", "phase": "12S"},
    "N02": {"config": "8.12-experiments/02-crack-stem-directional.yaml", "desc": "Directional Crack Detail Stem only", "phase": "12S"},
    "N03": {"config": "8.12-experiments/03-crack-merge-lite.yaml", "desc": "Crack Merge Lite only", "phase": "12M"},
    "N04": {"config": "8.12-experiments/04-crack-merge-directional.yaml", "desc": "Directional Crack Merge only", "phase": "12M"},
    "N05": {"config": "8.12-experiments/05-crack-front-lite.yaml", "desc": "Stem Lite + Merge Lite, standard VSS", "phase": "12M"},
    "N06": {"config": "8.12-experiments/06-unified-backbone.yaml", "desc": "Unified Crack-Aware VSS in backbone only", "phase": "12U"},
    "N07": {"config": "8.12-experiments/07-unified-all.yaml", "desc": "Unified Crack-Aware VSS in backbone and neck", "phase": "12U"},
    "N08": {"config": "8.12-experiments/08-full-lite-backbone.yaml", "desc": "Lite front end + unified backbone", "phase": "12U"},
    "N09": {"config": "8.12-experiments/09-full-lite-all.yaml", "desc": "Lite front end + unified backbone and neck", "phase": "12U"},
    "N10": {"config": "8.12-experiments/10-full-directional-backbone.yaml", "desc": "Directional front end + unified backbone", "phase": "12U"},
    "N11": {"config": "8.12-experiments/11-full-directional-all.yaml", "desc": "Directional front end + unified backbone and neck", "phase": "12U"},
}
ALL_EXPERIMENTS.update(AUG12_EXPERIMENTS)

# 2026-08-17: fixed architecture, parameter tuning only.
AUG17_EXPERIMENTS = {
    "T00": {"config": "8.17-tuning/T00-fixed-default.yaml", "desc": "Fixed method, default auxiliary losses and gates", "phase": "17A"},
    "T01": {"config": "8.17-tuning/T01-no-aux.yaml", "desc": "Fixed method, no guidance auxiliary supervision", "phase": "17A"},
    "T02": {"config": "8.17-tuning/T02-strong-aux.yaml", "desc": "Fixed method, stronger auxiliary supervision", "phase": "17A"},
    "T03": {"config": "8.17-tuning/T03-mild-gates.yaml", "desc": "Fixed method, milder write/scan gate initialization", "phase": "17G"},
    "T04": {"config": "8.17-tuning/T04-strong-gates.yaml", "desc": "Fixed method, stronger write/scan gate initialization", "phase": "17G"},
    "T05": {"config": "8.17-tuning/T05-temperature-1.yaml", "desc": "Fixed method, orientation temperature 1.0", "phase": "17G"},
}
ALL_EXPERIMENTS.update(AUG17_EXPERIMENTS)

# 2026-08-19: one mandatory fair control followed by two theory-consistent
# optimizations of the same Unified block. This is not a new architecture search.
AUG19_EXPERIMENTS = {
    "R00": {
        "config": "8.19-experiments/R00-fair-control.yaml",
        "desc": "Fair control: P3/P4 directional merge + original P5 + Standard VSS only",
        "phase": "19C",
    },
    "R01": {
        "config": "8.19-experiments/R01-nonnegative-noaux.yaml",
        "desc": "Unified block with nonnegative gates, no auxiliary supervision",
        "phase": "19G",
    },
    "R02": {
        "config": "8.19-experiments/R02-nonnegative-gatereg.yaml",
        "desc": "R01 plus weak gate-magnitude regularization",
        "phase": "19G",
    },
}
ALL_EXPERIMENTS.update(AUG19_EXPERIMENTS)

# 2026-08-23: restore the supported P5 directional merge, then perform
# write/scan component attribution without replacing the Unified block.
AUG23_EXPERIMENTS = {
    "Q00": {
        "config": "8.23-experiments/Q00-p5-directional-full.yaml",
        "desc": "Final candidate: P3/P4/P5 directional merge + full nonnegative Unified",
        "phase": "23F",
    },
    "Q01": {
        "config": "8.23-experiments/Q01-write-only.yaml",
        "desc": "Unified component ablation: nonnegative crack write only",
        "phase": "23A",
    },
    "Q02": {
        "config": "8.23-experiments/Q02-scan-only.yaml",
        "desc": "Unified component ablation: nonnegative orientation scan only",
        "phase": "23A",
    },
    "Q03": {
        "config": "8.23-experiments/Q03-role-specific.yaml",
        "desc": "Role-specific Unified: P3 write only + P4-last scan only",
        "phase": "23A",
    },
}
ALL_EXPERIMENTS.update(AUG23_EXPERIMENTS)

# 2026-08-24: correct Q00's H/V family-logit mapping and test only bounded
# stability controls. U00 is the exact causal correction; U01/U02/U04 tune one
# parameter family at a time; U03 tests the learned P3-write/P4-scan roles.
AUG24_EXPERIMENTS = {
    "U00": {
        "config": "8.24-experiments/U00-corrected-hv-full.yaml",
        "desc": "Corrected Q00: both structure channels are H/V family logits",
        "phase": "24F",
    },
    "U01": {
        "config": "8.24-experiments/U01-corrected-hv-temp1.yaml",
        "desc": "Corrected Q00 with softer orientation temperature 1.0",
        "phase": "24H",
    },
    "U02": {
        "config": "8.24-experiments/U02-corrected-hv-scanmax015.yaml",
        "desc": "Corrected Q00 with scan strength capped at 0.15",
        "phase": "24H",
    },
    "U03": {
        "config": "8.24-experiments/U03-corrected-hv-role-specific.yaml",
        "desc": "Corrected role split: P3 write-only and P4-last scan-only",
        "phase": "24R",
    },
    "U04": {
        "config": "8.24-experiments/U04-corrected-hv-learned-init.yaml",
        "desc": "Corrected Q00 initialized near previously learned gate roles",
        "phase": "24H",
    },
}
ALL_EXPERIMENTS.update(AUG24_EXPERIMENTS)

# 2026-08-26: true YOLO11-Seg topology and the efficient, edge-coupled CASP block.
AUG26_EXPERIMENTS = {
    "W00": {"config": "../11/8.26-experiments/W00-yolo11-seg-baseline.yaml", "desc": "True YOLO11-Seg C3k2 baseline", "phase": "26B"},
    "W01": {"config": "../11/8.26-experiments/W01-casp-p3p4.yaml", "desc": "Primary efficient CASP at backbone P3/P4", "phase": "26M"},
    "W02": {"config": "../11/8.26-experiments/W02-casp-backbone-all.yaml", "desc": "Adaptive CASP at every backbone C3k2", "phase": "26D"},
    "W03": {"config": "../11/8.26-experiments/W03-casp-all-c3k2.yaml", "desc": "Universality: replace every backbone/neck C3k2", "phase": "26D"},
    "W04": {"config": "../11/8.26-experiments/W04-casp-no-transition.yaml", "desc": "Ablation: no edge-conditioned memory decay", "phase": "26A"},
    "W05": {"config": "../11/8.26-experiments/W05-casp-no-write.yaml", "desc": "Ablation: no edge-conditioned state writing", "phase": "26A"},
    "W06": {"config": "../11/8.26-experiments/W06-casp-p3p4-ratio0125.yaml", "desc": "Efficiency: P3/P4 with 0.125 state ratio", "phase": "26M"},
    "W07": {"config": "../11/8.26-experiments/W07-casp-no-fusion.yaml", "desc": "Ablation: uniform merge while edge still controls memory", "phase": "26A"},
}
ALL_EXPERIMENTS.update(AUG26_EXPERIMENTS)

# 2026-08-27: fixed full CASP; performance-oriented parameter optimization.
AUG27_EXPERIMENTS = {
    "X00": {"config": "../11/8.27-experiments/X00-yolo11-seg-map50-baseline.yaml", "desc": "YOLO11 baseline with Mask-mAP50 checkpoint selection", "phase": "27B"},
    "X01": {"config": "../11/8.27-experiments/X01-casp-reference.yaml", "desc": "Full learnable CASP reference", "phase": "27F"},
    "X02": {"config": "../11/8.27-experiments/X02-guidance001.yaml", "desc": "Probability guidance weight 0.01", "phase": "27G"},
    "X03": {"config": "../11/8.27-experiments/X03-guidance005.yaml", "desc": "Probability guidance weight 0.05", "phase": "27G"},
    "X04": {"config": "../11/8.27-experiments/X04-guidance010.yaml", "desc": "Probability guidance weight 0.10", "phase": "27G"},
    "X05": {"config": "../11/8.27-experiments/X05-orientation0005.yaml", "desc": "H/V family supervision weight 0.005", "phase": "27O"},
    "X06": {"config": "../11/8.27-experiments/X06-orientation001.yaml", "desc": "H/V family supervision weight 0.01", "phase": "27O"},
    "X07": {"config": "../11/8.27-experiments/X07-route002.yaml", "desc": "Residual route init 0.02", "phase": "27R"},
    "X08": {"config": "../11/8.27-experiments/X08-route010.yaml", "desc": "Residual route init 0.10", "phase": "27R"},
    "X09": {"config": "../11/8.27-experiments/X09-direction-mix025.yaml", "desc": "Probability-dominant direction mix 0.25", "phase": "27D"},
    "X10": {"config": "../11/8.27-experiments/X10-direction-mix075.yaml", "desc": "Direction mix 0.75", "phase": "27D"},
    "X11": {"config": "../11/8.27-experiments/X11-ratio0375.yaml", "desc": "State ratio 0.375", "phase": "27C"},
    "X12": {"config": "../11/8.27-experiments/X12-ratio050.yaml", "desc": "State ratio 0.50", "phase": "27C"},
    "X13": {"config": "../11/8.27-experiments/X13-ratio0375-dstate16.yaml", "desc": "State ratio 0.375 with d_state 16", "phase": "27C"},
    "X14": {"config": "../11/8.27-experiments/X14-stage-specific-p3p4.yaml", "desc": "Stage-specific P3 detail and P4 semantic-memory strengths", "phase": "27C"},
}
ALL_EXPERIMENTS.update(AUG27_EXPERIMENTS)

# 2026-08-28: final sparse adaptive crack-path Mamba; full-module parameter tuning only.
AUG28_EXPERIMENTS = {
    "Y00": {"config": "../11/8.28-experiments/Y00-crack-path-reference.yaml", "desc": "Full sparse crack-path reference", "phase": "28F"},
    "Y01": {"config": "../11/8.28-experiments/Y01-seed-ratio001.yaml", "desc": "Sparse 1% path seeds", "phase": "28S"},
    "Y02": {"config": "../11/8.28-experiments/Y02-seed-ratio004.yaml", "desc": "Dense 4% path seeds", "phase": "28S"},
    "Y03": {"config": "../11/8.28-experiments/Y03-path-steps3.yaml", "desc": "Short local crack paths", "phase": "28S"},
    "Y04": {"config": "../11/8.28-experiments/Y04-path-steps6.yaml", "desc": "Longer curved crack paths", "phase": "28S"},
    "Y05": {"config": "../11/8.28-experiments/Y05-path-conf002.yaml", "desc": "Permissive path continuation", "phase": "28S"},
    "Y06": {"config": "../11/8.28-experiments/Y06-path-conf010.yaml", "desc": "Conservative path continuation", "phase": "28S"},
    "Y07": {"config": "../11/8.28-experiments/Y07-connectivity001.yaml", "desc": "Connectivity supervision 0.01", "phase": "28L"},
    "Y08": {"config": "../11/8.28-experiments/Y08-connectivity005.yaml", "desc": "Connectivity supervision 0.05", "phase": "28L"},
    "Y09": {"config": "../11/8.28-experiments/Y09-orientation000.yaml", "desc": "No explicit tangent supervision", "phase": "28L"},
    "Y10": {"config": "../11/8.28-experiments/Y10-orientation001.yaml", "desc": "Tangent supervision 0.01", "phase": "28L"},
    "Y11": {"config": "../11/8.28-experiments/Y11-route005.yaml", "desc": "Stronger sparse residual route", "phase": "28M"},
    "Y12": {"config": "../11/8.28-experiments/Y12-memory010.yaml", "desc": "Stronger probability retention", "phase": "28M"},
    "Y13": {"config": "../11/8.28-experiments/Y13-transition010.yaml", "desc": "Stronger path-edge transition", "phase": "28M"},
    "Y14": {"config": "../11/8.28-experiments/Y14-write010.yaml", "desc": "Stronger crack-token writing", "phase": "28M"},
    "Y15": {"config": "../11/8.28-experiments/Y15-dstate16.yaml", "desc": "Larger path-state capacity", "phase": "28M"},
}
ALL_EXPERIMENTS.update(AUG28_EXPERIMENTS)


def load_status():
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_status(status):
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)


def format_duration(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    mins, secs = divmod(seconds, 60)
    if mins < 60:
        return f"{mins}m {secs:02d}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins:02d}m"


def parse_epoch_progress(line, total_epochs):
    m = re.search(r"(\d+)/(\d+)\s+\d+(?:\.\d+)?[GMK]", line)
    if not m:
        return None
    cur = int(m.group(1))
    tot = int(m.group(2))
    if tot == total_epochs:
        return cur
    return None


def build_cmd(exp_name, exp_info, args):
    config_path = CONFIG_DIR / exp_info["config"]
    if not config_path.exists():
        print(f"  [WARN] Config not found: {config_path}")
        return None

    if exp_name in AUG28_EXPERIMENTS:
        default_project = f"./output_dir/{args.data_stem}-8.28"
    elif exp_name in AUG27_EXPERIMENTS:
        default_project = f"./output_dir/{args.data_stem}-8.27"
    elif exp_name in AUG26_EXPERIMENTS:
        default_project = f"./output_dir/{args.data_stem}-8.26"
    elif exp_name in AUG24_EXPERIMENTS:
        default_project = f"./output_dir/{args.data_stem}-8.24"
    elif exp_name in AUG23_EXPERIMENTS:
        default_project = f"./output_dir/{args.data_stem}-8.23"
    elif exp_name in AUG19_EXPERIMENTS:
        default_project = f"./output_dir/{args.data_stem}-8.19"
    elif exp_name in AUG17_EXPERIMENTS:
        default_project = f"./output_dir/{args.data_stem}-8.17"
    elif exp_name in AUG12_EXPERIMENTS:
        default_project = f"./output_dir/{args.data_stem}-8.12"
    elif exp_name in AUG9_EXPERIMENTS:
        default_project = f"./output_dir/{args.data_stem}-8.9"
    else:
        default_project = f"./output_dir/{args.data_stem}"
    project_dir = args.project if args.project else default_project
    name = f"{exp_name}_{args.data_stem}_seed{args.seed}"
    patience = max(args.patience, exp_info.get("patience", 0))

    cmd_parts = [
        sys.executable, str(ROOT / "mbyolo_train.py"),
        "--task", "train",
        "--data", args.data,
        "--config", str(config_path),
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--patience", str(patience),
        "--seed", str(args.seed),
        "--save_period", str(args.save_period),
        "--crack_metric_conf", str(args.crack_metric_conf),
        "--cldice_iters", str(args.cldice_iters),
        "--batch_size", str(args.batch_size),
        "--workers", str(args.workers),
        "--optimizer", args.optimizer,
        "--project", project_dir,
        "--name", name,
    ]
    if args.amp:
        cmd_parts.append("--amp")
    if args.half:
        cmd_parts.append("--half")
    if args.weights:
        cmd_parts += ["--weights", args.weights]
    if args.imgsz:
        cmd_parts += ["--imgsz", str(args.imgsz)]

    return cmd_parts


def run_experiments(experiment_ids, args):
    status = load_status()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total = len(experiment_ids)
    passed = 0
    failed = 0
    skipped = 0
    exp_durations = []
    batch_start = time.time()

    print("=" * 70)
    print(f"  Batch Training: {total} experiment(s)")
    print(f"  Data:   {args.data}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Seed:   {args.seed}")
    print(f"  Save period: {args.save_period}")
    print(f"  Batch:  {args.batch_size}")
    print(f"  AMP:    {args.amp}")
    print(f"  Console: {'verbose' if args.verbose else 'concise (ETA only)'}")
    if args.skip_completed:
        print(f"  Mode:   skip completed experiments")
    print("=" * 70)

    for idx, exp_id in enumerate(experiment_ids, 1):
        if exp_id not in ALL_EXPERIMENTS:
            print(f"\n[{idx}/{total}] {exp_id} - UNKNOWN, skipping.")
            skipped += 1
            continue

        exp_info = ALL_EXPERIMENTS[exp_id]
        config = exp_info["config"]
        desc = exp_info["desc"]

        # Check if already completed
        run_id = f"{exp_id}:seed{args.seed}"
        if args.skip_completed and status.get(run_id) == "completed":
            print(f"\n[{idx}/{total}] {exp_id} ({config}) - already COMPLETED, skipping.")
            skipped += 1
            continue

        print(f"\n[{idx}/{total}] {exp_id} ({config})")
        print(f"  Description: {desc}")

        cmd = build_cmd(exp_id, exp_info, args)
        if cmd is None:
            status[run_id] = "failed"
            failed += 1
            save_status(status)
            continue

        log_file = LOG_DIR / f"{exp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        print(f"  Log: {log_file}")
        print(f"  CMD: {' '.join(cmd)}")
        print("-" * 70)

        start_time = time.time()

        try:
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(f"Experiment: {exp_id}\n")
                lf.write(f"Config: {config}\n")
                lf.write(f"Command: {' '.join(cmd)}\n")
                lf.write(f"Start: {datetime.now().isoformat()}\n")
                lf.write("-" * 70 + "\n")
                lf.flush()

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(ROOT),
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )

                last_epoch = 0
                for line in process.stdout:
                    lf.write(line)
                    lf.flush()

                    if args.verbose:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    else:
                        cur_epoch = parse_epoch_progress(line, args.epochs)
                        if cur_epoch is not None and cur_epoch != last_epoch:
                            last_epoch = cur_epoch
                            elapsed = time.time() - start_time
                            remaining = elapsed * (args.epochs - cur_epoch) / max(cur_epoch, 1)
                            sys.stdout.write(
                                f"\r  [{exp_id}] epoch {cur_epoch}/{args.epochs} | "
                                f"elapsed {format_duration(elapsed)} | "
                                f"ETA {format_duration(remaining)}"
                            )
                            sys.stdout.flush()

                process.wait()

                elapsed = time.time() - start_time
                elapsed_str = format_duration(elapsed)
                exp_durations.append(elapsed)

                if not args.verbose:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                if process.returncode == 0:
                    status[run_id] = "completed"
                    passed += 1
                    print(f"  [OK] {exp_id} completed in {elapsed_str}")
                    lf.write(f"\n[OK] Completed in {elapsed_str}\n")
                else:
                    status[run_id] = "failed"
                    failed += 1
                    print(f"  [FAIL] {exp_id} failed with code {process.returncode} ({elapsed_str})")
                    lf.write(f"\n[FAIL] Exit code: {process.returncode} ({elapsed_str})\n")
                    if not args.continue_on_error:
                        save_status(status)
                        print("\n  Stopping batch (use --continue-on-error to keep going).")
                        break

        except KeyboardInterrupt:
            print(f"\n  [ABORT] Batch interrupted during {exp_id}.")
            status[run_id] = "interrupted"
            save_status(status)
            sys.exit(1)

        save_status(status)

        # Batch-level progress / ETA estimate
        remaining_exps = total - idx
        if exp_durations and remaining_exps > 0:
            avg = sum(exp_durations) / len(exp_durations)
            batch_elapsed = time.time() - batch_start
            batch_eta = avg * remaining_exps
            print(f"  [BATCH] {idx}/{total} done | elapsed {format_duration(batch_elapsed)} "
                  f"| ~{format_duration(batch_eta)} remaining")

    print("\n" + "=" * 70)
    print(f"  SUMMARY: {passed} passed, {failed} failed, {skipped} skipped out of {total}")
    print(f"  Status saved to: {STATUS_FILE}")
    print("=" * 70)


def list_experiments(args):
    group_filter = args.list_group
    print("=" * 70)
    for exp_id, info in ALL_EXPERIMENTS.items():
        phase = info["phase"]
        if group_filter and str(phase) != group_filter:
            continue
        print(f"  {exp_id:<8s} | phase={phase!s:<4s} | {info['config']}")
        print(f"           | {info['desc']}")
        print()
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch training for Mamba-YOLO crack/structure experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # List all available experiments
  python batch_train.py --list

  # Reproduce B0/S1/C2 with two additional seeds
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase A --seeds 1 2

  # Split the C2 auxiliary supervision
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase B --seeds 0

  # Test centered and last-P4 memory designs
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase C --seeds 0

  # Tune the fixed 8.17 architecture (auxiliary loss first, then gates)
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 17A 17G --seeds 0

  # 8.19: run the mandatory control first, then the two gated variants
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 19C --seeds 0
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 19G --seeds 0

  # 8.23: final candidate first; run component ablations only after it passes
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 23F --seeds 0
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 23A --seeds 0

  # 8.26: true YOLO11 baseline, primary method and efficiency control
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 26B 26M --seeds 0

  # 8.27: establish the full-CASP reference, then tune one parameter family at a time
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 27B 27F --seeds 0
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 27G 27O 27R 27D 27C --seeds 0

  # 8.28: final dynamic crack-path module; tune path geometry, structure learning and memory
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 28F --seeds 0
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 28S 28L 28M --seeds 0

  # 8.24: corrected Q00 first, then seed-0 stability controls
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 24F --seeds 0 1 2
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase 24H 24R --seeds 0

  # Train delta-v2 experiments
  python batch_train.py --data ../crack-seg/crack-seg.yaml --phase dv2

  # Train with resume/skip support
  python batch_train.py --data ../crack-seg/crack-seg.yaml --experiments B0 S1 S2 --skip-completed --continue-on-error
        """,
    )

    # Experiment selection
    exp_group = parser.add_argument_group("Experiment selection")
    exp_group.add_argument("--experiments", nargs="+", default=None,
                           help="Specific experiment IDs to run (e.g. B0 S1 S2)")
    exp_group.add_argument("--phase", nargs="+", default=None,
                           help="Run phases 28F/28S/28L/28M, 27B/27F/27G/27O/27R/27D/27C, or legacy")
    exp_group.add_argument("--exclude", nargs="+", default=None,
                           help="Experiment IDs to exclude")

    # Listing
    parser.add_argument("--list", action="store_true",
                        help="List all available experiments and exit")
    parser.add_argument("--list-group", default=None,
                        help="Filter list by phase (28F, 28S, 28L, 28M, 27B, 27F, etc.)")

    # Data & training params
    train_group = parser.add_argument_group("Training configuration")
    train_group.add_argument("--data", type=str, required=False,
                             help="Dataset YAML path (required for training)")
    train_group.add_argument("--device", type=str, default="0,1",
                             help="CUDA device(s)")
    train_group.add_argument("--epochs", type=int, default=300,
                             help="Number of epochs per experiment")
    train_group.add_argument("--patience", type=int, default=100,
                             help="Early-stopping patience (E06 is forced to at least 300)")
    train_group.add_argument("--seeds", nargs="+", type=int, default=[0],
                             help="One or more seeds; experiments run once per seed")
    train_group.add_argument("--save-period", type=int, default=10,
                             help="Save a checkpoint every N epochs for mask-centric post-selection")
    train_group.add_argument("--batch-size", type=int, default=16,
                             help="Batch size (default 16, matching completed 8.19 runs)")
    train_group.add_argument("--workers", type=int, default=128,
                             help="DataLoader workers")
    train_group.add_argument("--optimizer", type=str, default="SGD",
                             help="Optimizer (SGD, Adam, AdamW)")
    train_group.add_argument("--amp", action="store_true", default=True,
                             help="Enable AMP (default: on)")
    train_group.add_argument("--no-amp", dest="amp", action="store_false",
                             help="Disable AMP")
    train_group.add_argument("--half", action="store_true",
                             help="Use FP16 half precision")
    train_group.add_argument("--weights", type=str, default="",
                             help="Pretrained weights path")
    train_group.add_argument("--imgsz", type=int, default=640,
                             help="Input image size")
    train_group.add_argument("--crack-metric-conf", type=float, default=0.25,
                             help="Confidence threshold for validation union-mask mIoU/clDice")
    train_group.add_argument("--cldice-iters", type=int, default=20,
                             help="Morphological skeleton iterations used by validation clDice")
    train_group.add_argument("--project", type=str, default="",
                             help="Output project directory (default: ./output_dir/<dataset>)")

    # Run control
    ctrl_group = parser.add_argument_group("Run control")
    ctrl_group.add_argument("--skip-completed", action="store_true",
                            help="Skip experiments marked as completed in status file")
    ctrl_group.add_argument("--continue-on-error", action="store_true",
                            help="Continue to next experiment even if current one fails")
    ctrl_group.add_argument("--dry-run", action="store_true",
                            help="Print commands without executing")
    ctrl_group.add_argument("--verbose", action="store_true",
                            help="Stream full training output to console (default: concise with ETA)")

    return parser.parse_args()


def resolve_experiments(args):
    exp_ids = set()
    use_all = False

    if args.experiments:
        for e in args.experiments:
            exp_ids.add(e)
    if args.phase:
        for p in args.phase:
            for eid, info in ALL_EXPERIMENTS.items():
                if str(info["phase"]) == p:
                    exp_ids.add(eid)
    if not exp_ids and not args.list:
        # Default: run the current 8.28 full crack-path parameter-search set.
        use_all = True
        exp_ids = set(AUG28_EXPERIMENTS.keys())

    if args.exclude:
        for e in args.exclude:
            exp_ids.discard(e)

    # Sort by phase then by name for a sensible order
    phase_order = {"28F": 0, "28S": 1, "28L": 2, "28M": 3,
                   "27B": 4, "27F": 5, "27G": 6, "27O": 7, "27R": 8, "27D": 9, "27C": 10,
                   "26B": 11, "26M": 12, "26D": 13, "26A": 14,
                   "24F": 11, "24H": 12, "24R": 13, "23F": 14, "23A": 15,
                   "19C": 16, "19G": 17, "17A": 18, "17G": 19, "12S": 20,
                   "12M": 21, "12U": 22, "A": 23, "B": 24, "C": 25,
                   "1": 26, "2": 27, "3": 28, "4": 29, "dv2": 30, "v8": 31}
    sorted_ids = sorted(
        exp_ids,
        key=lambda x: (phase_order.get(str(ALL_EXPERIMENTS[x]["phase"]), 99), x),
    )

    return sorted_ids


def main():
    args = parse_args()

    if args.list or args.list_group:
        list_experiments(args)
        return

    if not args.data:
        print("[ERROR] --data is required for training. Use --list to see experiments.")
        sys.exit(1)

    args.data_stem = Path(args.data).stem

    experiment_ids = resolve_experiments(args)

    if not experiment_ids:
        print("[ERROR] No experiments selected.")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN ===")
        for seed in args.seeds:
            args.seed = seed
            for exp_id in experiment_ids:
                if exp_id not in ALL_EXPERIMENTS:
                    print(f"  {exp_id}: UNKNOWN")
                    continue
                exp_info = ALL_EXPERIMENTS[exp_id]
                cmd = build_cmd(exp_id, exp_info, args)
                if cmd:
                    print(f"  {exp_id}/seed{seed}: {' '.join(cmd)}")
                else:
                    print(f"  {exp_id}/seed{seed}: CONFIG NOT FOUND")
        return

    for seed in args.seeds:
        args.seed = seed
        run_experiments(experiment_ids, args)


if __name__ == "__main__":
    main()
