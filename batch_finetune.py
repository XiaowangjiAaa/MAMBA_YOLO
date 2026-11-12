"""从模型 YAML 构建分割模型，在同一数据集上批量从头训练，不使用预训练权重。

运行：python batch_finetune.py
预览：python batch_finetune.py --dry-run
覆盖数据集：python batch_finetune.py --data /path/to/data.yaml
相对模型配置、输出路径均以本脚本所在目录为基准。
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ==================== 用户填写区 ====================
# 与相邻 ultralytics/batch_train.py 使用同一个数据集。
DATA_YAML = r'crack-seg/crack-seg.yaml'
MODELS = {
    "mambayolo": {
        "config": "ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml",
    },
    "ours": {
        "config": "ultralytics/cfg/models/11/9.3-experiments/H01-g01-full.yaml",
    },
}
EPOCHS = 300
IMGSZ = 640
BATCH = 16
DEVICE = "0"
WORKERS = 8
PROJECT = "output_dir/Panoramic view"
SEEDS = [0]
OPTIMIZER = "auto"
PATIENCE = 100
CRACK_METRIC_CONF = 0.25
CLDICE_ITERS = 20
SKIP_COMPLETED = True
CONTINUE_ON_ERROR = True
# ==================================================


def absolute_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def prepare_dataset(source, output_dir):
    """生成绝对数据根目录的 YAML，避免跨项目运行时解析到错误的数据集。"""
    import yaml

    with source.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not data.get("train") or not data.get("val"):
        raise ValueError("数据集 YAML 必须包含 train 和 val。")
    if not data.get("names") and not data.get("nc"):
        raise ValueError("数据集 YAML 必须包含 names 或 nc。")
    root = Path(data.get("path") or source.parent).expanduser()
    if not root.is_absolute():
        # 同时兼容 path: . 和 path: yolo_instance_dataset 两种常见写法。
        candidates = [source.parent / root, source.parent.parent / root]
        matches = list(dict.fromkeys(p.resolve() for p in candidates if p.is_dir()))
        if len(matches) != 1:
            raise ValueError(f"无法唯一确定数据集根目录，请在 {source} 中填写绝对 path: {matches}")
        root = matches[0]
    if not root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在: {root}")
    data["path"] = str(root.resolve())
    for split in ("train", "val", "test"):
        entries = data.get(split)
        if not entries:
            continue
        for entry in entries if isinstance(entries, list) else [entries]:
            path = Path(entry)
            path = path if path.is_absolute() else root / path
            if not path.exists():
                raise FileNotFoundError(f"数据集 {split} 路径不存在: {path}")
    if data["train"] == data["val"]:
        print("[WARN] 数据集 train 和 val 指向同一位置；验证指标不能作为独立泛化评估。")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "dataset.resolved.yaml"
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
    return target


def train_one(config, data, project, name, seed):
    import torch
    from ultralytics import YOLO

    model = None
    try:
        # 只读取模型结构，随机初始化参数，不加载任何已有 checkpoint。
        model = YOLO(str(config), task="segment")
        if model.task != "segment":
            raise ValueError(f"期望分割模型，实际 task={model.task}: {config}")
        model.train(
            data=str(data),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            workers=WORKERS,
            project=str(project),
            name=name,
            optimizer=OPTIMIZER,
            patience=PATIENCE,
            seed=seed,
            crack_metric_conf=CRACK_METRIC_CONF,
            cldice_iters=CLDICE_ITERS,
            pretrained=False,
            resume=False,
            exist_ok=False,
        )
        # 只将正常返回且生成了权重的训练标为完成，避免首轮 best.pt 导致误跳过。
        save_dir = Path(model.trainer.save_dir)
        if not any((save_dir / "weights").glob("best*.pt")):
            raise RuntimeError(f"训练返回但未生成 best*.pt: {save_dir}")
        return save_dir
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_YAML), help="数据集 YAML 路径")
    parser.add_argument("--dry-run", action="store_true", help="只检查文件并打印计划，不加载模型")
    args = parser.parse_args(argv)
    data = absolute_path(args.data)
    models = {name: absolute_path(cfg["config"]) for name, cfg in MODELS.items()}
    if any(path.suffix.lower() not in {".yaml", ".yml"} for path in models.values()):
        raise ValueError("从头训练必须使用模型 YAML 配置，不能使用权重文件。")
    missing = [path for path in [data, *models.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下文件不存在:\n" + "\n".join(map(str, missing)))
    seeds = SEEDS if isinstance(SEEDS, (list, tuple)) else [SEEDS]
    if not seeds or not models:
        raise ValueError("MODELS 和 SEEDS 不能为空。")
    base_project = absolute_path(PROJECT)
    print(f"数据集: {data}")
    print(f"{len(models)} 个模型 × {len(seeds)} 个种子 = {len(models) * len(seeds)} 次从头训练")
    print(f"epochs={EPOCHS}, imgsz={IMGSZ}, batch={BATCH}, device={DEVICE}, optimizer={OPTIMIZER}")
    print("[INFO] 从 YAML 随机初始化模型，pretrained=False，resume=False。")
    passed = failed = skipped = 0
    for seed in seeds:
        project = base_project.with_name(f"{base_project.name}-seed{seed}")
        for name, config in models.items():
            marker = project / f".{name}.complete.json"
            if SKIP_COMPLETED and marker.is_file():
                saved = json.loads(marker.read_text(encoding="utf-8"))
                output = Path(saved["save_dir"])
                if (saved.get("config") == str(config) and saved.get("pretrained") is False
                        and saved.get("data") == str(data)
                        and any((output / "weights").glob("best*.pt"))):
                    print(f"[SKIP] {name}, seed={seed}: {output}")
                    skipped += 1
                    continue
            print(f"\n[PLAN] {name}, seed={seed}\n  config={config}\n  pretrained=False\n  output={project / name}")
            if args.dry_run:
                continue
            started = time.monotonic()
            try:
                resolved_data = prepare_dataset(data, project)
                output = train_one(config, resolved_data, project, name, seed)
                marker.write_text(json.dumps({
                    "config": str(config), "pretrained": False,
                    "data": str(data), "save_dir": str(output),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                passed += 1
                print(f"[OK] {name}, seed={seed}: {time.monotonic() - started:.0f}s")
            except Exception:
                failed += 1
                traceback.print_exc()
                if not CONTINUE_ON_ERROR:
                    raise
    if args.dry_run:
        print("\n预览结束，文件路径检查通过；未加载模型、写入文件或启动训练。")
    else:
        print(f"\n完成: {passed} 成功，{failed} 失败，{skipped} 跳过。")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
