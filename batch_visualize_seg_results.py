"""批量预测所有权重 × 数据集组合，每张图一次推理输出多种配色。"""

import time
from pathlib import Path

import visualize_seg_results as visualization

# 用户配置区：权重可为 .pt 或 .pth；原始 state_dict 的 .pth 需填写对应 model_yaml。
MODELS = {
    "mambayolo": {
            "weights": r"output_dir\mambayolo\mambayolo_crackseg\weights\best_map50.pt",
        },
    "ours": {
        "weights": r"output_dir/crack-seg-9.3/H01_crack-seg_seed0/weights/best_map50.pt",
    },
    "RF-DETR": {
        "weights": r"output_dir/crack-seg-9.3/H00_crack-seg_seed0/weights/best_map50.pt",
    },
}
DATASETS = {

    "SUT-Crack-4": {
            "image": r"C:\Users\xiaow\Desktop\CUT-CRACK\image",
            "GT": r"C:\Users\xiaow\Desktop\CUT-CRACK\gt",
        },





}
OUTPUT_DIR = "predictresults/batch"
ERROR_COLORS = {"FP": "#D98282", "FN": "#D8AD63"}  # 误检：柔和珊瑚色；漏检：暖金色。
SAVE_MASK_OVERLAYS = True  # overlays 含框和标签；errors/errors_white 均无框、无标签。
SKIP_EXISTING = True  # 重画完成后可改为 True；检查全部配色和白底输出是否齐全。
BATCH_SIZE = 1
SPLIT = "all"  # YAML 可选 train、val、test、all。
COLOR_VARIANTS = {
    "cyan": {0: "#02F2FC"},
    "orange": {0: "#E97132"},
    "lime_green": {0: "#55D944"},
    "teal": {0: "#08B5AD"},
}
MASK_ALPHA = 0.5  # 掩膜不透明度：0 完全透明，1 完全不透明。
DEVICE = "0"  # 无 GPU 时改为 "cpu"。
IMGSZ = 640
VISUAL_CONF = 0.25
MAX_IMAGES = 0  # 0 预测全部图像；可先改为 1 检查输出。


def main():
    """复用原脚本的模型加载、数据集划分、预测和指标逻辑。"""
    visualization.PROGRESS.update(total=0, done=0, processed=0, skipped=0, started=time.perf_counter())
    visualization.COLOR_VARIANTS = COLOR_VARIANTS
    visualization.ERROR_COLORS = ERROR_COLORS
    visualization.SAVE_MASK_OVERLAYS = SAVE_MASK_OVERLAYS
    visualization.SKIP_EXISTING = SKIP_EXISTING
    visualization.SPLIT = SPLIT
    datasets = []
    total_images = 0
    for dataset_name, source in DATASETS.items():
        config = (
            {key.lower(): value for key, value in source.items()} if isinstance(source, dict) else {"image": source}
        )
        source_path = Path(config["image"]).expanduser()
        visualization.log_progress(f"预扫描数据集 {dataset_name}：{source_path}")
        is_yaml = visualization.is_dataset_yaml(source_path)
        splits = visualization.available_dataset_splits(source_path) if is_yaml else [None]
        if is_yaml and SPLIT != "all":
            splits = [split for split in splits if split == SPLIT]
        image_count = 0
        for split in splits:
            images, _, _ = visualization.resolve_source(source_path, split or "val", True)
            image_count += min(len(images), MAX_IMAGES) if MAX_IMAGES > 0 else len(images)
        datasets.append((dataset_name, config))
        total_images += image_count * len(MODELS)
        split_names = ", ".join(split or "文件夹" for split in splits)
        visualization.log_progress(f"数据集 {dataset_name}：每个模型处理 {image_count} 张，划分={split_names}")
    visualization.PROGRESS["total"] = total_images
    total_jobs = len(MODELS) * len(datasets)
    visualization.log_progress(
        f"计划：{len(MODELS)} 个模型 × {len(datasets)} 个数据集，共 {total_jobs} 组、{total_images} 个图片任务"
    )
    job_index = 0
    for model_name, model_config in MODELS.items():
        for dataset_name, config in datasets:
            job_index += 1
            job_started = time.perf_counter()
            visualization.GT_DIR = config.get("gt")
            source = config["image"]
            argv = [
                "--weights",
                model_config["weights"],
                "--source",
                source,
                "--output-dir",
                str(Path(OUTPUT_DIR) / dataset_name / model_name),
                "--alpha",
                str(MASK_ALPHA),
                "--device",
                DEVICE,
                "--imgsz",
                str(IMGSZ),
                "--conf",
                str(VISUAL_CONF),
                "--visual-conf",
                str(VISUAL_CONF),
                "--max-images",
                str(MAX_IMAGES),
                "--batch",
                str(BATCH_SIZE),
                "--recursive",
            ]
            if model_config.get("model_yaml"):
                argv.extend(["--model-yaml", model_config["model_yaml"]])
            visualization.log_progress(f"开始任务 {job_index}/{total_jobs}：模型={model_name}；数据集={dataset_name}")
            visualization.main(visualization.parse_args(argv))
            visualization.log_progress(
                f"任务 {job_index}/{total_jobs} 结束，耗时 {time.perf_counter() - job_started:.2f}s"
            )
    progress = visualization.PROGRESS
    visualization.log_progress(
        f"运行结束：新处理 {progress['processed']}，跳过 {progress['skipped']}，"
        f"未完成 {progress['total'] - progress['done']}；输出目录={OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
