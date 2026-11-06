"""批量预测所有权重 × 数据集组合，每张图一次推理输出多种配色。"""

from pathlib import Path

import visualize_seg_results as visualization

# 用户配置区：权重可为 .pt 或 .pth；原始 state_dict 的 .pth 需填写对应 model_yaml。
MODELS = {
    "mambayolo": {
        "weights": r"output_dir/mambayolo/mambayolo_crackseg/weights/best_map50.pt",
    },
    "ours": {
        "weights": r"output_dir\crack-seg-9.3\H01_crack-seg_seed0\weights\best_map50.pt",
    },
    "RF-DETR": {
        "weights": r"output_dir\crack-seg-9.3\H00_crack-seg_seed0\weights\best_map50.pt",
    },
}
DATASETS = {
    "SUT-Crack": {
        "image": r"H:\dataset\Public_crack_dataset\SUT-Crack\VOC2012\JPEGImages",
        "GT": r"H:\dataset\Public_crack_dataset\SUT-Crack\VOC2012\cnnlabels",
    },
}
OUTPUT_DIR = "predictresults/batch"
ERROR_COLORS = {"FP": "#D98282", "FN": "#D8AD63"}  # 误检：柔和珊瑚色；漏检：暖金色。
SAVE_MASK_OVERLAYS = True  # overlays 含框和标签；errors/errors_white 均无框、无标签。
SKIP_EXISTING = False  # 重画完成后可改为 True；检查全部配色和白底输出是否齐全。
BATCH_SIZE = 1
SPLIT = "all"  # YAML 可选 train、val、test、all。
COLOR_VARIANTS = {
    "lavender": {0: "#DEC6E7"},
    "teal": {0: "#08B5AD"},
    "blue_gray": {0: "#D1DFEA"},
    "mint": {0: "#D3E8DD"},
    "pink": {0: "#FFE2E5"},
    "periwinkle": {0: "#E6E5FA"},
    "sky_blue": {0: "#C9E8FD"},
    "seafoam": {0: "#73BDB5"},
    "pale_lavender": {0: "#EEEBFE"},
    "powder_blue": {0: "#C1E2F9"},
    "lilac": {0: "#E0CFF9"},
}
MASK_ALPHA = 0.5  # 掩膜不透明度：0 完全透明，1 完全不透明。
DEVICE = "0"  # 无 GPU 时改为 "cpu"。
IMGSZ = 640
VISUAL_CONF = 0.25
MAX_IMAGES = 0  # 0 预测全部图像；可先改为 1 检查输出。


def main():
    """复用原脚本的模型加载、数据集划分、预测和指标逻辑。"""
    visualization.COLOR_VARIANTS = COLOR_VARIANTS
    visualization.ERROR_COLORS = ERROR_COLORS
    visualization.SAVE_MASK_OVERLAYS = SAVE_MASK_OVERLAYS
    visualization.SKIP_EXISTING = SKIP_EXISTING
    visualization.SPLIT = SPLIT
    for model_name, model_config in MODELS.items():
        for dataset_name, source in DATASETS.items():
            config = (
                {key.lower(): value for key, value in source.items()} if isinstance(source, dict) else {"image": source}
            )
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
            print(f"\n模型：{model_name}；数据集：{dataset_name}")
            visualization.main(visualization.parse_args(argv))


if __name__ == "__main__":
    main()
