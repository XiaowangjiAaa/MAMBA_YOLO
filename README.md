# MAMBA_YOLO

Crack instance-segmentation research fork based on **Mamba-YOLO** and a YOLO11 segmentation head. The current research line combines directional detail-preserving stage transitions with crack-aware state writing and adaptive horizontal/vertical scan fusion.

- Experiment history and design notes: [docs/experiments](docs/experiments/README.md)
- Reproducible training commands: [docs/commands](docs/commands/README.md)
- Current corrected Q00 experiments: [8.24 design](docs/experiments/8.24_Q00_CORRECTED_HV_AND_STABILITY_EXPERIMENTS.md)

Training outputs and checkpoints under `output_dir/` are intentionally excluded from version control.

## Linux 一键安装环境

Linux / NVIDIA GPU 用户可使用 [setup_env.sh](setup_env.sh)。请先安装 Conda、NVIDIA 驱动、CUDA Toolkit（含 `nvcc`）以及与 Toolkit 兼容的 GCC / G++。脚本不会使用 `sudo` 安装系统组件，也不会下载数据集或权重。

```bash
git clone https://github.com/XiaowangjiAaa/MAMBA_YOLO.git
cd MAMBA_YOLO
bash setup_env.sh
```

默认创建或复用 `mambayolo` 环境（Python `3.11`），安装 PyTorch `2.3.0` / torchvision `0.18.0` / torchaudio `2.3.0` 的 `cu121` wheel、项目依赖和当前仓库，编译 `selective_scan_cuda_core`，最后执行导入和 GPU 前向检查。已有环境的 Python 版本必须为 `3.11`；已有 core 扩展可导入时默认跳过编译。安装日志写入 `setup_env_<时间戳>.log`。editable 打包失败时会回退为 `.pth` 注册仓库路径。

```bash
# 查看参数，不执行安装
bash setup_env.sh --help

# 使用清华 PyPI 镜像，降低编译并行数
bash setup_env.sh --china-mirror --max-jobs 2

# 指定环境和本机实际 CUDA Toolkit 路径
bash setup_env.sh --env-name mambayolo --cuda-home /usr/local/cuda-12.1

# 指定 Conda 可执行文件；也支持 --cuda-tag cu118
bash setup_env.sh --conda-exe "$HOME/miniconda3/bin/conda"
```

`--full-selective-scan` 编译 core、ndstate、oflex 全部模块；默认只编译项目使用的 core。`--skip-verify` 跳过最终导入和前向检查，GPU 可用性仍是安装的必要条件。若 CUDA 扩展编译失败，请检查日志中的 CUDA Toolkit、PyTorch CUDA 和 GCC / G++ 兼容性。

安装后，自行准备分割数据集 YAML 并运行：

```bash
conda activate mambayolo
python batch_finetune.py --data /path/to/data.yaml --dry-run
python batch_finetune.py --data /path/to/data.yaml
```

当前 `batch_finetune.py` 从 YAML 随机初始化模型，使用 `pretrained=False`、`resume=False`，从头训练。

## Windows 一键安装环境

本仓库提供 [setup_env.ps1](setup_env.ps1)，用于自动创建或复用 Conda 环境、安装 Python 依赖、编译 CUDA 扩展并执行环境自检。

### 前置要求

请先安装 Miniconda / Anaconda、NVIDIA 显卡驱动、CUDA Toolkit（含 `nvcc`），以及 Visual Studio Build Tools 的“使用 C++ 的桌面开发”工作负载。脚本会自动查找这些工具，但不会安装它们。

### 安装命令

在 PowerShell 中运行：

```powershell
git clone https://github.com/XiaowangjiAaa/MAMBA_YOLO.git
cd MAMBA_YOLO
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```

已有仓库时，在仓库根目录运行最后一条命令即可。默认环境名为 `mambayolo`，Python 为 `3.11`。脚本会依次安装：

1. pip、setuptools、wheel 工具链。
2. PyTorch `2.3.0`、torchvision `0.18.0`、torchaudio `2.3.0`，默认使用 CUDA `12.1` wheel（`cu121`）。CUDA Toolkit 是另外安装的本地编译工具，需与 PyTorch CUDA 版本及编译器兼容。
3. 项目 Python 依赖，并固定 NumPy `1.26.4`。
4. 当前仓库的 `ultralytics`（editable 安装）。
5. 本项目使用的 `selective_scan_cuda_core` 扩展。
6. 导入检查和 CUDA 前向检查；无可用 CUDA 时会跳过前向检查。

安装日志保存为仓库根目录的 `setup_env_<时间戳>.log`。脚本可重复运行，默认复用已有环境；已有可导入的 core 扩展会跳过编译。只有脚本成功结束且 CUDA 检查通过，才说明 GPU 训练环境已验证。

### 常用参数

```powershell
# 使用清华 PyPI 镜像；PyTorch wheel 仍从官方源下载
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -UseChinaMirror

# 自定义环境名称和编译并行数
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -EnvName mambayolo -MaxJobs 2

# 自动查找失败时，指定本机 Conda 和 CUDA Toolkit 的实际安装路径
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1 -CondaExe 'C:\Miniconda3\Scripts\conda.exe' -CudaHome 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1'
```

`-Recreate` 会删除同名 Conda 环境并重建，请按需使用。`-FullSelectiveScan` 可编译 core、ndstate、oflex 全部模块。`-SkipSelectiveScan` 或 `-CudaTag cpu` 不能保证 Mamba GPU 训练环境可用；未安装 core 扩展时，脚本会以非零退出码结束。`-SkipVerify` 会跳过最后的环境检查。

### 安装后从头训练

```powershell
conda activate mambayolo
python batch_finetune.py --data C:\datasets\crack-seg\crack-seg.yaml --dry-run
python batch_finetune.py --data C:\datasets\crack-seg\crack-seg.yaml
```

请把示例数据路径替换为自己的分割数据集 YAML；数据集需自行准备，仓库不包含数据集、预训练权重、训练输出或预测输出。虽然脚本名为 `batch_finetune.py`，当前实现从模型 YAML 随机初始化，设置 `pretrained=False` 和 `resume=False`，不读取已有训练权重。模型、训练轮数、批大小和种子等参数在脚本顶部配置。

## Upstream project

This project is derived from **[AAAI 2025] Mamba YOLO: A Simple Baseline for Object Detection with State Space Model**.

![Python 3.11](https://img.shields.io/badge/python-3.11-g) ![pytorch 2.3.0](https://img.shields.io/badge/pytorch-2.3.0-blue.svg) [![docs](https://img.shields.io/badge/docs-latest-blue)](README.md)


<div align="center">
  <img src="./asserts/mambayolo.jpg" width="1200px"/>
</div>

## Model Zoo

We've pre-trained YOLO-World-T/M/L from scratch and evaluate on the `MSCOCO2017 val`. 

### Inference on MSCOCO2017 dataset


| model | Params| FLOPs | ${AP}^{val}$ | ${AP}_{{50}}^{val}$ | ${AP}_{{75}}^{val}$ | ${AP}_{{S}}^{val}$ | ${AP}_{{M}}^{val}$ | ${AP}_{{L}}^{val}$ |
| :------------------------------------------------------------------------------------------------------------------- | :------------------- | :----------------- | :--------------: | :------------: | :------------: | :------------: | :-------------: | :------------: |
| [Mamba YOLO-T](./ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml) | 5.8M | 13.2G |       44.5       |          61.2           |          48.2           |          24.7          |          48.8          |          62.0          |
| [Mamba YOLO-M](./ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml) | 19.1M | 45.4G  |       49.1       |          66.5           |          53.5           |          30.6          |          54.0          |          66.4          |
| [Mamba YOLO-L](./ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L.yaml)  | 57.6M | 156.2G |       52.1       |          69.8           |          56.5           |          34.1          |          57.3          |          68.1          |




## Getting started

### Manual installation (upstream reference)

Windows 用户可优先使用上面的 `setup_env.ps1`。以下保留上游手动安装和 COCO 检测示例；本仓库裂缝分割训练可使用上面的 `batch_finetune.py` 命令。

Mamba YOLO is developed based on `torch==2.3.0` `pytorch-cuda==12.1` and `CUDA Version==12.6`. 

#### 2.Clone Project 

```bash
git clone https://github.com/XiaowangjiAaa/MAMBA_YOLO.git
cd MAMBA_YOLO
```

#### 3.Create and activate a conda environment.
```bash
conda create -n mambayolo -y python=3.11
conda activate mambayolo
```

#### 4. Install torch

```bash
pip3 install torch===2.3.0 torchvision torchaudio
```

#### 5. Install Dependencies
```bash
pip install seaborn thop timm einops
cd selective_scan && pip install . && cd ..
pip install -v -e .
```

#### 6. Prepare MSCOCO2017 Dataset
Make sure your dataset structure as follows:
```
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
```

#### 7. Training Mamba-YOLO-T
```bash
python mbyolo_train.py --task train --data ultralytics/cfg/datasets/coco.yaml \
 --config ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml \
--amp  --project ./output_dir/mscoco --name mambayolo_n
```

## Acknowledgement

This repo is modified from open source real-time object detection codebase [Ultralytics](https://github.com/ultralytics/ultralytics). The selective-scan from [VMamba](https://github.com/MzeroMiko/VMamba).

## Citations
If you find [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO) is useful in your research or applications, please consider giving us a star 🌟 and citing it.

```bibtex
@misc{wang2024mambayolossmsbasedyolo,
      title={Mamba YOLO: SSMs-Based YOLO For Object Detection}, 
      author={Zeyu Wang and Chen Li and Huiying Xu and Xinzhong Zhu},
      year={2024},
      eprint={2406.05835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.05835}, 
}
```
