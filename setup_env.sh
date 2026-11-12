#!/usr/bin/env bash
# Linux + Conda + NVIDIA GPU environment setup. Run: bash setup_env.sh
set -Eeuo pipefail

ENV_NAME=mambayolo
PYTHON_VERSION=3.11
CUDA_TAG=cu121
MAX_JOBS=4
FULL_SCAN=0
SKIP_VERIFY=0
CHINA_MIRROR=0
CONDA_EXE=${CONDA_EXE:-conda}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_ROOT=''

usage() {
    cat <<'HELP'
Usage: bash setup_env.sh [options]
  --env-name NAME         Conda environment (default: mambayolo; reused if present)
  --conda-exe PATH        Conda executable
  --cuda-tag cu121|cu118  PyTorch CUDA wheel (default: cu121)
  --cuda-home PATH        CUDA Toolkit directory containing bin/nvcc
  --max-jobs N            Parallel CUDA build jobs (default: 4)
  --china-mirror          Use Tsinghua PyPI mirror (torch uses official index)
  --full-selective-scan   Build core, ndstate and oflex (default: core only)
  --skip-verify           Skip final import and GPU forward verification
  -h, --help              Show help without installing anything

Prerequisites: Linux, Conda, NVIDIA driver, CUDA Toolkit, compatible gcc/g++.
This script installs Python dependencies; it does not install system tools or data.
HELP
}
while (($#)); do
    case "$1" in
        --env-name|--conda-exe|--cuda-tag|--cuda-home|--max-jobs)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                echo "Missing value for $1" >&2; exit 2
            fi
            case "$1" in
                --env-name) ENV_NAME=$2 ;;
                --conda-exe) CONDA_EXE=$2 ;;
                --cuda-tag) CUDA_TAG=$2 ;;
                --cuda-home) export CUDA_HOME=$2 ;;
                --max-jobs) MAX_JOBS=$2 ;;
            esac
            shift 2 ;;
        --china-mirror) CHINA_MIRROR=1; shift ;;
        --full-selective-scan) FULL_SCAN=1; shift ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done
[[ "$CUDA_TAG" == cu121 || "$CUDA_TAG" == cu118 ]] || { echo 'Use cu121 or cu118' >&2; exit 2; }
[[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'max-jobs must be a positive integer' >&2; exit 2; }
[[ "$ENV_NAME" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] || { echo 'Invalid environment name' >&2; exit 2; }
[[ "$(uname -s)" == Linux ]] || { echo 'This script requires Linux.' >&2; exit 1; }
command -v "$CONDA_EXE" >/dev/null || { echo 'Install Conda first, or pass --conda-exe.' >&2; exit 1; }
for tool in gcc g++ tee mktemp; do
    command -v "$tool" >/dev/null || { echo "Required tool missing: $tool" >&2; exit 1; }
done
if [[ -z "${CUDA_HOME:-}" ]]; then
    command -v nvcc >/dev/null || { echo 'Install CUDA Toolkit or pass --cuda-home.' >&2; exit 1; }
    CUDA_HOME=$(dirname -- "$(dirname -- "$(readlink -f -- "$(command -v nvcc)")")")
    export CUDA_HOME
fi
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "nvcc not found: $CUDA_HOME/bin/nvcc" >&2; exit 1; }
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS
export PYTHONUNBUFFERED=1
LOG_FILE="$REPO_ROOT/setup_env_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
cleanup() {
    if [[ -n "$BUILD_ROOT" && -d "$BUILD_ROOT" ]]; then
        rm -rf -- "$BUILD_ROOT"
    fi
}
trap cleanup EXIT
trap 'echo "[FAIL] Installation stopped at line $LINENO. See $LOG_FILE" >&2' ERR
echo "[INFO] Repo: $REPO_ROOT; environment: $ENV_NAME; CUDA_HOME: $CUDA_HOME"
"$CUDA_HOME/bin/nvcc" --version
CONDA_BASE=$("$CONDA_EXE" info --base)
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"
if ! conda activate "$ENV_NAME"; then
    "$CONDA_EXE" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    conda activate "$ENV_NAME"
fi
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit('Use a Python 3.11 environment; choose a new --env-name.')
print('[INFO] Python:', sys.executable)
PY
PIP_INDEX=()
if ((CHINA_MIRROR)); then
    PIP_INDEX=(--index-url https://pypi.tuna.tsinghua.edu.cn/simple)
fi
pip_install() { python -m pip install --disable-pip-version-check --no-input "${PIP_INDEX[@]}" "$@"; }
pip_install --upgrade pip setuptools wheel
python -m pip install --disable-pip-version-check --no-input \
    --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
    torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 numpy==1.26.4
pip_install numpy==1.26.4 opencv-python==4.10.0.84 pillow==10.4.0 PyYAML==6.0.2 \
    requests==2.32.3 scipy==1.13.1 tqdm==4.66.5 psutil==6.0.0 py-cpuinfo==9.0.0 \
    pandas==2.2.3 seaborn==0.13.2 matplotlib==3.9.2 ultralytics-thop==2.0.0 \
    timm==1.0.9 einops==0.8.0 'packaging>=23.2' 'ninja>=1.11' pycocotools==2.0.8
if ! pip_install tensorboard==2.17.1; then
    echo '[WARN] Optional TensorBoard installation failed.'
fi
if ! pip_install --no-deps --no-build-isolation -e "$REPO_ROOT"; then
    echo '[WARN] Editable packaging failed; registering repository through a .pth file.'
    python - "$REPO_ROOT" <<'PY'
import pathlib, sys, sysconfig
target = pathlib.Path(sysconfig.get_paths()['purelib']) / 'mamba_yolo_repo.pth'
target.write_text(sys.argv[1] + '\n', encoding='utf-8')
print('[INFO] Wrote', target)
PY
fi
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('No usable NVIDIA GPU detected; check the driver and CUDA wheel.')
print('[INFO] GPU:', torch.cuda.get_device_name(0), 'PyTorch CUDA:', torch.version.cuda)
PY
if ((FULL_SCAN == 0)) && python -c 'import torch, selective_scan_cuda_core'; then
    echo '[OK] Existing selective_scan_cuda_core is importable; skipping build.'
else
    # Build a private copy, leaving the repository source unchanged.
    BUILD_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/mambayolo-scan.XXXXXXXX")
    cp -a -- "$REPO_ROOT/selective_scan" "$BUILD_ROOT/selective_scan"
    python - "$BUILD_ROOT/selective_scan/setup.py" "$FULL_SCAN" <<'PY'
import pathlib, re, sys
path = pathlib.Path(sys.argv[1])
modes = '["core", "ndstate", "oflex"]' if sys.argv[2] == '1' else '["core"]'
source, count = re.subn(r'^MODES\s*=.*$', 'MODES = ' + modes,
                        path.read_text(encoding='utf-8'), flags=re.MULTILINE)
if count != 1:
    raise SystemExit('Expected exactly one MODES assignment in selective_scan/setup.py')
path.write_text(source, encoding='utf-8')
PY
    pip_install --no-deps --no-build-isolation "$BUILD_ROOT/selective_scan"
fi
if ((SKIP_VERIFY == 0)); then
    python - "$REPO_ROOT" <<'PY'
import pathlib, sys
import torch, torchvision, cv2, einops, numpy, pandas, scipy, seaborn, timm, yaml
import selective_scan_cuda_core
import ultralytics
repo = pathlib.Path(sys.argv[1]).resolve()
if pathlib.Path(ultralytics.__file__).resolve().parent != repo / 'ultralytics':
    raise SystemExit('Imported ultralytics from a different repository/environment')
from ultralytics import YOLO
model = YOLO(str(repo / 'ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml'))
model.model.eval().to('cuda:0')
with torch.no_grad():
    model.model(torch.randn(1, 3, 128, 128, device='cuda:0'))
torch.cuda.synchronize()
print('[OK] Imports and Mamba-YOLO CUDA forward passed.')
PY
else
    echo '[WARN] Final import and forward verification skipped.'
fi
echo "[OK] Installation finished. Log: $LOG_FILE"
printf 'Next: conda activate %s\n' "$ENV_NAME"
echo 'Training: python batch_finetune.py --data /path/to/data.yaml'
