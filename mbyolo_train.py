from ultralytics import YOLO
import argparse
import os
import shutil
import tempfile

import torch

ROOT = os.path.abspath('.') + "/"


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT + '/ultralytics/cfg/datasets/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--config', type=str,
                        default=ROOT + '/ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T-yolo11.yaml',
                        help='model config YAML path. Use Mamba-YOLO-T-yolo11.yaml for YOLO11-based, '
                             'or Mamba-YOLO-T.yaml for original YOLOv8-based')
    parser.add_argument('--weights', type=str, default='', help='trained .pth / .pt weights file (for val/test/predict)')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--task', default='train', help='train, val, test, speed or study')
    parser.add_argument('--device', default='0,1,2,3,4,5,6,7', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=128, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100, help='early-stopping patience; use >= epochs for full run')
    parser.add_argument('--seed', type=int, default=0, help='training random seed')
    parser.add_argument('--save_period', type=int, default=-1, help='save checkpoint every N epochs; -1 disables')
    parser.add_argument('--crack_metric_conf', type=float, default=0.25,
                        help='confidence threshold for foreground-union mIoU/clDice')
    parser.add_argument('--cldice_iters', type=int, default=20,
                        help='morphological skeleton iterations used by clDice validation')
    parser.add_argument('--optimizer', default='SGD', help='SGD, Adam, AdamW')
    parser.add_argument('--amp', action='store_true', help='open amp')
    parser.add_argument('--project', default=ROOT + '/output_dir/mscoco', help='save to project/name')
    parser.add_argument('--name', default='mambayolo11', help='save to project/name')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--source', type=str, default='', help='image/video source for predict task')
    parser.add_argument('--nc', type=int, default=None, help='(optional) override number of classes')
    opt = parser.parse_args()
    return opt


def _load_model_from_weights(weights_path, task=None):
    """
    Load a YOLO model from a trained .pt or .pth checkpoint.

    Ultralytics' YOLO() constructor naturally handles .pt files (it reads the full
    checkpoint, including metadata like nc, so the model matches the training config).
    For .pth files we create a temporary symlink/copy with .pt extension.
    """
    ext = os.path.splitext(weights_path)[1].lower()

    if ext == '.pt':
        print(f"[INFO] Loading checkpoint: {weights_path}")
        return YOLO(weights_path, task=task)

    # For .pth files, try loading directly first (some are actually .pt-format files)
    print(f"[INFO] Loading .pth checkpoint: {weights_path}")
    try:
        model = YOLO(weights_path, task=task)
        print("[INFO] .pth loaded via YOLO() directly")
        return model
    except Exception as e:
        print(f"[WARN] Direct .pth load failed ({e}), trying via .pt extension...")

    # Fallback: copy to .pt and load
    tmp_pt = weights_path.rsplit('.', 1)[0] + '_tmp.pt'
    try:
        shutil.copy2(weights_path, tmp_pt)
        model = YOLO(tmp_pt, task=task)
        print(f"[INFO] Loaded via temporary {tmp_pt}")
        return model
    finally:
        if os.path.exists(tmp_pt):
            os.remove(tmp_pt)


if __name__ == '__main__':
    opt = parse_opt()
    task = opt.task

    args = {
        "data": opt.data,
        "epochs": opt.epochs,
        "patience": opt.patience,
        "seed": opt.seed,
        "save_period": opt.save_period,
        "crack_metric_conf": opt.crack_metric_conf,
        "cldice_iters": opt.cldice_iters,
        "workers": opt.workers,
        "batch": opt.batch_size,
        "imgsz": opt.imgsz,
        "optimizer": opt.optimizer,
        "device": opt.device,
        "amp": opt.amp,
        "project": opt.project,
        "name": opt.name,
    }

    # --- Load model: from weights (preserving original nc) or from YAML ---
    if opt.weights:
        model = _load_model_from_weights(opt.weights, task=task)
        # Override nc if user specified (must match checkpoint)
        if opt.nc is not None:
            model.model.args['nc'] = opt.nc
    else:
        print(f"[INFO] Building model from config: {opt.config}")
        model = YOLO(opt.config)
        if opt.nc is not None:
            model.model.args['nc'] = opt.nc

    # --- Execute task ---
    if task == 'train':
        model.train(**args)
    elif task in ('val', 'test'):
        model.val(**args)
    elif task == 'predict':
        source = opt.source
        if not source:
            source = input("Enter image/video path: ")
        model.predict(source=source, save=True, device=opt.device)
    elif task == 'speed':
        model.val(**args, plots=False)
    else:
        print(f"Unknown task: '{task}'. Supported: train, val, test, predict, speed")
