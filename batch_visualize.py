import cv2
import numpy as np
from pathlib import Path
import random

def load_yolo_seg_txt(txt_path, img_w, img_h):
    res = []
    if not Path(txt_path).exists():
        return res
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            if len(coords) < 6 or len(coords) % 2 != 0:
                continue
            pts = np.array(coords, dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            pts = np.round(pts).astype(np.int32)
            res.append((cls, pts))
    return res

def draw_polys(img, polys, thickness=2):
    out = img.copy()
    for cls, pts in polys:
        pts2 = pts.reshape(-1, 1, 2)
        cv2.polylines(out, [pts2], True, (0, 255, 0), thickness)
    return out

def batch_visualize(images_dir, labels_dir, out_dir, sample_n=30, seed=42):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = []
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        imgs += list(images_dir.glob(f"*{ext}"))
    imgs = sorted(imgs)

    random.seed(seed)
    if sample_n is not None and sample_n < len(imgs):
        imgs = random.sample(imgs, sample_n)

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        txt_path = labels_dir / f"{img_path.stem}.txt"

        polys = load_yolo_seg_txt(txt_path, w, h)
        vis = draw_polys(img, polys, thickness=2)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), vis)

    print("Saved to:", out_dir)

# === 改成你自己的路径 ===
batch_visualize(
    images_dir=r"crack-seg\images\train",
    labels_dir=r"crack-seg\labels\train",
    out_dir=r"crack-seg\images\vis_train_sample",
    sample_n=None  # None=全量；50=抽50张
)
