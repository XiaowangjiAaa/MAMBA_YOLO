"""Reproducible paired segmentation, feature, routing and memory figures.

Runs one model/image at a time. All panels use the SAME unfused FP32 forward.
Only load trusted checkpoints: Ultralytics .pt files contain Python objects.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--runs-dir', type=Path, default=Path('output_dir/crack-seg-9.11'))
    p.add_argument('--experiments', nargs='+', default=['K01', 'K06', 'K10'], help='IDs or all: discover every run/seed with the selected checkpoint')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--checkpoint', default='best_map50.pt')
    p.add_argument('--manifest', type=Path, help='JSON list of model specs; layers/mechanism_layers may override auto selection')
    p.add_argument('--source', required=True, help='Image / directory / TXT / dataset YAML')
    p.add_argument('--split', choices=['train', 'val', 'test', 'external', 'all'], default='val', help='all = YAML train/val/test; external = label a direct image folder as an external set')
    p.add_argument('--gt-dir', type=Path, help='Optional directory of raster GT masks matched to images by filename stem')
    p.add_argument('--gt-invert', action='store_true', help='Treat zero pixels as foreground in raster GT masks')
    p.add_argument('--layers', '--feature-layers', dest='layers', default='auto', help='auto = discover backbone/neck stages; or comma-separated layer indices/names')
    p.add_argument('--mechanism-layers', default='auto', help='auto = every actual CPSB state (independent of feature layers); none = disabled; or parent layer indices/names')
    p.add_argument('--max-images', type=int, default=6, help='0 = all images; random selection independent of predictions')
    p.add_argument('--sample-seed', type=int, default=1)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--iou', type=float, default=.7, help='NMS IoU, not mask matching threshold')
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--max-paths-draw', type=int, default=32, help='Only limits drawing, NOT coverage statistics')
    p.add_argument('--roi', help='Optional fixed normalized x1,y1,x2,y2 crop for ALL models/images')
    p.add_argument('--qualitative-scanning', action='store_true',
                   help='Create a paper-style Input/GT/Raster/Cross/... error-mask grid with identical ROIs')
    p.add_argument('--qualitative-only', action='store_true',
                   help='Skip standard feature/path panels; keep requested qualitative and/or scan-influence figures')
    p.add_argument('--qualitative-cases', type=Path,
                   help='Optional JSON list: [{"label":"Curved","image":"name.jpg","roi":[x1,y1,x2,y2]}]; coordinates are normalized')
    p.add_argument('--qualitative-roi-size', type=float, default=.28,
                   help='GT-centred automatic square ROI side as a fraction of the shorter image side')
    p.add_argument('--scan-influence', action='store_true',
                   help='Export exact scan-residual intensity and memory-delta maps; compare scanning methods on P3/P4')
    p.add_argument('--scan-influence-roles', default='backbone_P3,backbone_P4',
                   help='Comma-separated semantic CPSB roles used in cross-method scan-influence figures')
    p.add_argument('--scan-influence-top-fraction', type=float, default=.10,
                   help='Fraction of valid feature cells used for top-response GT hit rate')
    p.add_argument('--scan-influence-gt-dilate', type=int, default=1,
                   help='GT dilation radius in feature cells for scan-focus metrics and contour display')
    p.add_argument('--output-dir', type=Path, default=Path('paper_visualizations/9.11'),
                   help='Resumable output directory; matching complete image/model items are skipped')
    return p.parse_args()


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def scatter_path(values, indices, valid, shape):
    """Mean over ALL valid visits; unvisited locations stay NaN (not zero)."""
    idx = indices[valid].astype(np.int64)
    count = np.bincount(idx, minlength=int(np.prod(shape)))
    sums = np.bincount(idx, weights=values[valid], minlength=count.size)
    result = np.full(count.size, np.nan, dtype=np.float32)
    np.divide(sums, count, out=result, where=count > 0)
    return result.reshape(shape), (count > 0).reshape(shape)


def memory_diagnostics(module, inputs):
    """Reconstruct forward-direction recurrence decay, without changing forward.

    Decay is averaged over channels/state dimensions, not interpreted as a
    calibrated memory lifetime. Baseline uses the same x, A, bias without gates.
    """
    import torch
    import torch.nn.functional as F
    x, p, edge, valid = inputs
    x = x.float() * valid.float()
    projected = torch.einsum('ncl,dc->ndl', x, module.x_proj_weight[0].float())
    dt = torch.einsum('nrl,cr->ncl', projected[:, :module.dt_rank], module.dt_projs_weight[0].float())
    base = dt + module.dt_projs_bias[0].float()[None, :, None]
    retention = module.effective_memory().float() * (.5 - p.float())
    transition = module.effective_transition().float() * (1 - edge.float())
    delta = F.softplus(base + retention + transition)
    delta_base = F.softplus(base)
    a = -module.A_logs.float().exp().view(2, module.channels, module.d_state)[0]
    result = {
        'retention_bias': retention[:, 0], 'transition_bias': transition[:, 0],
        'write_gain': (1 + module.effective_write().float() * (2*p.float()-1))[:, 0],
        'decay': torch.exp(delta[:, :, None, :] * a[None, :, :, None]).mean((1, 2)),
        'decay_without_gates': torch.exp(delta_base[:, :, None, :] * a[None, :, :, None]).mean((1, 2)),
        'predecessor': edge[:, 0],
    }
    return {k: v.detach().cpu().numpy() for k, v in result.items()}


def resolve_specs(args):
    if args.manifest:
        specs = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
        for spec in specs:
            for key in ('weights', 'yaml'):
                if spec.get(key) and not Path(spec[key]).is_absolute():
                    spec[key] = str(args.manifest.parent / spec[key])
    elif any(e.lower() == 'all' for e in args.experiments):
        if len(args.experiments) != 1:
            raise ValueError('Use --experiments all alone, not mixed with IDs')
        specs = []
        for run in sorted(args.runs_dir.iterdir()):
            if not run.is_dir():
                continue
            weight = run / 'weights' / args.checkpoint
            if not weight.is_file():
                print(f'[SKIP] {run.name}: missing weights/{args.checkpoint}', flush=True)
                continue
            specs.append({'label': run.name, 'weights': str(weight)})
    else:
        specs = [{'label': eid, 'weights': str(args.runs_dir / f'{eid}_crack-seg_seed{args.seed}' / 'weights' / args.checkpoint)}
                 for eid in args.experiments]
    if not specs or len({s['label'] for s in specs}) != len(specs):
        raise ValueError('Model labels must be nonempty and unique')
    for s in specs:
        if s['label'].lower() == 'comparisons':
            raise ValueError('Model label comparisons is reserved for paired output')
        if not s['label'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in s['label']):
            raise ValueError('Use simple ASCII letters/digits/_/- for model labels')
        for key in ('weights', 'yaml'):
            if s.get(key):
                path = Path(s[key]).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                s[key] = str(path)
                s[key + '_sha256'] = fingerprint(path)
        s['layers'] = s.get('layers', args.layers)
        s['mechanism_layers'] = s.get('mechanism_layers', getattr(args, 'mechanism_layers', 'auto'))
    return specs


def select_stage_layers(core, shapes):
    """Use actual spatial strides and head inputs, not family-specific indices."""
    backbone_count = len(core.yaml.get('backbone', []))
    if not backbone_count:
        raise ValueError('Auto layers require checkpoint model.yaml.backbone; use --layers explicitly')
    selected = {}
    stage_types = ('C3', 'C2f', 'C2F', 'C3k2', 'A2C2f', 'RepNCSPELAN4', 'VSS', 'CrackPath', 'CASP')
    for stage, stride in [('P2',4), ('P3',8), ('P4',16), ('P5',32)]:
        candidates = [i for i in range(backbone_count) if shapes.get(i) == stride]
        blocks = [i for i in candidates if any(t in type(core.model[i]).__name__ for t in stage_types)]
        if candidates:
            selected['backbone_'+stage] = 'model.'+str((blocks or candidates)[-1])
    head = core.model[-1]
    sources = head.f if isinstance(head.f, (list, tuple)) else [head.f]
    for source in sources:
        index = source if source >= 0 else len(core.model)-1+source
        stride = shapes.get(index)
        if stride in (8,16,32):
            key = 'neck_P'+str({8:3,16:4,32:5}[stride])
            if key in selected:
                raise ValueError(f'Ambiguous head input for {key}; select layers manually')
            selected[key] = 'model.'+str(index)
    if not selected:
        raise ValueError('Could not resolve spatial stages; select layers manually')
    return selected


def discover_layers(core, spec, imgsz, device):
    import torch
    from compare_model_features import resolve_layer_names
    modules = dict(core.named_modules())
    # Observe strides in both auto and manual feature-layer modes. Mechanism
    # modules still need these shapes to receive a truthful backbone/neck P-level.
    shapes, handles = {}, []
    for i, module in enumerate(core.model):
        def hook(m, inputs, output, i=i):
            if torch.is_tensor(output) and output.ndim == 4:
                h,w = output.shape[-2:]
                if h and w and imgsz % h == 0 and imgsz % w == 0 and imgsz//h == imgsz//w:
                    shapes[i] = imgsz//h
        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            core(torch.zeros(1,3,imgsz,imgsz,device=device))
    finally:
        for handle in handles:
            handle.remove()
    if spec['layers'].lower() == 'auto':
        features = select_stage_layers(core, shapes)
    else:
        features = {f'manual_{i}': n for i,n in enumerate(resolve_layer_names(core, spec['layers']))}
    policy = spec['mechanism_layers'].lower()
    parents = None if policy in ('auto','none') else resolve_layer_names(core, spec['mechanism_layers'])
    states = {n:m for n,m in modules.items() if hasattr(m,'path_ssm') and hasattr(m,'last_path_indices')
              and policy != 'none' and (parents is None or any(n == p or n.startswith(p+'.') for p in parents))}
    if parents and not states:
        raise ValueError('Selected mechanism layers contain no CPSB path state')
    backbone_count = len(core.yaml.get('backbone', []))
    stage_for_stride = {4:'P2', 8:'P3', 16:'P4', 32:'P5'}
    mechanism_roles = {}
    for name in states:
        match = re.match(r'^model\.(\d+)(?:\.|$)', name)
        index = int(match.group(1)) if match else None
        exact = next((role for role,parent in features.items()
                      if name == parent or name.startswith(parent+'.')), None)
        if exact:
            mechanism_roles[name] = exact
        elif index is not None and shapes.get(index) in stage_for_stride:
            region = 'backbone' if index < backbone_count else 'neck_intermediate'
            mechanism_roles[name] = region+'_'+stage_for_stride[shapes[index]]
        else:
            mechanism_roles[name] = 'non_output_stage'
    plan = {'features': features, 'mechanisms': list(states), 'mechanism_roles': mechanism_roles,
            'mechanism_status': 'found' if states else ('disabled' if policy=='none' else 'no CPSB in model'),
            'selection': 'Observed spatial strides + checkpoint backbone boundary + segmentation head inputs'}
    return plan, states


def semantic_role(module_name, plan):
    """Map a nested CPSB state back to its containing semantic top-level stage."""
    if module_name in plan.get('mechanism_roles', {}):
        return plan['mechanism_roles'][module_name]
    matches = [(role, parent) for role, parent in plan['features'].items()
               if module_name == parent or module_name.startswith(parent + '.')]
    return max(matches, key=lambda item: len(item[1]))[0] if matches else 'non_output_stage'


def requested_feature_items(plan, qualitative_only):
    """Do not consume ordinary feature hooks in influence-only fast mode."""
    return [] if qualitative_only else list(plan['features'].items())


def comparison_groups(specs):
    """K01 is shared by the three ablations; compare family pairs separately."""
    definitions = {
        'scanning': ({'K01','K02','K03','K04','K05','K06'}, 'backbone_'),
        'memory': ({'K01','K10','K11','K12','K13'}, 'backbone_'),
        'placement': ({'K00','K01','K20','K21','K22','K23','K24','K26','K27'}, 'neck_'),
        'family_v5': ({'K30','K31'}, 'neck_'), 'family_v8': ({'K32','K33'}, 'neck_'),
        'family_v9': ({'K28','K29'}, 'neck_'), 'family_v11': ({'K00','K01','K34','K35'}, 'neck_'),
        'family_v12': ({'K38','K39'}, 'neck_'), 'family_v26_compat': ({'K36','K37'}, 'neck_')}
    groups, used = [], set()
    for name,(ids,prefix) in definitions.items():
        subset = [s for s in specs if re.split(r'[_-]',s['label'])[0] in ids]
        if len(subset) >= 2:
            groups.append((name, subset, prefix))
            used.update(s['label'] for s in subset)
    remaining = [s for s in specs if s['label'] not in used]
    if remaining:
        groups.append(('other', remaining, ''))
    return groups


def resolve_splits(source, requested):
    if requested == 'external':
        if source.suffix.lower() in {'.yaml', '.yml'}:
            raise ValueError('--split external is for an image/TXT/directory source, not a dataset YAML')
        return ['external']
    if requested != 'all':
        return [requested]
    if source.suffix.lower() not in {'.yaml', '.yml'}:
        raise ValueError('--split all requires a dataset YAML as --source')
    import yaml
    data = yaml.safe_load(source.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError('Expected a dataset YAML mapping')
    splits = [s for s in ('train', 'val', 'test') if data.get(s)]
    if not splits:
        raise ValueError('Dataset YAML declares no train/val/test sources')
    for s in ('train', 'val', 'test'):
        if s not in splits:
            print(f'[SKIP] Dataset split {s} is not declared', flush=True)
    return splits


def build_gt_index(gt_dir):
    """Index raster masks by stem and reject ambiguous names."""
    if gt_dir is None:
        return None
    gt_dir = Path(gt_dir).resolve()
    if not gt_dir.is_dir():
        raise NotADirectoryError(gt_dir)
    supported = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    index = {}
    for path in sorted(p for p in gt_dir.rglob('*') if p.is_file() and p.suffix.lower() in supported):
        key = path.stem.casefold()
        if key in index:
            raise ValueError(f'Ambiguous GT stem {path.stem!r}: {index[key]} and {path}')
        index[key] = path
    if not index:
        raise ValueError(f'No raster GT masks found in {gt_dir}')
    return index


def load_external_gt(image_path, gt_index, image_shape, invert=False):
    """Load an already aligned binary raster GT mask without rotating or resizing it."""
    from PIL import Image
    path = gt_index.get(image_path.stem.casefold())
    if path is None:
        raise FileNotFoundError(f'No GT mask with stem {image_path.stem!r} for {image_path}')
    with Image.open(path) as source:
        raw = np.asarray(source)
    if raw.ndim == 3:
        raw = raw[..., :3].max(axis=2)
    mask = raw == 0 if invert else raw != 0
    if mask.shape != tuple(image_shape):
        raise ValueError(
            f'GT/image shape mismatch: {path} is {mask.shape}, {image_path} is {tuple(image_shape)}. '
            'Images and GT must already be spatially aligned; refusing to rotate or resize GT.'
        )
    return mask.astype(bool), path


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    """Read one of this script's CSV files while restoring scalar types."""
    path = Path(path)
    if not path.is_file():
        return []

    def scalar(value):
        if value == '':
            return ''
        if value.lower() in {'true', 'false'}:
            return value.lower() == 'true'
        try:
            return int(value) if re.fullmatch(r'[+-]?\d+', value) else float(value)
        except ValueError:
            return value

    with path.open('r', encoding='utf-8-sig', newline='') as handle:
        return [{key: scalar(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def completion_signature(args, spec, split, image_path, plan):
    """Identify outputs that are safe to reuse without hashing a large checkpoint."""
    image_stat = image_path.stat()
    checkpoint = Path(spec['weights']).resolve()
    checkpoint_stat = checkpoint.stat()
    return {
        'version': 1,
        'split': split,
        'image': str(image_path.resolve()),
        'image_size': image_stat.st_size,
        'image_mtime_ns': image_stat.st_mtime_ns,
        'checkpoint': str(checkpoint),
        'checkpoint_size': checkpoint_stat.st_size,
        'checkpoint_mtime_ns': checkpoint_stat.st_mtime_ns,
        'imgsz': args.imgsz,
        'conf': args.conf,
        'iou': args.iou,
        'max_det': args.max_det,
        'gt_dir': str(args.gt_dir.resolve()) if args.gt_dir else None,
        'gt_invert': args.gt_invert,
        'qualitative_only': args.qualitative_only,
        'scan_influence': args.scan_influence,
        'scan_influence_roles': args.scan_influence_roles,
        'scan_influence_top_fraction': args.scan_influence_top_fraction,
        'scan_influence_gt_dilate': args.scan_influence_gt_dilate,
        'features': plan['features'],
        'mechanisms': plan['mechanisms'],
    }


def load_completion(path, signature):
    """Return cached per-image result rows only when its signature matches exactly."""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload.get('rows') if payload.get('signature') == signature else None


def legacy_manifest_matches(path, args, split):
    """Allow one-time migration of complete outputs made before complete.json existed."""
    if not path.is_file():
        return False
    try:
        old = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if old.get('split') != split:
        return False
    old_args = old.get('arguments', {})
    keys = ('source', 'gt_dir', 'gt_invert', 'imgsz', 'conf', 'iou', 'max_det', 'qualitative_only',
            'scan_influence', 'scan_influence_roles', 'scan_influence_top_fraction',
            'scan_influence_gt_dilate')
    for key in keys:
        current = getattr(args, key)
        current = str(current) if isinstance(current, Path) else current
        if old_args.get(key) != current:
            return False
    return True


def legacy_completion(folder, spec, image_path, plan, states, args, previous):
    """Recover rows from a complete pre-resume run; reject every partial artifact set."""
    label, image = spec['label'], str(image_path)
    required = [folder / 'input.png', folder / 'gt.png', folder / f'{label}_prediction.png',
                folder / f'{label}_error.png']
    rows = {name: [row for row in values if row.get('model') == label and row.get('image') == image]
            for name, values in previous.items()}
    if len(rows['metrics']) != 1:
        return None
    if not args.qualitative_only:
        for role in plan['features']:
            required.extend((folder / f'{label}_{role}.npy', folder / f'{label}_{role}.png'))
        if len(rows['focus']) < len(plan['features']):
            return None
        for name in states:
            stem = folder / f'{label}_{name.replace(".", "_")}_structure'
            required.extend((stem.with_suffix('.png'), stem.with_suffix('.pdf')))
        if len(rows['routing']) < len(states):
            return None
    if args.scan_influence:
        for name in states:
            stem = folder / f'{label}_{name.replace(".", "_")}_scan_influence'
            required.extend((stem.with_suffix('.npz'), stem.with_suffix('.png'), stem.with_suffix('.pdf')))
        if len(rows['influence']) < len(states):
            return None
    return rows if all(path.is_file() for path in required) else None


SCAN_DISPLAY_NAMES = {
    'K01': 'Adaptive', 'K02': 'Raster', 'K03': 'Cross',
    'K04': 'Serpentine', 'K05': 'Diagonal', 'K06': 'Fixed straight',
}


def experiment_id(label):
    return re.split(r'[_-]', label)[0]


def scan_display_name(label):
    eid = experiment_id(label)
    return SCAN_DISPLAY_NAMES.get(eid, label)


def validate_normalized_roi(roi):
    roi = tuple(float(v) for v in roi)
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError(f'ROI {roi} must satisfy 0<=x1<x2<=1 and 0<=y1<y2<=1')
    return roi


def resolve_qualitative_cases(path, images):
    """Resolve an optional, prediction-independent list of paper figure cases."""
    if path is None:
        return [{'label': image.stem, 'image': image, 'roi': None} for image in images]
    raw = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(raw, list) or not raw:
        raise ValueError('--qualitative-cases must be a nonempty JSON list')
    by_name, by_stem = {}, {}
    for image in images:
        by_name.setdefault(image.name.casefold(), []).append(image)
        by_stem.setdefault(image.stem.casefold(), []).append(image)
    cases = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not item.get('image'):
            raise ValueError(f'Qualitative case {index} needs an image field')
        wanted = Path(str(item['image']))
        direct = wanted.resolve() if wanted.is_absolute() else None
        matches = [direct] if direct is not None and direct in images else by_name.get(wanted.name.casefold(), [])
        if not matches:
            matches = by_stem.get(wanted.stem.casefold(), [])
        if len(matches) != 1:
            raise ValueError(f'Qualitative image {item["image"]!r} matched {len(matches)} source images')
        roi = validate_normalized_roi(item['roi']) if item.get('roi') is not None else None
        cases.append({'label': str(item.get('label') or matches[0].stem), 'image': matches[0], 'roi': roi})
    return cases


def automatic_gt_roi(gt_mask, side_fraction=.28):
    """Choose an ROI from GT alone, never from one model's prediction."""
    if not 0 < side_fraction <= 1:
        raise ValueError('--qualitative-roi-size must be in (0, 1]')
    height, width = gt_mask.shape
    side = max(1, int(round(min(height, width) * side_fraction)))
    ys, xs = np.nonzero(gt_mask)
    if len(xs):
        # Median foreground location is stable for thin structures and cannot favour a model.
        cx, cy = float(np.median(xs)), float(np.median(ys))
    else:
        cx, cy = (width - 1) / 2, (height - 1) / 2
    x1 = int(np.clip(round(cx - side / 2), 0, max(width - side, 0)))
    y1 = int(np.clip(round(cy - side / 2), 0, max(height - side, 0)))
    return x1 / width, y1 / height, min(x1 + side, width) / width, min(y1 + side, height) / height


def mask_error_image(pred, gt):
    """Black-background RGB/BGR-safe evidence map: TP cyan, FN coral, FP purple."""
    result = np.zeros((*gt.shape, 3), dtype=np.uint8)
    result[pred & gt] = (255, 255, 0)       # BGR cyan
    result[~pred & gt] = (113, 116, 255)    # BGR coral/red
    result[pred & ~gt] = (237, 116, 176)    # BGR purple
    return result


def scan_influence_diagnostics(state_name, state, memory, gt_square, valid_pixels, meta, image_shape,
                               output, top_fraction=.10, gt_dilate=1):
    """Save exact scan-branch residual maps and prediction-independent focus statistics."""
    import cv2
    import matplotlib.pyplot as plt

    if state.last_route_update_rms is None or state.last_route_relative is None:
        return []
    absolute = state.last_route_update_rms[0].detach().cpu().numpy().astype(np.float32)
    relative = state.last_route_relative[0].detach().cpu().numpy().astype(np.float32)
    indices = state.last_path_indices[0].detach().cpu().numpy()
    valid = state.last_path_mask[0].detach().cpu().numpy().astype(bool)
    memory_delta, visited = scatter_path(
        memory['decay'] - memory['decay_without_gates'], indices, valid, absolute.shape
    )
    height, width = absolute.shape
    valid_small = cv2.resize(valid_pixels.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    gt_small = None
    gt_band = np.zeros_like(valid_small)
    if gt_square is not None:
        gt_small = cv2.resize(gt_square.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        if gt_dilate:
            size = 2 * gt_dilate + 1
            gt_small = cv2.dilate(gt_small, np.ones((size, size), np.uint8))
        gt_band = gt_small.astype(bool) & valid_small
    background = valid_small & ~gt_band
    crack_mean = float(relative[gt_band].mean()) if gt_band.any() else float('nan')
    background_mean = float(relative[background].mean()) if background.any() else float('nan')
    concentration = crack_mean / max(background_mean, 1e-12) if np.isfinite(crack_mean) else float('nan')
    valid_indices = np.flatnonzero(valid_small)
    top_hit = float('nan')
    if len(valid_indices) and gt_square is not None:
        count = max(1, int(np.ceil(len(valid_indices) * top_fraction)))
        scores = relative.reshape(-1)[valid_indices]
        chosen = valid_indices[np.argpartition(scores, -count)[-count:]]
        top_hit = float(gt_band.reshape(-1)[chosen].mean())
    np.savez_compressed(
        output.with_suffix('.npz'), absolute=absolute, relative=relative, memory_delta=memory_delta,
        visited=visited, gt_band=gt_band, valid_feature=valid_small,
        letterbox_meta=np.asarray(meta, dtype=np.float32), canvas_hw=np.asarray(valid_pixels.shape, dtype=np.int32),
        original_hw=np.asarray(image_shape, dtype=np.int32), module=np.asarray(state_name),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [(absolute, 'Absolute scan update RMS', 'magma', False),
              (relative, 'Relative scan update / input', 'magma', False),
              (memory_delta, 'Memory decay difference', 'coolwarm', True)]
    for ax, (values, title, cmap, signed) in zip(axes, panels):
        finite = values[np.isfinite(values)]
        limit = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
        limit = max(limit, 1e-12)
        im = ax.imshow(values, cmap=cmap, vmin=-limit if signed else 0, vmax=limit)
        if gt_band.any():
            ax.contour(gt_band.astype(float), levels=[.5], colors=['cyan'], linewidths=.7)
        ax.set_title(title + '\nlocal scale (comparison uses shared scale)', fontsize=9)
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=.65, extend='both' if signed else 'max')
    fig.suptitle(state_name + '; cyan contour = dilated GT band', fontsize=10)
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + suffix), dpi=220)
    plt.close(fig)
    return [{
        'module': state_name, 'feature_height': height, 'feature_width': width,
        'route_strength': float(state.effective_route().detach().cpu()),
        'absolute_mean_valid': float(absolute[valid_small].mean()) if valid_small.any() else float('nan'),
        'relative_mean_valid': float(relative[valid_small].mean()) if valid_small.any() else float('nan'),
        'relative_mean_crack_band': crack_mean, 'relative_mean_background': background_mean,
        'crack_concentration': concentration, 'top_response_gt_hit_rate': top_hit,
        'memory_delta_mean_visited': float(memory_delta[visited].mean()) if visited.any() else float('nan'),
        'npz': str(output.with_suffix('.npz')),
    }]


def restore_influence_map(values, payload, output_shape):
    """Undo the square letterbox so a feature-space diagnostic aligns with the original image."""
    import cv2
    canvas_h, canvas_w = map(int, payload['canvas_hw'])
    ratio, left, top, rw, rh = payload['letterbox_meta'].tolist()
    square = cv2.resize(np.nan_to_num(values, nan=0.0), (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    cropped = square[int(top):int(top + rh), int(left):int(left + rw)]
    return cv2.resize(cropped, output_shape[::-1], interpolation=cv2.INTER_LINEAR)


def save_scan_influence_detail(output, original, heat, limit, cmap, signed, gt=None):
    """Save text-free, full-resolution heatmap panels for direct visual inspection."""
    import cv2
    import matplotlib.pyplot as plt
    from visualize_seg_results import write_image

    output.mkdir(parents=True, exist_ok=True)
    safe_heat = np.nan_to_num(heat, nan=0.0, posinf=limit, neginf=-limit)
    if signed:
        normalized = np.clip((safe_heat + limit) / (2.0 * limit), 0.0, 1.0)
    else:
        normalized = np.clip(safe_heat / limit, 0.0, 1.0)
    color_rgb = (plt.get_cmap(cmap)(normalized)[..., :3] * 255.0).round().astype(np.uint8)
    color_bgr = color_rgb[..., ::-1]
    strength = np.clip(np.abs(safe_heat) / limit, 0.0, 1.0)
    alpha = (0.78 * strength)[..., None]
    overlay = np.clip(original.astype(np.float32) * (1.0 - alpha) + color_bgr.astype(np.float32) * alpha,
                      0, 255).astype(np.uint8)

    if not (output / 'heatmap.png').is_file():
        write_image(output / 'heatmap.png', color_bgr)
    if not any((output / name).is_file() for name in ('overlay.jpg', 'overlay.png')):
        write_image(output / 'overlay.jpg', overlay)
    if gt is not None:
        overlay_gt = overlay.copy()
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_gt, contours, -1, (255, 255, 0), max(1, round(max(original.shape[:2]) / 1000)))
        if not any((output / name).is_file() for name in ('overlay_gt.jpg', 'overlay_gt.png')):
            write_image(output / 'overlay_gt.jpg', overlay_gt)


def scan_influence_detail_complete(output, require_gt):
    """Accept both the original PNG layout and the compact resumable JPG layout."""
    required = (output / 'heatmap.png').is_file()
    required &= any((output / name).is_file() for name in ('overlay.jpg', 'overlay.png'))
    if require_gt:
        required &= any((output / name).is_file() for name in ('overlay_gt.jpg', 'overlay_gt.png'))
    return bool(required)


def render_scan_influence_comparison(args, split, images, specs, rows, group_out):
    """Compare exact scan updates with one shared scale per image/semantic level."""
    import cv2
    import matplotlib.pyplot as plt
    from visualize_seg_results import read_image, write_image

    wanted_roles = [r.strip() for r in args.scan_influence_roles.split(',') if r.strip()]
    lookup = {}
    for row in rows:
        if row['model'] in {s['label'] for s in specs} and row['semantic_role'] in wanted_roles:
            key = (row['model'], row['image'], row['semantic_role'])
            if key in lookup:
                raise ValueError(f'Multiple scan states map to {key}; choose an unambiguous semantic role/module')
            lookup[key] = row
    output = group_out / 'scan_influence'
    output.mkdir(exist_ok=True)
    previous_scales = read_rows(output / 'shared_scales.csv')
    scale_lookup = {(row.get('split'), row.get('image'), row.get('role'), row.get('field')): row
                    for row in previous_scales}
    for image_index, image_path in enumerate(images):
        prefix = f'{image_index:04d}_{image_path.stem}'
        print(f'  Scan-influence render {image_index + 1}/{len(images)}: {image_path.name}', flush=True)
        first = args.output_dir.resolve() / specs[0]['label'] / split / prefix
        original = read_image(first / 'input.png')
        gt_file = first / 'gt_union.png'
        gt = read_image(gt_file)[..., 0] > 0 if gt_file.is_file() else None
        image_detail_root = output / 'details' / prefix
        image_detail_root.mkdir(parents=True, exist_ok=True)
        if not any((image_detail_root / name).is_file() for name in ('input.jpg', 'input.png')):
            write_image(image_detail_root / 'input.jpg', original)
        if gt is not None and not (image_detail_root / 'ground_truth.png').is_file():
            write_image(image_detail_root / 'ground_truth.png', gt.astype(np.uint8) * 255)
        display_scale = min(1.0, 1400.0 / max(original.shape[:2]))
        if display_scale < 1.0:
            display_size = (round(original.shape[1] * display_scale), round(original.shape[0] * display_scale))
            display = cv2.resize(original, display_size, interpolation=cv2.INTER_AREA)
            gt_display = cv2.resize(gt.astype(np.uint8), display_size, interpolation=cv2.INTER_NEAREST).astype(bool) if gt is not None else None
        else:
            display, gt_display = original, gt
        for field, figure_name, cmap, signed, description in (
            ('absolute', 'absolute_scan_update', 'magma', False, 'Absolute scan-induced feature update RMS'),
            ('relative', 'relative_scan_update', 'magma', False, 'Relative scan-induced feature update'),
            ('memory_delta', 'memory_decay_difference', 'coolwarm', True, 'Gated decay minus ungated decay'),
        ):
            available_roles = [role for role in wanted_roles if all((s['label'], str(image_path), role) in lookup for s in specs)]
            if not available_roles:
                continue
            comparison_files = [output / f'{prefix}_{figure_name}.{suffix}' for suffix in ('png', 'pdf')]
            detail_complete = all(
                scan_influence_detail_complete(
                    output / 'details' / prefix / role / figure_name / spec['label'], gt is not None
                )
                for role in available_roles for spec in specs
            )
            if detail_complete and all(path.is_file() for path in comparison_files):
                for role in available_roles:
                    scale_key = (split, str(image_path), role, field)
                    if scale_key not in scale_lookup:
                        restored = []
                        for spec in specs:
                            record = lookup[(spec['label'], str(image_path), role)]
                            with np.load(record['npz']) as payload:
                                restored.append(restore_influence_map(payload[field], payload, original.shape[:2]))
                        finite = np.concatenate([value[np.isfinite(value)] for value in restored])
                        limit = max(float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0, 1e-12)
                        scale_lookup[scale_key] = {
                            'split': split, 'image': str(image_path), 'role': role, 'field': field,
                            'shared_vmin': -limit if signed else 0.0, 'shared_vmax': limit, 'percentile': 99,
                        }
                print(f'    SKIP rendered: {figure_name}', flush=True)
                continue
            print(f'    Rendering: {figure_name}', flush=True)
            fig, axes = plt.subplots(len(available_roles), len(specs) + 2,
                                     figsize=(2.8 * (len(specs) + 2), 3.0 * len(available_roles)), squeeze=False,
                                     constrained_layout=True)
            for row_index, role in enumerate(available_roles):
                restored_full = []
                for spec in specs:
                    record = lookup[(spec['label'], str(image_path), role)]
                    with np.load(record['npz']) as payload:
                        restored_full.append(restore_influence_map(payload[field], payload, original.shape[:2]))
                finite = np.concatenate([v[np.isfinite(v)] for v in restored_full])
                limit = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
                limit = max(limit, 1e-12)
                scale_row = {'split': split, 'image': str(image_path), 'role': role, 'field': field,
                             'shared_vmin': -limit if signed else 0.0, 'shared_vmax': limit,
                             'percentile': 99}
                scale_lookup[(split, str(image_path), role, field)] = scale_row
                detail_root = output / 'details' / prefix / role / figure_name
                detail_root.mkdir(parents=True, exist_ok=True)
                for spec, heat_full in zip(specs, restored_full):
                    save_scan_influence_detail(
                        detail_root / spec['label'], original, heat_full, limit, cmap, signed, gt
                    )
                restored = [cv2.resize(v, display.shape[1::-1], interpolation=cv2.INTER_LINEAR)
                            if v.shape != display.shape[:2] else v for v in restored_full]
                axes[row_index, 0].imshow(display[..., ::-1])
                axes[row_index, 0].set_title('Input' if row_index == 0 else '')
                if gt_display is not None:
                    gt_view = np.zeros_like(display)
                    gt_view[gt_display] = 255
                    axes[row_index, 1].imshow(gt_view[..., ::-1])
                else:
                    axes[row_index, 1].imshow(np.zeros_like(display))
                axes[row_index, 1].set_title('Ground truth' if row_index == 0 else '')
                last_im = None
                for col_index, (spec, heat) in enumerate(zip(specs, restored), start=2):
                    ax = axes[row_index, col_index]
                    ax.imshow(display[..., ::-1])
                    normalized = np.clip(np.abs(heat) / limit, 0, 1)
                    alpha = .78 * normalized
                    last_im = ax.imshow(heat, cmap=cmap, vmin=-limit if signed else 0, vmax=limit, alpha=alpha)
                    if gt_display is not None and gt_display.any():
                        contour = cv2.dilate(gt_display.astype(np.uint8), np.ones((5, 5), np.uint8))
                        ax.contour(contour.astype(float), levels=[.5], colors=['cyan'], linewidths=.45)
                    if row_index == 0:
                        ax.set_title(scan_display_name(spec['label']), fontweight='bold')
                for ax in axes[row_index]:
                    ax.axis('off')
                axes[row_index, 0].text(-.06, .5, role.replace('backbone_', ''), transform=axes[row_index, 0].transAxes,
                                        rotation=90, va='center', ha='right', fontsize=11, fontweight='bold')
                fig.colorbar(last_im, ax=axes[row_index, 2:], shrink=.7,
                             extend='both' if signed else 'max', label=description + ' (shared scale)')
            fig.suptitle(f'{description}: {image_path.name}', fontsize=14, fontweight='bold')
            for suffix in ('png', 'pdf'):
                fig.savefig(output / f'{prefix}_{figure_name}.{suffix}', dpi=300, bbox_inches='tight')
            plt.close(fig)
    write_rows(output / 'shared_scales.csv', list(scale_lookup.values()))


def render_scanning_qualitative(args, split, images, cases, specs, metrics, group_out):
    """Render the observable effect of scan methods, separately from path mechanics."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Patch, Rectangle
    from visualize_seg_results import read_image

    if not cases:
        return
    labels = {s['label'] for s in specs}
    lookup = {(r['model'], r['image']): r for r in metrics if r['model'] in labels and r['gt_available']}
    image_index = {str(image): i for i, image in enumerate(images)}
    reference = next((s['label'] for s in specs if experiment_id(s['label']) == 'K02'), specs[0]['label'])
    columns = [('Input', None), ('Ground truth', None)] + [(scan_display_name(s['label']), s['label']) for s in specs]
    nrows, ncols = len(cases), len(columns)
    fig = plt.figure(figsize=(2.7 * ncols, 3.9 * nrows + .9), facecolor='white')
    grid = fig.add_gridspec(nrows * 2, ncols, height_ratios=sum(([1.0, .68] for _ in cases), []),
                           hspace=.06, wspace=.06, left=.075, right=.99, top=.91, bottom=.10)
    case_rows = []
    for row_index, case in enumerate(cases):
        image = case['image']
        prefix = f'{image_index[str(image)]:04d}_{image.stem}'
        first_folder = args.output_dir.resolve() / specs[0]['label'] / split / prefix
        original = read_image(first_folder / 'input.png')
        gt = read_image(first_folder / 'gt_union.png')[..., 0] > 0
        roi = case['roi'] or automatic_gt_roi(gt, args.qualitative_roi_size)
        x1, y1, x2, y2 = roi
        height, width = gt.shape
        px = (int(x1 * width), int(y1 * height), max(int(x2 * width), int(x1 * width) + 1),
              max(int(y2 * height), int(y1 * height) + 1))
        gt_view = np.zeros_like(original)
        gt_view[gt] = 255
        panels = [('Input', original, None, None), ('Ground truth', gt_view, None, None)]
        reference_metrics = lookup.get((reference, str(image)))
        record = {'split': split, 'case': case['label'], 'image': str(image),
                  'roi_x1': x1, 'roi_y1': y1, 'roi_x2': x2, 'roi_y2': y2,
                  'roi_source': 'manual_json' if case['roi'] is not None else 'automatic_gt_only'}
        for spec in specs:
            label = spec['label']
            folder = args.output_dir.resolve() / label / split / prefix
            panel = read_image(folder / f'{label}_error_mask.png')
            metric = lookup.get((label, str(image)))
            panels.append((scan_display_name(label), panel, metric, label))
            if metric:
                record[label + '_iou'] = metric.get('pixel_iou')
                record[label + '_cldice'] = metric.get('cldice')
        case_rows.append(record)
        for col_index, (heading, panel, metric, model_label) in enumerate(panels):
            top_ax = fig.add_subplot(grid[row_index * 2, col_index])
            crop_ax = fig.add_subplot(grid[row_index * 2 + 1, col_index])
            top_ax.imshow(panel[..., ::-1])
            top_ax.add_patch(Rectangle((px[0], px[1]), px[2] - px[0], px[3] - px[1], fill=False,
                                       edgecolor='#0877d1', linewidth=1.6, linestyle=(0, (4, 2))))
            crop_ax.imshow(panel[px[1]:px[3], px[0]:px[2], ::-1])
            for ax in (top_ax, crop_ax):
                ax.axis('off')
            if row_index == 0:
                top_ax.set_title(heading, fontsize=12, fontweight='bold', pad=8)
            if col_index == 0:
                top_ax.text(-.08, .5, case['label'], transform=top_ax.transAxes, rotation=90,
                            va='center', ha='right', fontsize=11, fontweight='bold')
            if metric:
                delta = ''
                if reference_metrics and model_label != reference:
                    delta = f"  ΔIoU {100 * (metric['pixel_iou'] - reference_metrics['pixel_iou']):+.1f} pp"
                crop_ax.text(.5, -.08, f"IoU {metric['pixel_iou']:.3f}  clDice {metric['cldice']:.3f}{delta}",
                             transform=crop_ax.transAxes, ha='center', va='top', fontsize=8)
            fig.add_artist(ConnectionPatch(xyA=(px[0], px[3]), coordsA=top_ax.transData,
                                           xyB=(0, 1), coordsB=crop_ax.transAxes,
                                           color='#0877d1', linewidth=.65, alpha=.7))
            fig.add_artist(ConnectionPatch(xyA=(px[2], px[3]), coordsA=top_ax.transData,
                                           xyB=(1, 1), coordsB=crop_ax.transAxes,
                                           color='#0877d1', linewidth=.65, alpha=.7))
    fig.suptitle('(A) Qualitative scanning ablation: effect on segmentation', fontsize=15, fontweight='bold', y=.975)
    fig.legend(handles=[Patch(color='#00ffff', label='Correct (TP)'), Patch(color='#ff7471', label='Missed (FN)'),
                        Patch(color='#b074ed', label='Extra (FP)')],
               loc='lower center', ncol=3, frameon=False, fontsize=10)
    output = group_out / 'scanning_qualitative_effect'
    for suffix in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + suffix), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    write_rows(group_out / 'scanning_qualitative_cases.csv', case_rows)


def load_core(spec, device):
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel
    from visualize_model_features import safe_torch_load, extract_state_dict, key_variants
    if not spec.get('yaml'):
        # Architecture comes from this checkpoint, not today's possibly changed YAML.
        core = YOLO(spec['weights'], task='segment').model
    else:
        checkpoint = safe_torch_load(Path(spec['weights']))
        stored = (checkpoint.get('ema') or checkpoint.get('model')) if isinstance(checkpoint, dict) else checkpoint
        nc = len(stored.names) if hasattr(stored, 'names') else None
        core = SegmentationModel(spec['yaml'], ch=3, nc=nc, verbose=False)
        target = core.state_dict()
        candidates = key_variants(extract_state_dict(checkpoint))
        source = next((s for s in candidates if set(s) == set(target) and
                       all(s[k].shape == target[k].shape for k in target)), None)
        if source is None:
            raise ValueError('YAML/checkpoint mismatch: refusing partial/randomly initialized model')
        core.load_state_dict(source, strict=True)
    if 'segment' not in type(core.model[-1]).__name__.lower():
        raise ValueError('Expected a segmentation checkpoint')
    return core.float().to(device).eval()


def figure_grid(rows, titles, path, roi=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), len(titles), figsize=(3*len(titles), 3*len(rows)), squeeze=False)
    for row, axes_row in zip(rows, axes):
        for (label, image), ax in zip(row, axes_row):
            h, w = image.shape[:2]
            if roi:
                x1, y1, x2, y2 = roi
                image = image[int(y1*h):max(int(y2*h), int(y1*h)+1), int(x1*w):max(int(x2*w), int(x1*w)+1)]
            ax.imshow(image[..., ::-1])
            ax.set_title(label, fontsize=10)
            ax.axis('off')
    fig.tight_layout(pad=.5)
    for ext in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + ext), dpi=220)
    plt.close(fig)


def structure_figures(state_name, state, memory, canvas, gt, valid_image, output, max_draw):
    import matplotlib.pyplot as plt
    from PIL import Image
    p = state.last_guidance[0, 0].detach().float().cpu().numpy()
    indices = state.last_path_indices[0].cpu().numpy() if state.last_path_indices is not None else None
    if indices is None:
        return []
    valid = state.last_path_mask[0].cpu().numpy().astype(bool)
    height, width = p.shape
    np.savez_compressed(output.with_suffix('.npz'), probability=p, indices=indices, valid=valid, **memory)
    coverage = scatter_path(np.ones_like(indices), indices, valid, p.shape)[1]
    # Pixel-space masks evaluate coverage without dropping thin GT through nearest downsampling.
    covered_pixels = np.asarray(Image.fromarray(coverage.astype(np.uint8)).resize(canvas.shape[1::-1], resample=Image.Resampling.NEAREST)).astype(bool) & valid_image
    row = {'module': state_name, 'valid_token_ratio': float(valid.mean()),
           'unique_feature_coverage': float(coverage.mean()),
           'valid_image_coverage': float(covered_pixels.sum()/max(valid_image.sum(), 1)),
           'path_gt_precision': float(gt[covered_pixels].mean()) if gt is not None and covered_pixels.any() else float('nan'),
           'gt_coverage_recall': float(covered_pixels[gt & valid_image].mean()) if gt is not None and gt.any() else float('nan')}
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()
    im = axes[0].imshow(p, vmin=0, vmax=1, cmap='viridis')
    axes[0].set_title('Crack guidance p (fixed 0-1)')
    fig.colorbar(im, ax=axes[0], shrink=.65)
    axes[1].imshow(canvas[..., ::-1], extent=(0, width, height, 0))
    for n in range(min(len(indices), max_draw)):
        ix = indices[n]
        x, y = ix % width + .5, ix // width + .5
        axes[1].scatter(x[valid[n]], y[valid[n]], s=4)
        for t in range(1, len(ix)):
            if valid[n, t-1] and valid[n, t]:
                axes[1].plot(x[t-1:t+1], y[t-1:t+1], lw=.8)
    axes[1].set_title(f'Paths: first {min(len(indices),max_draw)}/{len(indices)} drawn')
    keys = [('retention_bias', -.25, .25, 'coolwarm'), ('transition_bias', 0, .5, 'viridis'),
            ('write_gain', .75, 1.25, 'coolwarm'), ('decay', 0, 1, 'viridis'),
            ('decay_without_gates', 0, 1, 'viridis')]
    for ax, (key, low, high, cmap) in zip(axes[2:7], keys):
        mapped, _ = scatter_path(memory[key], indices, valid, p.shape)
        im = ax.imshow(mapped, vmin=low, vmax=high, cmap=cmap)
        ax.set_title(key + ' (forward)')
        fig.colorbar(im, ax=ax, shrink=.65, extend='both')
        selected = memory[key][valid]
        row[key + '_mean'] = float(selected.mean()) if selected.size else float('nan')
    axes[7].imshow(coverage, vmin=0, vmax=1, cmap='gray')
    axes[7].set_title(f'ALL-path coverage: {coverage.mean():.1%}')
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(state_name + '; white memory pixels = not visited; gates are pre-softplus biases / B multiplier', fontsize=10)
    for ext in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.' + ext), dpi=220)
    plt.close(fig)
    return [row]


def paired_plot(metrics, specs, out):
    """Show the complete selected-image distribution, including failures."""
    import matplotlib.pyplot as plt
    reference = specs[0]['label']
    lookup = {(r['model'], r['image']): r for r in metrics if r['gt_available']}
    if len(specs) < 2 or not lookup:
        return
    fig, axes = plt.subplots(1, 2, figsize=(max(10, min(24, len(specs))), 6), constrained_layout=True)
    for ax, key in zip(axes, ('pixel_iou', 'cldice')):
        for j, spec in enumerate(specs[1:]):
            pairs = [(v, lookup.get((reference, image))) for (model, image), v in lookup.items() if model == spec['label']]
            delta = np.array([100*(a[key]-b[key]) for a,b in pairs if b is not None])
            if delta.size:
                ax.scatter(j+np.linspace(-.12,.12,len(delta)), delta, s=14, alpha=.65)
                ax.plot([j-.2,j+.2], [delta.mean()]*2, color='black', lw=2)
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.set_xticks(range(len(specs)-1))
        ax.set_xticklabels([s['label'] for s in specs[1:]], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'{key}: paired difference vs {reference} (pp)')
        ax.set_title('Each dot = one selected image; black line = mean')
    for ext in ('png', 'pdf'):
        fig.savefig(out / ('paired_pixel_metrics.'+ext), dpi=220)
    plt.close(fig)


def run_split(args, specs, split):
    if args.imgsz <= 0 or args.imgsz % 32 or args.max_images < 0 or args.max_paths_draw < 0:
        raise ValueError('imgsz must be a positive multiple of 32; counts must be nonnegative')
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1 or args.max_det < 1:
        raise ValueError('Invalid prediction thresholds/max-det')
    roi = validate_normalized_roi(args.roi.split(',')) if args.roi else None
    if not 0 < args.qualitative_roi_size <= 1:
        raise ValueError('--qualitative-roi-size must be in (0, 1]')
    if not 0 < args.scan_influence_top_fraction <= 1 or args.scan_influence_gt_dilate < 0:
        raise ValueError('--scan-influence-top-fraction must be in (0,1]; dilation must be nonnegative')
    if args.scan_influence and not [r.strip() for r in args.scan_influence_roles.split(',') if r.strip()]:
        raise ValueError('--scan-influence-roles must contain at least one semantic role')
    if args.qualitative_only and not (args.qualitative_scanning or args.scan_influence):
        raise ValueError('--qualitative-only requires --qualitative-scanning and/or --scan-influence')
    if args.qualitative_only and any(experiment_id(s['label']) not in SCAN_DISPLAY_NAMES for s in specs):
        raise ValueError('--qualitative-only accepts scanning ablations K01-K06 only')
    from visualize_seg_results import (GroundTruth, prepare_device_environment, resolve_source, read_image, write_image,
                                       load_yolo_segmentation_gt, instance_visualization, union_metrics)
    prepare_device_environment(args.device)
    import torch
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    from ultralytics.utils import ops
    from compare_model_features import (preprocess, letterbox_mask, resolve_layer_names, capture_hook,
                                        reduce_feature, robust_normalize, heat_overlay, focus_statistics)
    device = torch.device('cpu' if args.device == 'cpu' else ('cuda:' + args.device if args.device.isdigit() else args.device))
    images, root, _ = resolve_source(Path(args.source).resolve(), split, True)
    images = sorted(set(images))
    if args.qualitative_scanning and args.qualitative_cases:
        qualitative_cases = resolve_qualitative_cases(args.qualitative_cases, images)
        images = [case['image'] for case in qualitative_cases]
    else:
        if args.max_images and len(images) > args.max_images:
            images = sorted(random.Random(args.sample_seed).sample(images, args.max_images))
        qualitative_cases = resolve_qualitative_cases(None, images) if args.qualitative_scanning else []
    if not images:
        raise ValueError('No source images')
    gt_index = build_gt_index(args.gt_dir)
    out = args.output_dir.resolve() / 'comparisons' / split
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'split': split, 'models': specs, 'images': [{'path': str(p), 'sha256': fingerprint(p)} for p in images],
                'precision': 'FP32 unfused',
                'feature_normalization': 'feature panels: per-image/model; scan influence: shared pooled 99th percentile by image+role',
                'metrics': 'class-agnostic per-image union metrics at fixed confidence; NOT instance AP',
                'mask_postprocess': 'repository process_mask on square letterbox; nearest resize after unpadding',
                'torch_version': torch.__version__, 'device': str(device)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    metrics, focus, routing, influence = [], [], [], []
    for spec in specs:
        model_out = args.output_dir.resolve() / spec['label'] / split
        model_out.mkdir(parents=True, exist_ok=True)
        old_manifest_matches = legacy_manifest_matches(model_out / 'manifest.json', args, split)
        previous = {
            'metrics': read_rows(model_out / 'per_image_metrics.csv'),
            'focus': read_rows(model_out / 'feature_focus.csv'),
            'routing': read_rows(model_out / 'routing_memory.csv'),
            'influence': read_rows(model_out / 'scan_influence.csv'),
        }
        (model_out / 'manifest.json').write_text(json.dumps({**manifest, 'models': [spec]}, indent=2), encoding='utf-8')
        print(f"Loading {spec['label']}: {spec['weights']}", flush=True)
        core = load_core(spec, device)
        if args.qualitative_only and not args.scan_influence:
            plan = {'features': {}, 'mechanisms': [], 'mechanism_roles': {}, 'mechanism_status': 'skipped',
                    'selection': 'Skipped by --qualitative-only'}
            states = {}
        else:
            plan, states = discover_layers(core, spec, args.imgsz, device)
        spec['resolved_plan'] = plan
        (model_out / 'layer_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
        print(f"  Features: {plan['features']}; mechanisms: {plan['mechanisms'] or plan['mechanism_status']}", flush=True)
        layers = [] if args.qualitative_only else list(dict.fromkeys(plan['features'].values()))
        modules = dict(core.named_modules())
        for state in states.values():
            state.export_visual_debug = bool(args.scan_influence)
        for image_index, image_path in enumerate(images):
            prefix = f'{image_index:04d}_{image_path.stem}'
            folder = model_out / prefix
            folder.mkdir(exist_ok=True)
            signature = completion_signature(args, spec, split, image_path, plan)
            cached = load_completion(folder / 'complete.json', signature)
            if cached is not None and legacy_completion(folder, spec, image_path, plan, states, args, cached) is None:
                cached = None
            if cached is None and old_manifest_matches:
                cached = legacy_completion(folder, spec, image_path, plan, states, args, previous)
                if cached is not None:
                    (folder / 'complete.json').write_text(
                        json.dumps({'signature': signature, 'rows': cached}, indent=2,
                                   default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
                        encoding='utf-8',
                    )
            if cached is not None:
                metrics.extend(cached.get('metrics', []))
                focus.extend(cached.get('focus', []))
                routing.extend(cached.get('routing', []))
                influence.extend(cached.get('influence', []))
                print(f"  SKIP complete: {spec['label']} / {split} {image_index+1}/{len(images)} "
                      f"{image_path.name}", flush=True)
                continue
            row_starts = (len(metrics), len(focus), len(routing), len(influence))
            image = read_image(image_path)
            h, w = image.shape[:2]
            canvas, tensor, valid_pixels, meta = preprocess(image, args.imgsz, device, torch)
            if gt_index is not None:
                gt_union, gt_path = load_external_gt(image_path, gt_index, (h, w), args.gt_invert)
                masks = gt_union[None] if gt_union.any() else np.zeros((0, h, w), dtype=bool)
                gt = GroundTruth(masks, np.zeros(len(masks), dtype=np.int64), gt_path)
                known_gt = True
            else:
                gt = load_yolo_segmentation_gt(image_path, h, w, root)
                known_gt = gt.label_path is not None
                gt_union = gt.masks.any(0) if known_gt else None
            if known_gt and gt_index is None:
                lines = [s.split() for s in gt.label_path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
                if gt.malformed_lines or any(len(s) < 7 for s in lines):
                    raise ValueError(f'Require true, valid segmentation polygons: {gt.label_path}')
            gt_square = letterbox_mask(gt_union, args.imgsz, meta) if known_gt else None
            features, memory, handles = {}, {}, []
            for name in layers:
                handles.append(modules[name].register_forward_hook(capture_hook(features, name)))
            for name, state in states.items():
                def prehook(m, inputs, name=name):
                    memory[name] = memory_diagnostics(m, inputs)
                handles.append(state.path_ssm.register_forward_pre_hook(prehook))
            try:
                with torch.inference_mode():
                    pred = core(tensor)
                    nms = ops.non_max_suppression(pred[0], args.conf, args.iou, nc=len(core.names), max_det=args.max_det)[0]
                    proto = pred[1][-1] if isinstance(pred[1], tuple) else pred[1]
                    masks = ops.process_mask(proto[0], nms[:, 6:], nms[:, :4], tensor.shape[2:], upsample=True) if len(nms) else None
            finally:
                for handle in handles:
                    handle.remove()
            ratio, left, top, rw, rh = meta
            pm = np.stack([cv2.resize(m[top:top+rh, left:left+rw].cpu().numpy().astype(np.uint8), (w, h),
                                     interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]) if masks is not None else np.zeros((0,h,w), bool)
            boxes = nms[:, :4].detach().cpu().numpy().copy()
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / ratio).clip(0, w)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / ratio).clip(0, h)
            classes, conf = nms[:, 5].cpu().numpy().astype(int), nms[:, 4].cpu().numpy()
            vis = instance_visualization(image, pm, classes, conf, boxes, core.names, .45, 2)
            write_image(folder / f"{spec['label']}_prediction.png", vis)
            write_image(folder / 'input.png', image)
            error, gt_view = image.copy(), image.copy()
            if known_gt:
                pu = pm.any(0)
                for active, color in [(pu & gt_union, (0,210,0)), (pu & ~gt_union, (0,0,255)), (~pu & gt_union, (255,0,0))]:
                    error[active] = color
                gt_view[gt_union] = (.5*gt_view[gt_union] + .5*np.array([0,255,255])).astype(np.uint8)
                write_image(folder / 'gt_union.png', gt_union.astype(np.uint8) * 255)
                write_image(folder / f'{spec["label"]}_pred_union.png', pu.astype(np.uint8) * 255)
                write_image(folder / f'{spec["label"]}_error_mask.png', mask_error_image(pu, gt_union))
            write_image(folder / f"{spec['label']}_error.png", error)
            write_image(folder / 'gt.png', gt_view)
            metrics.append({'model': spec['label'], 'split': split, 'image': str(image_path), 'gt_available': known_gt,
                            'gt_label': str(gt.label_path) if known_gt else '',
                            'gt_format': 'raster_mask' if gt_index is not None else 'yolo_polygon',
                            'gt_alignment': 'direct_no_rotation' if gt_index is not None else 'yolo_polygon',
                            **(union_metrics(pm, gt.masks) if known_gt else {})})
            for role, name in requested_feature_items(plan, args.qualitative_only):
                if name not in features:
                    raise ValueError(f'Layer does not output a spatial tensor: {name}')
                heat = reduce_feature(features[name], 'l2')
                np.save(folder / f"{spec['label']}_{role}.npy", heat)
                norm = robust_normalize(heat, 99.)
                overlay = heat_overlay(canvas, norm, .45)[top:top+rh, left:left+rw]
                write_image(folder / f"{spec['label']}_{role}.png", cv2.resize(overlay, (w,h)))
                focus.append({'model': spec['label'], 'split': split, 'image': str(image_path), 'layer': name, 'role': role,
                              **focus_statistics(heat, gt_square, valid_pixels)})
            for name, state in states.items():
                if name in memory:
                    role = semantic_role(name, plan)
                    if not args.qualitative_only:
                        rows = structure_figures(name, state, memory[name], canvas, gt_square, valid_pixels,
                                                 folder / (spec['label']+'_'+name.replace('.', '_')+'_structure'), args.max_paths_draw)
                        routing.extend({'model': spec['label'], 'split': split, 'image': str(image_path),
                                        'semantic_role': role, **r} for r in rows)
                    if args.scan_influence:
                        rows = scan_influence_diagnostics(
                            name, state, memory[name], gt_square, valid_pixels, meta, (h, w),
                            folder / (spec['label']+'_'+name.replace('.', '_')+'_scan_influence'),
                            args.scan_influence_top_fraction, args.scan_influence_gt_dilate,
                        )
                        influence.extend({'model': spec['label'], 'split': split, 'image': str(image_path),
                                          'semantic_role': role, **r} for r in rows)
            completed_rows = {
                'metrics': metrics[row_starts[0]:],
                'focus': focus[row_starts[1]:],
                'routing': routing[row_starts[2]:],
                'influence': influence[row_starts[3]:],
            }
            (folder / 'complete.json').write_text(
                json.dumps({'signature': signature, 'rows': completed_rows}, indent=2,
                           default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
                encoding='utf-8',
            )
            print(f"  {spec['label']} / {split} {image_index+1}/{len(images)} {image_path.name}", flush=True)
            del pred, tensor, nms, masks, features, memory
        del core, modules, states
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        for name, rows in [('per_image_metrics', metrics), ('feature_focus', focus), ('routing_memory', routing),
                           ('scan_influence', influence)]:
            write_rows(model_out / (name+'.csv'), [r for r in rows if r['model'] == spec['label']])
    for name, rows in [('per_image_metrics', metrics), ('feature_focus', focus), ('routing_memory', routing),
                       ('scan_influence', influence)]:
        write_rows(out / (name+'.csv'), rows)
    groups = comparison_groups(specs)
    (out/'layer_plans.json').write_text(json.dumps({s['label']:s['resolved_plan'] for s in specs}, indent=2), encoding='utf-8')
    for group, subset, role_prefix in groups:
        group_out = out/group
        group_out.mkdir(exist_ok=True)
        group_metrics = [r for r in metrics if r['model'] in {s['label'] for s in subset}]
        write_rows(group_out/'per_image_metrics.csv', group_metrics)
        paired_plot(group_metrics, subset, group_out)
        if args.qualitative_scanning and group == 'scanning':
            if not all(r['gt_available'] for r in group_metrics):
                raise ValueError('--qualitative-scanning requires GT for every selected image/model')
            render_scanning_qualitative(args, split, images, qualitative_cases, subset, group_metrics, group_out)
        if args.scan_influence and group == 'scanning':
            group_influence = [r for r in influence if r['model'] in {s['label'] for s in subset}]
            write_rows(group_out / 'scan_influence.csv', group_influence)
            render_scan_influence_comparison(args, split, images, subset, group_influence, group_out)
        if args.qualitative_only:
            continue
        roles = list(dict.fromkeys(k for s in subset for k in s['resolved_plan']['features']
                                   if k.startswith(role_prefix) or k.startswith('manual_')))
        if not roles:
            raise ValueError(f'No requested semantic feature roles found for group {group}; use --layers manually')
        (group_out/'comparison_plan.json').write_text(json.dumps({'models':[s['label'] for s in subset],
                                                                'roles':roles, 'reference':subset[0]['label']},indent=2), encoding='utf-8')
        for i, image in enumerate(images):
            prefix = f'{i:04d}_{image.stem}'
            comparison_folder = group_out/prefix
            comparison_folder.mkdir(exist_ok=True)
            rows = []
            for s in subset:
                label = s['label']
                folder = args.output_dir.resolve()/label/split/prefix
                has_gt = next(r['gt_available'] for r in group_metrics if r['model']==label and r['image']==str(image))
                original = read_image(folder/'input.png')
                row = [('Input',original), ('GT' if has_gt else 'GT unavailable',read_image(folder/'gt.png')),
                       (label+' prediction',read_image(folder/(label+'_prediction.png'))),
                       ('TP green / FP red / FN blue' if has_gt else 'No GT: error unavailable',read_image(folder/(label+'_error.png')))]
                for role in roles:
                    name = s['resolved_plan']['features'].get(role)
                    row.append((role+' '+name+' RMS' if name else role+' unavailable',
                                read_image(folder/f'{label}_{role}.png') if name else np.full_like(original,235)))
                rows.append(row)
            for start in range(0,len(rows),6):
                page = rows[start:start+6]
                suffix = f'_{start//6+1:02d}' if len(rows)>6 else ''
                figure_grid(page,list(range(len(page[0]))),comparison_folder/('comparison'+suffix))
                if roi:
                    figure_grid(page,list(range(len(page[0]))),comparison_folder/('comparison_roi'+suffix),roi)
    print(f'Done: {out}')


def main():
    args = parse_args()
    specs = resolve_specs(args)
    if args.gt_dir is not None and args.split == 'all':
        raise ValueError('--gt-dir with a flat image directory uses one logical split; do not combine it with --split all')
    splits = resolve_splits(Path(args.source).resolve(), args.split)
    out = args.output_dir.resolve()
    resuming = out.exists() and any(out.iterdir())
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_plan.json').write_text(json.dumps({'models': specs, 'splits': splits,
                                                'max_images_per_split': args.max_images}, indent=2), encoding='utf-8')
    print(f'{len(specs)} models; splits: {", ".join(splits)}; max images per split: {args.max_images or "ALL"}', flush=True)
    if resuming:
        print(f'Resume mode: reuse matching complete.json entries in {out}; incomplete/mismatched items rerun.', flush=True)
    for split in splits:
        run_split(args, specs, split)


if __name__ == '__main__':
    main()
