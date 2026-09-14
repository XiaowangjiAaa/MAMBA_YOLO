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
    p.add_argument('--manifest', type=Path, help='JSON list of {label,weights,yaml(optional),layers(optional)}; overrides runs')
    p.add_argument('--source', required=True, help='Image / directory / TXT / dataset YAML')
    p.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='val', help='all = every declared train/val/test split in --source YAML')
    p.add_argument('--layers', '--feature-layers', dest='layers', default='auto', help='auto = discover backbone/neck stages; or comma-separated layer indices/names')
    p.add_argument('--mechanism-layers', default='auto', help='auto = all actual CPSB states; none = disabled; or parent layer indices/names')
    p.add_argument('--max-images', type=int, default=6, help='0 = all images; random selection independent of predictions')
    p.add_argument('--sample-seed', type=int, default=1)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    p.add_argument('--conf', type=float, default=.25)
    p.add_argument('--iou', type=float, default=.7, help='NMS IoU, not mask matching threshold')
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--max-paths-draw', type=int, default=32, help='Only limits drawing, NOT coverage statistics')
    p.add_argument('--roi', help='Optional fixed normalized x1,y1,x2,y2 crop for ALL models/images')
    p.add_argument('--output-dir', type=Path, default=Path('paper_visualizations/9.11'))
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
    if spec['layers'].lower() == 'auto':
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
        features = select_stage_layers(core, shapes)
    else:
        features = {f'manual_{i}': n for i,n in enumerate(resolve_layer_names(core, spec['layers']))}
    policy = spec['mechanism_layers'].lower()
    parents = None if policy in ('auto','none') else resolve_layer_names(core, spec['mechanism_layers'])
    states = {n:m for n,m in modules.items() if hasattr(m,'path_ssm') and hasattr(m,'last_path_indices')
              and policy != 'none' and (parents is None or any(n == p or n.startswith(p+'.') for p in parents))}
    if parents and not states:
        raise ValueError('Selected mechanism layers contain no CPSB path state')
    plan = {'features': features, 'mechanisms': list(states),
            'mechanism_status': 'found' if states else ('disabled' if policy=='none' else 'no CPSB in model'),
            'selection': 'Observed spatial strides + checkpoint backbone boundary + segmentation head inputs'}
    return plan, states


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


def write_rows(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


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
    roi = tuple(map(float, args.roi.split(','))) if args.roi else None
    if roi and (len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1)):
        raise ValueError('ROI must satisfy 0<=x1<x2<=1, 0<=y1<y2<=1')
    from visualize_seg_results import (prepare_device_environment, resolve_source, read_image, write_image,
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
    if args.max_images and len(images) > args.max_images:
        images = sorted(random.Random(args.sample_seed).sample(images, args.max_images))
    if not images:
        raise ValueError('No source images')
    out = args.output_dir.resolve() / 'comparisons' / split
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'split': split, 'models': specs, 'images': [{'path': str(p), 'sha256': fingerprint(p)} for p in images],
                'precision': 'FP32 unfused', 'feature_normalization': 'per-image per-model 1st-99th percentile',
                'metrics': 'class-agnostic per-image union metrics at fixed confidence; NOT instance AP',
                'mask_postprocess': 'repository process_mask on square letterbox; nearest resize after unpadding',
                'torch_version': torch.__version__, 'device': str(device)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    metrics, focus, routing = [], [], []
    for spec in specs:
        model_out = args.output_dir.resolve() / spec['label'] / split
        model_out.mkdir(parents=True, exist_ok=True)
        (model_out / 'manifest.json').write_text(json.dumps({**manifest, 'models': [spec]}, indent=2), encoding='utf-8')
        print(f"Loading {spec['label']}: {spec['weights']}", flush=True)
        core = load_core(spec, device)
        plan, states = discover_layers(core, spec, args.imgsz, device)
        spec['resolved_plan'] = plan
        (model_out / 'layer_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
        print(f"  Features: {plan['features']}; mechanisms: {plan['mechanisms'] or plan['mechanism_status']}", flush=True)
        layers = list(dict.fromkeys(plan['features'].values()))
        modules = dict(core.named_modules())
        for image_index, image_path in enumerate(images):
            prefix = f'{image_index:04d}_{image_path.stem}'
            folder = model_out / prefix
            folder.mkdir(exist_ok=True)
            image = read_image(image_path)
            h, w = image.shape[:2]
            canvas, tensor, valid_pixels, meta = preprocess(image, args.imgsz, device, torch)
            gt = load_yolo_segmentation_gt(image_path, h, w, root)
            known_gt = gt.label_path is not None
            if known_gt:
                lines = [s.split() for s in gt.label_path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
                if gt.malformed_lines or any(len(s) < 7 for s in lines):
                    raise ValueError(f'Require true, valid segmentation polygons: {gt.label_path}')
            gt_union = gt.masks.any(0) if known_gt else None
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
            write_image(folder / f"{spec['label']}_error.png", error)
            write_image(folder / 'gt.png', gt_view)
            metrics.append({'model': spec['label'], 'split': split, 'image': str(image_path), 'gt_available': known_gt,
                            'gt_label': str(gt.label_path) if known_gt else '',
                            **(union_metrics(pm, gt.masks) if known_gt else {})})
            for role, name in plan['features'].items():
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
                    rows = structure_figures(name, state, memory[name], canvas, gt_square, valid_pixels,
                                             folder / (spec['label']+'_'+name.replace('.', '_')+'_structure'), args.max_paths_draw)
                    routing.extend({'model': spec['label'], 'split': split, 'image': str(image_path), **r} for r in rows)
            print(f"  {spec['label']} / {split} {image_index+1}/{len(images)} {image_path.name}", flush=True)
            del pred, tensor, nms, masks, features, memory
        del core, modules, states
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        for name, rows in [('per_image_metrics', metrics), ('feature_focus', focus), ('routing_memory', routing)]:
            write_rows(model_out / (name+'.csv'), [r for r in rows if r['model'] == spec['label']])
    for name, rows in [('per_image_metrics', metrics), ('feature_focus', focus), ('routing_memory', routing)]:
        write_rows(out / (name+'.csv'), rows)
    groups = comparison_groups(specs)
    (out/'layer_plans.json').write_text(json.dumps({s['label']:s['resolved_plan'] for s in specs}, indent=2), encoding='utf-8')
    for group, subset, role_prefix in groups:
        group_out = out/group
        group_out.mkdir(exist_ok=True)
        group_metrics = [r for r in metrics if r['model'] in {s['label'] for s in subset}]
        write_rows(group_out/'per_image_metrics.csv', group_metrics)
        paired_plot(group_metrics, subset, group_out)
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
    splits = resolve_splits(Path(args.source).resolve(), args.split)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Choose a new/empty output directory to avoid mixing runs: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'run_plan.json').write_text(json.dumps({'models': specs, 'splits': splits,
                                                'max_images_per_split': args.max_images}, indent=2), encoding='utf-8')
    print(f'{len(specs)} models; splits: {", ".join(splits)}; max images per split: {args.max_images or "ALL"}', flush=True)
    for split in splits:
        run_split(args, specs, split)


if __name__ == '__main__':
    main()
