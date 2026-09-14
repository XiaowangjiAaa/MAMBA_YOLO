"""Audit 9.11 ablations without loading torch or modifying training outputs.

Reported checkpoint evaluation and training-CSV peaks are deliberately separate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import yaml


LABELS = {
    'K00': 'YOLO11n baseline', 'K01': 'Adaptive / full memory / backbone P2-P5',
    'K02': 'Raster', 'K03': 'Cross', 'K04': 'Serpentine', 'K05': 'Diagonal',
    'K06': 'Fixed straight', 'K10': 'Standard memory', 'K11': 'Retention only',
    'K12': 'Retention + transition', 'K13': 'Retention + write',
    'K20': 'Backbone P3', 'K21': 'Backbone P4', 'K22': 'Backbone P3+P4',
    'K23': 'Backbone P2+P3', 'K24': 'Backbone P4+P5', 'K26': 'Neck only',
    'K27': 'Backbone + neck', 'K28': 'YOLOv9c baseline', 'K29': 'YOLOv9c + CPSB',
    'K30': 'YOLOv5n baseline', 'K31': 'YOLOv5n + CPSB',
    'K32': 'YOLOv8n baseline', 'K33': 'YOLOv8n + CPSB',
    'K36': 'YOLO26n compat baseline', 'K37': 'YOLO26n compat + CPSB',
    'K38': 'YOLO12n baseline', 'K39': 'YOLO12n + CPSB',
}
GROUPS = {
    'Scanning': ['K02', 'K03', 'K04', 'K05', 'K06', 'K01'],
    'Memory': ['K10', 'K11', 'K12', 'K13', 'K01'],
    'Placement': ['K00', 'K20', 'K21', 'K22', 'K23', 'K24', 'K01', 'K26', 'K27'],
}
PAIRS = [('YOLOv5n', 'K30', 'K31'), ('YOLOv8n', 'K32', 'K33'),
         ('YOLOv9c', 'K28', 'K29'), ('YOLO11n', 'K00', 'K01'),
         ('YOLO12n', 'K38', 'K39'), ('YOLO26n compatibility', 'K36', 'K37')]


def read_curve(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return [{k.strip(): float(v) for k, v in row.items() if k and v.strip()}
                for row in csv.DictReader(f)]


def parse_report(path):
    text = path.read_text(encoding='utf-8-sig')
    out = {}
    for label, key in [('mAP@0.5', 'mask_map50'), ('mAP@0.75', 'mask_map75'),
                       ('mAP@0.5:0.95', 'mask_map5095'), ('Precision', 'mask_precision'),
                       ('Recall', 'mask_recall'), ('mIoU (union)', 'union_iou'), ('clDice', 'cldice')]:
        match = re.search(r'^' + re.escape(label) + r'\s+(?:[\d.]+|-)\s+([\d.]+)\s*$', text, re.M)
        if not match:
            raise ValueError(f'Missing {label} in {path}')
        out[key] = float(match[1])
    for label, key in [('Images evaluated', 'images'), ('Total instances', 'instances')]:
        out[key] = int(re.search(re.escape(label) + r':\s*(\d+)', text)[1])
    out['params'] = int(re.search(r'Parameters:\s*[\d.]+ M \(([\d,]+)\)', text)[1].replace(',', ''))
    out['gflops_reported'] = float(re.search(r'GFLOPs:\s*([\d.]+)', text)[1])
    out['fps_reported'] = float(re.search(r'FPS:\s*([\d.]+)', text)[1])
    out['report_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def collect(root, seed):
    records, protocols = [], {}
    for run in sorted(root.glob(f'K*_crack-seg_seed{seed}')):
        eid = run.name.split('_')[0]
        args = yaml.safe_load((run / 'args.yaml').read_text(encoding='utf-8-sig'))
        if args['seed'] != seed:
            raise ValueError(f'Seed mismatch: {run}')
        curve = read_curve(run / 'results.csv')
        valid = [r for r in curve if math.isfinite(r.get('metrics/mAP50(M)', math.nan))]
        if not valid:
            raise ValueError(f'No finite mask mAP50: {run}')
        peak = max(valid, key=lambda r: r['metrics/mAP50(M)'])
        report = run / 'segmentation_eval_report_best_map50.txt'
        row = {'id': eid, 'description': LABELS.get(eid, eid), 'seed': seed,
               'run': run.as_posix(), 'report': report.as_posix(),
               'completed_marker': (run / '.batch_completed.json').is_file(),
               'epochs_recorded': len(curve), 'last_epoch': int(curve[-1]['epoch']),
               'csv_peak_epoch': int(peak['epoch']), 'csv_peak_mask_map50': peak['metrics/mAP50(M)'],
               'csv_peak_row_json': json.dumps(peak, sort_keys=True), 'model_yaml_recorded': args['model']}
        if report.is_file():
            row.update(parse_report(report))
            row['report_minus_csv_pp'] = 100 * (row['mask_map50'] - row['csv_peak_mask_map50'])
        else:
            row['status'] = 'missing best_map50 report; no fallback'
        records.append(row)
        protocols[eid] = args
    if not records:
        raise ValueError(f'No seed{seed} runs under {root}')
    return records, protocols


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def tables(records):
    rows = {r['id']: r for r in records}
    lines = ['# 9.11 ablation tables', '',
             'Source: segmentation_eval_report_best_map50.txt; rates in percent; seed=1 unless CLI changed.', '',
             'Missing values are not zero. These are archived reports, not newly rerun evaluation.', '']
    for title, ids in GROUPS.items():
        lines += [f'## {title}', '', '| ID | Method | Mask AP50 | AP50-95 | Recall | union IoU | clDice | Params M |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for eid in ids:
            r = rows.get(eid, {})
            values = [f'{100*r[k]:.2f}' if k in r else '—' for k in
                      ('mask_map50', 'mask_map5095', 'mask_recall', 'union_iou', 'cldice')]
            params = f"{r['params']/1e6:.3f}" if 'params' in r else '—'
            lines.append('| ' + ' | '.join([eid, LABELS[eid]] + values + [params]) + ' |')
        lines.append('')
    lines += ['## Family', '', '| Family | Pair | Baseline Mask AP50 | CPSB Mask AP50 | Delta pp |', '|---|---|---:|---:|---:|']
    for family, a, b in PAIRS:
        x, y = rows.get(a, {}).get('mask_map50'), rows.get(b, {}).get('mask_map50')
        lines.append(f'| {family} | {a}/{b} | ' + (f'{100*x:.2f}' if x is not None else '—') +
                     ' | ' + (f'{100*y:.2f}' if y is not None else '—') + ' | ' +
                     (f'{100*(y-x):+.2f}' if x is not None and y is not None else '—') + ' |')
    lines += ['', '## Checkpoint audit', '', '| ID | CSV peak epoch | CSV peak AP50 | Report AP50 | Report - CSV pp | Last epoch |',
              '|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['id']} | {r['csv_peak_epoch']} | {100*r['csv_peak_mask_map50']:.3f} | " +
                     (f"{100*r['mask_map50']:.2f} | {r['report_minus_csv_pp']:+.3f}" if 'mask_map50' in r else '— | —') +
                     f" | {r['last_epoch']} |")
    return '\n'.join(lines) + '\n'


def plot_results(records, root, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup = {r['id']: r for r in records}
    for title, ids in GROUPS.items():
        ids = [i for i in ids if 'mask_map50' in lookup.get(i, {})]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        vals = [lookup[i]['mask_map50']*100 for i in ids]
        axes[0].scatter(vals, range(len(ids)), color='#246b91', s=45)
        axes[0].set_yticks(range(len(ids)))
        axes[0].set_yticklabels(ids)
        axes[0].invert_yaxis()
        axes[0].set_xlim(min(vals)-1.0, max(vals)+1.2)
        for n, v in enumerate(vals):
            axes[0].annotate(f'{v:.2f}', (v, n), xytext=(6, 0), textcoords='offset points', va='center')
        axes[0].set_xlabel('Archived best_map50 report: Mask AP50 (%)')
        axes[0].set_title(title + ' (single seed; no uncertainty estimate)')
        for eid in ids:
            curve = read_curve(root / f'{eid}_crack-seg_seed{lookup[eid]["seed"]}' / 'results.csv')
            axes[1].plot([r['epoch'] for r in curve], [100*r['metrics/mAP50(M)'] for r in curve], label=eid, lw=1, alpha=.8)
        axes[1].set(xlabel='Epoch', ylabel='Training-validation Mask AP50 (%)', title='Raw curves (not smoothed)')
        axes[1].legend(ncol=3, fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'{title.lower()}.{ext}', dpi=240)
        plt.close(fig)
    available = [(family, lookup[a], lookup[b]) for family,a,b in PAIRS
                 if 'mask_map50' in lookup.get(a,{}) and 'mask_map50' in lookup.get(b,{})]
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    for i, (family, a, b) in enumerate(available):
        x, y = a['mask_map50']*100, b['mask_map50']*100
        ax.plot([x,y], [i,i], color='gray')
        ax.scatter(x, i, c='#777777', label='Baseline' if i==0 else None)
        ax.scatter(y, i, c='#246b91', label='CPSB' if i==0 else None)
        ax.annotate(f'{y-x:+.2f} pp', (max(x,y),i), xytext=(10,0), textcoords='offset points', va='center')
    ax.set_yticks(range(len(available)))
    ax.set_yticklabels([a[0] for a in available])
    ax.set(xlabel='Archived best_map50 report: Mask AP50 (%)', title='Family transfer (paired within each family; seed 1)')
    ax.margins(x=.25, y=.3)
    ax.legend(loc='lower right')
    ax.grid(alpha=.2)
    for ext in ('png','pdf'):
        fig.savefig(out / ('family.'+ext), dpi=240)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-dir', type=Path, default=Path('output_dir/crack-seg-9.11'))
    p.add_argument('--output-dir', type=Path, default=Path('docs/results/9.14-crack-seg-9.11'))
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--plots', action='store_true')
    args = p.parse_args()
    records, protocols = collect(args.runs_dir, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'metrics_audit.csv', records)
    (args.output_dir / 'tables.md').write_text(tables(records), encoding='utf-8')
    shared = {k: v for k, v in next(iter(protocols.values())).items()
              if all(d.get(k) == v for d in protocols.values())}
    varying = {k: {eid: d.get(k) for eid, d in protocols.items()}
               for k in set().union(*(d.keys() for d in protocols.values())) if k not in shared}
    (args.output_dir / 'protocol_audit.json').write_text(json.dumps({'shared': shared, 'varying': varying}, indent=2), encoding='utf-8')
    if args.plots:
        plot_results(records, args.runs_dir, args.output_dir)
    print(tables(records))
    print('Protocol fields varying:', ', '.join(sorted(varying)))


if __name__ == '__main__':
    main()
