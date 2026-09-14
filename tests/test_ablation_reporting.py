"""Run directly: python tests/test_ablation_reporting.py (no Ultralytics import)."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from summarize_ablation_results import collect, parse_report, tables
from visualize_ablation_paper import scatter_path, memory_diagnostics, figure_grid, structure_figures, resolve_specs, resolve_splits, write_rows


class ReportingTests(unittest.TestCase):
    def test_all_runs_includes_all_seeds_and_skips_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('K01_crack-seg_seed0', 'K01_crack-seg_seed1', 'Y10_other_seed2'):
                folder = root/name/'weights'
                folder.mkdir(parents=True)
                (folder/'best_map50.pt').write_bytes(b'test only')
            (root/'incomplete').mkdir()
            args = SimpleNamespace(manifest=None, experiments=['all'], runs_dir=root,
                                   seed=1, checkpoint='best_map50.pt', layers='4,6')
            specs = resolve_specs(args)
            self.assertEqual(len(specs), 3)
            self.assertIn('K01_crack-seg_seed0', [s['label'] for s in specs])
            args.experiments = ['K01']
            self.assertEqual(resolve_specs(args)[0]['label'], 'K01')
            args.experiments = ['all', 'K01']
            with self.assertRaises(ValueError):
                resolve_specs(args)

    def test_all_splits_and_partitioned_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'data.yaml'
            source.write_text('train: images/train\nval: images/val\ntest: images/test\n')
            self.assertEqual(resolve_splits(source, 'all'), ['train', 'val', 'test'])
            source.write_text('train: images/train\nval: images/val\ntest: null\n')
            self.assertEqual(resolve_splits(source, 'all'), ['train', 'val'])
            self.assertEqual(resolve_splits(source, 'test'), ['test'])
            with self.assertRaises(ValueError):
                resolve_splits(root/'image.jpg', 'all')
            for split in ('train', 'val'):
                folder = root/'K01'/split
                folder.mkdir(parents=True)
                write_rows(folder/'metrics.csv', [{'model':'K01', 'split':split, 'value':1}])
                self.assertIn(split, (folder/'metrics.csv').read_text(encoding='utf-8-sig'))

    def test_actual_reports(self):
        root = Path(__file__).resolve().parents[1] / 'output_dir/crack-seg-9.11'
        if not root.exists():
            self.skipTest('Training artifacts not on this machine')
        rows, _ = collect(root, 1)
        d = {r['id']: r for r in rows}
        self.assertAlmostEqual(d['K01']['mask_map50'], .6905)
        self.assertAlmostEqual(d['K01']['csv_peak_mask_map50'], .69656)
        self.assertAlmostEqual(100*(d['K29']['mask_map50']-d['K28']['mask_map50']), 2.05)
        self.assertNotIn('K00', d)
        self.assertIn('YOLO11n baseline', tables(rows))

    def test_missing_metric_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'bad.txt'
            p.write_text('mAP@0.5 0.8 0.7')
            with self.assertRaises(ValueError):
                parse_report(p)

    def test_scatter_invalid_and_repeated(self):
        idx = np.array([[0,1,1], [2,3,0]])
        valid = np.array([[1,0,1], [0,1,1]], bool)
        values = np.array([[1,999,3], [999,7,5]])
        result, cover = scatter_path(values, idx, valid, (2,2))
        np.testing.assert_allclose(result, [[3,3], [np.nan,7]], equal_nan=True)
        self.assertEqual(int(cover.sum()), 3)

    def test_memory_and_render(self):
        import torch
        import matplotlib
        matplotlib.use('Agg')
        torch.manual_seed(1)
        m = SimpleNamespace(channels=2, d_state=3, dt_rank=1,
                            x_proj_weight=torch.randn(2,7,2), dt_projs_weight=torch.randn(2,2,1),
                            dt_projs_bias=torch.randn(2,2), A_logs=torch.zeros(4,3),
                            effective_memory=lambda: torch.tensor(.2),
                            effective_transition=lambda: torch.tensor(.1),
                            effective_write=lambda: torch.tensor(.2))
        x = torch.zeros(2,2,3)
        p = torch.tensor([[[0.,.5,1.]], [[1.,.5,0.]]])
        edge = torch.ones_like(p)
        valid = torch.ones_like(p, dtype=torch.bool)
        d = memory_diagnostics(m, (x,p,edge,valid))
        np.testing.assert_allclose(d['retention_bias'][0], [.1,0,-.1], atol=1e-6)
        np.testing.assert_allclose(d['write_gain'][0], [.8,1,1.2], atol=1e-6)
        self.assertLess(d['decay'][0,0], d['decay_without_gates'][0,0])
        self.assertGreater(d['decay'][0,2], d['decay_without_gates'][0,2])
        m.effective_memory = lambda: torch.tensor(0.)
        m.effective_transition = lambda: torch.tensor(0.)
        m.effective_write = lambda: torch.tensor(0.)
        standard = memory_diagnostics(m, (x,p,edge,valid))
        np.testing.assert_allclose(standard['decay'], standard['decay_without_gates'])
        np.testing.assert_allclose(standard['write_gain'], 1.)
        canvas = np.full((64,64,3), 100, np.uint8)
        state = SimpleNamespace(last_guidance=torch.ones(1,1,2,2)*.5,
                                last_path_indices=torch.tensor([[[0,1,3],[0,2,3]]]),
                                last_path_mask=torch.ones(1,2,3,dtype=torch.bool))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            figure_grid([[('synthetic',canvas),('synthetic',canvas)]], ['a','b'], path/'panel')
            rows = structure_figures('synthetic', state, d, canvas, None, np.ones((64,64), bool), path/'state', 1)
            self.assertEqual(rows[0]['unique_feature_coverage'], 1.)
            self.assertTrue((path/'panel.pdf').is_file())
            self.assertTrue((path/'state.png').is_file())


if __name__ == '__main__':
    unittest.main()
