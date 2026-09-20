"""The original seven analysis tests, preserved for the portable package."""
import json
from pathlib import Path
import tempfile
import unittest
from imut_cdr_gc.naturalness import (calibrate, apply_calibration, score_qc, read_pairs,
    read_score_rows, build_layout298, SEGMENTS, WIDTHS)


class NaturalnessTests(unittest.TestCase):
    def test_quantile_filter(self):
        cal = calibrate(list(range(101)), 'model:sha:normalization', .05, .975, 'reference')
        self.assertEqual(cal['lower_cutoff'], 5)
        self.assertEqual(cal['upper_cutoff'], 97.5)
        labels, counts = apply_calibration([None, 4.99, 5, 97.5, 98], cal, 'model:sha:normalization')
        self.assertEqual(labels, ['missing', 'below_lower', 'pass', 'pass', 'above_upper'])
        self.assertEqual(sum(counts.values()), 5)

    def test_guard_model(self):
        cal = calibrate(list(range(10)), 'one', None, None, 'reference')
        with self.assertRaises(ValueError):
            apply_calibration([1], cal, 'two')

    def test_degenerate(self):
        self.assertTrue(score_qc([0, 0, None])['degenerate'])
        with self.assertRaises(ValueError):
            calibrate([0]*20, 'one', .05, 1, 'reference')

    def test_nonfinite(self):
        self.assertEqual(score_qc([1, float('nan'), None])['missing_or_nonfinite'], 2)

    def test_layout_and_no_truncation(self):
        light = {s: 'A'*n for s, n in zip(SEGMENTS, WIDTHS['light'])}
        heavy = {s: 'G'*n for s, n in zip(SEGMENTS, WIDTHS['heavy'])}
        self.assertEqual(len(build_layout298(heavy, light)), 298)
        heavy['cdr1'] += 'G'
        with self.assertRaises(ValueError):
            build_layout298(heavy, light)

    def test_no_founder_fallback(self):
        root = Path(__file__).resolve().parents[1]/'.cache/tmp'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix='naturalness_test_') as folder:
            path = Path(folder)/'input.jsonl'
            path.write_text(json.dumps(dict(mutant_id='m', heavy_full='AAA', light_full='CCC'))+'\n')
            self.assertEqual(read_pairs(path)[0]['heavy'], '')

    def test_scorer_jsonl_can_be_reused(self):
        root = Path(__file__).resolve().parents[1]/'.cache/tmp'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root, prefix='score_io_') as folder:
            path = Path(folder)/'scores.jsonl'
            record = dict(id='one', average_log_likelihood=-1.2)
            path.write_text(json.dumps(record)+'\n')
            self.assertEqual(read_score_rows(path), [record])


if __name__ == '__main__':
    unittest.main()
