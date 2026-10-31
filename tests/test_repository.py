"""Offline regressions for provenance, immutable splits and public entry points."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'revision_v2'))
from rfml_benchmark.data import make_splits


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        np.save(self.root / 'y.npy', np.arange(12) % 2)
        (self.root / 'meta.json').write_text(json.dumps({
            'split_protocol': 'raw_time_disjoint_before_hilbert_resampling_and_fading'}))
        self.parts = dict(train=np.arange(8), val=np.arange(8, 10), test=np.arange(10, 12))

    def save(self):
        np.savez(self.root / 'benchmark_split_indices.npz', **self.parts)

    def test_saved_splits_do_not_change_with_training_seed(self):
        self.save()
        for seed in [1, 20261029, 20261030]:
            actual = make_splits(self.root, .7, .15, seed)
            for key, part in zip(['train', 'val', 'test'], actual):
                np.testing.assert_array_equal(part, self.parts[key])

    def test_missing_saved_split_cannot_fall_back_to_random(self):
        with self.assertRaisesRegex(ValueError, 'missing its predefined split'):
            make_splits(self.root, .7, .15, 1)

    def test_duplicate_index_is_rejected(self):
        self.parts['test'][0] = self.parts['train'][0]
        self.save()
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            make_splits(self.root, .7, .15, 1)

    def test_predefined_subsampling_is_rejected(self):
        self.save()
        with self.assertRaisesRegex(ValueError, 'Subsampling'):
            make_splits(self.root, .7, .15, 1, max_samples=6)


class RepositoryTests(unittest.TestCase):
    def test_local_markdown_links_resolve(self):
        for rel in ['README.md', 'data/README.md', 'revision_v2/README.md', 'legacy/README.md']:
            path = ROOT / rel
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                if '://' not in target and not target.startswith('#'):
                    self.assertTrue((path.parent / target.split('#')[0]).exists(), (rel, target))

    def test_scope_and_profile_maps_match(self):
        meta = json.loads((ROOT / 'data/schema/meta.json').read_text())
        names = json.loads((ROOT / 'data/schema/mode_names.json').read_text())
        profiles = json.loads((ROOT / 'data/schema/channel_profiles.json').read_text())
        self.assertEqual(len(names), 17)
        self.assertEqual(profiles, ['good', 'moderate', 'poor', 'flutter'])
        self.assertEqual(meta['split_counts'], dict(train=114240, val=24480, test=24480))
        self.assertEqual(meta['vectors_per_mode_snr'], 1200)

    def test_measured_reference_and_repeats_are_distinct(self):
        with (ROOT / 'revision_v2/results/model_metrics.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 7)
        self.assertEqual({int(row['test_samples']) for row in rows}, {24480})
        reference = next(row for row in rows if row['model'] == 'conformer')
        self.assertAlmostEqual(float(reference['accuracy']), .8526143790849673)
        with (ROOT / 'revision_v2/supplementary_experiments_20261030/repeat_summary.csv').open(encoding='utf-8-sig') as stream:
            repeats = list(csv.DictReader(stream))
        self.assertEqual({row['model'] for row in repeats}, {'conformer', 'random_forest'})
        self.assertTrue(all(int(row['seeds']) == 3 for row in repeats))

    def test_public_builders_dispatch_to_verified_sources(self):
        for entry in ['build_datasets.py', 'build_all_dataset.py', 'revision_v2/train_suite.py',
                      'revision_v2/record_verified_sources.py']:
            result = subprocess.run([sys.executable, str(ROOT / entry), '--help'],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            flag = '--list-devices' if 'record_verified' in entry else (
                '--dataset-root' if 'train_suite' in entry else '--expected-modes')
            self.assertIn(flag, result.stdout)

    def test_source_mapping_hashes(self):
        mapping = json.loads((ROOT / 'data/repository_source_mapping.json').read_text())
        self.assertTrue(mapping['files'])
        for item in mapping['files']:
            digest = hashlib.sha256((ROOT / item['repository_path']).read_bytes()).hexdigest()
            self.assertEqual(digest, item['repository_sha256'], item['repository_path'])
            self.assertEqual(item['byte_identical'], digest == item['snapshot_sha256'])

    def test_download_checker_detects_corruption(self):
        spec = importlib.util.spec_from_file_location('verify_download', ROOT / 'scripts/verify_download.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'fixture.bin'
            path.write_bytes(b'fixture')
            records = [dict(file=path.name, bytes=7, sha256=hashlib.sha256(b'fixture').hexdigest())]
            manifest = root / 'SHA256SUMS.json'
            manifest.write_text(json.dumps(records))
            self.assertEqual(module.verify(root, manifest)['status'], 'passed')
            path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                module.verify(root, manifest)


if __name__ == '__main__':
    unittest.main()
