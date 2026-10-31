"""Exercise portable builders, all model shapes and one-epoch CPU training.

Synthetic tones are integration fixtures, never additional research results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import wavfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rfml_benchmark.models import MODEL_NAMES, build_model


def run(args, log):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                            capture_output=True, text=True, timeout=180)
    log.append(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError('Command failed: ' + ' '.join(map(str, args)) + '\n' + log[-1][-5000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    log = []
    with tempfile.TemporaryDirectory(prefix='hfdigiset_smoke_') as tmp:
        root = Path(tmp)
        raw = root / 'raw'
        raw.mkdir()
        fs = 8000
        t = np.arange(20 * fs) / fs
        for cls in range(2):
            folder = raw / str(cls)
            folder.mkdir()
            path = folder / 'source.wav'
            audio = (.4 * 32767 * np.sin(2 * np.pi * (1500 + cls * 100) * t)).astype('int16')
            wavfile.write(path, fs, audio)
            record = dict(mode_name='fixture_' + str(cls), requested_modem='fixture',
                qc=dict(passed=True, obw99_hz=100),
                readback=dict(actual_modem='fixture', bandwidth_hz=0, txid=False, rsid=False, carrier_hz=1500),
                monitor=[dict(modem='fixture', txid=False, trx='TX', carrier_hz=1500)],
                wav=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                payload_interval_seconds=[1, 19], duration_sec=20)
            (folder / 'recording.json').write_text(json.dumps(record))
        dataset = root / 'dataset'
        run(['build_datasets.py', '--raw-root', raw, '--out-root', dataset,
             '--expected-modes', 2, '--vectors-per-mode-snr', 80, '--vector-len', 256,
             '--guard-sec', .1, '--seed', 3], log)
        run(['revision_v2/verify_dataset.py', dataset], log)
        for name in MODEL_NAMES:
            model = build_model(name, num_classes=2, input_channels=2, width=16, dropout=0).eval()
            with torch.no_grad():
                output = model(torch.randn(2, 2, 2048))
            assert tuple(output.shape) == (2, 2) and torch.isfinite(output).all(), name
        runs = root / 'neural_runs'
        command = ['revision_v2/train_suite.py', '--dataset-root', dataset,
                   '--out-dir', runs, '--models', 'iq_cnn', '--epochs', 1,
                   '--seeds', '3', '--jobs', 1, '--device', 'cpu']
        run(command, log)
        neural = runs / 'seed_3/iq_cnn'
        run(['revision_v2/train_classical_baseline.py', '--dataset-root', dataset,
             '--out-dir', root / 'random_forest', '--seed', 3], log)
        with np.load(dataset / 'benchmark_split_indices.npz') as split:
            for run_root in [neural, root / 'random_forest']:
                with np.load(run_root / 'test_predictions.npz') as pred:
                    np.testing.assert_array_equal(pred['index'], split['test'])
                metrics = json.loads((run_root / 'metrics.json').read_text())
                assert metrics['test_samples'] == 192
        rejected = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT,
                                  capture_output=True, text=True, timeout=60)
        assert rejected.returncode != 0 and 'Refusing to overwrite' in rejected.stderr
        run(['visualize_iq_samples.py', '--dataset-root', dataset,
             '--first', 1, '--out-dir', root / 'plots', '--spectrogram-nperseg', 128], log)
        import matplotlib.image as mpimg
        image = root / 'plots/sample_000000.png'
        pixels = mpimg.imread(image)
        assert pixels.shape[0] > 400 and float(pixels.std()) > .01
        report = dict(status='passed', fixture_only=True, full_research_training=False,
                      vectors=1280, test_vectors=192, model_forward_checks=len(MODEL_NAMES),
                      cpu_trainers=['iq_cnn_one_epoch', 'random_forest'],
                      fixed_test_indices=True, overwrite_rejected=True, visualization_nonblank=True)
        if args.report_dir:
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / 'smoke_training.json').write_text(json.dumps(report, indent=2))
            (args.report_dir / 'smoke_training.log').write_text('\n'.join(log), encoding='utf-8')
            shutil.copy2(image, args.report_dir / 'iq_smoke.png')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
