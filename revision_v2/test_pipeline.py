"""Focused numerical and provenance tests. Fixtures are not research data."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy import signal
from scipy.io import wavfile
from channel import gaussian_fir,scatter_tap
from build_verified_dataset import build,read_audio
from verify_dataset import verify


class PipelineTests(unittest.TestCase):
    def test_doppler_power_spectrum_width(self):
        for spread in [.1,.5,1.,10.]:
            rate=max(30*spread,3)
            h=gaussian_fir(spread,rate)
            f,H=signal.freqz(h,worN=65536,whole=True,fs=rate)
            f=np.where(f>=rate/2,f-rate,f);p=np.abs(H)**2
            measured=2*np.sqrt(np.sum(f*f*p)/p.sum())
            self.assertAlmostEqual(measured/spread,1.,delta=.005)

    def test_scatter_seed_and_length(self):
        a=scatter_tap(12000,6000,.1,np.random.default_rng(3))
        b=scatter_tap(12000,6000,.1,np.random.default_rng(3))
        c=scatter_tap(12000,6000,.1,np.random.default_rng(4))
        self.assertEqual(a.shape,(12000,));self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a,c))

    def test_stereo_pcm_and_disjoint_build(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            fs=8000;t=np.arange(20*fs)/fs
            for cls in range(2):
                folder=raw/str(cls);folder.mkdir()
                audio=(np.sin(2*np.pi*(1500+cls*100)*t)*.4*32767).astype(np.int16)
                path=folder/'source.wav';wavfile.write(path,fs,np.stack([audio,audio],axis=1))
                rate,mono=read_audio(path)
                self.assertEqual(rate,fs);self.assertLess(np.max(abs(mono)),.41)
                record=dict(mode_name='fixture_'+str(cls),requested_modem='fixture',qc=dict(passed=True,obw99_hz=100),
                    readback=dict(actual_modem='fixture',bandwidth_hz=0,txid=False,rsid=False,carrier_hz=1500),wav=str(path),
                    monitor=[dict(modem='fixture',txid=False,trx='TX',carrier_hz=1500)],
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    payload_interval_seconds=[1,19],duration_sec=20)
                (folder/'recording.json').write_text(json.dumps(record))
            out=root/'data'
            build(SimpleNamespace(raw_root=raw,out_root=out,modes='',expected_modes=2,
                vectors_per_mode_snr=80,vector_len=256,fs=6000,freq_offset_hz=250,
                guard_sec=.1,activity_power_ratio=.001,seed=3))
            report=verify(out)
            self.assertEqual(report['split_counts'],dict(train=896,val=192,test=192))
            self.assertLess(abs(report['realized_snr_error_mean_db']),.1)
            # A tampered split must fail verification.
            path=out/'benchmark_split_indices.npz'
            with np.load(path) as z:parts={k:z[k] for k in z.files}
            parts['test'][0]=parts['train'][0]
            np.savez(path,**parts)
            with self.assertRaises(ValueError):verify(out)


if __name__=='__main__':unittest.main()
