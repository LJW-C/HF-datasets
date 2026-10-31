"""Validate paired metadata and the shared underlying Gaussian-noise realization."""
from pathlib import Path
import json
import numpy as np


def main():
    root=Path(__file__).resolve().parent/'paired_windows_20261030'
    data={length:{p.stem:np.load(p,mmap_mode='r') for p in (root/str(length)).glob('*.npy')}
          for length in [2048,8192]}
    parts=json.loads((root/'8192/source_partitions.json').read_text())
    by_key={(p['source_id'],i%3):p for i,p in enumerate(parts)}
    for key in data[2048]:
        if key not in ['X','clean_power','realized_snr_db']:
            assert np.array_equal(data[2048][key],data[8192][key]),key
    errors=[]
    for idx in np.random.default_rng(1).choice(len(data[8192]['y']),128,replace=False):
        large=data[8192]
        p=by_key[(int(large['source_id'][idx]),int(large['split_id'][idx]))]
        rng=np.random.default_rng(int(large['impairment_seed'][idx]))
        rng.integers(p['eligible_crop_starts'])
        cfo=rng.uniform(-250,250);phase=rng.uniform(0,2*np.pi)
        assert abs(cfo-large['cfo_hz'][idx])<2e-5
        assert abs(phase-large['phase_rad'][idx])<3e-7
        variance=10**(-float(large['snr_db'][idx])/10)
        noise=(np.sqrt(variance/2)*(rng.standard_normal(8192)+1j*rng.standard_normal(8192))).astype('complex64')
        clean_short=data[2048]['X'][idx]-noise[:2048]
        clean_long=large['X'][idx,:2048]-noise[:2048]
        scale=np.sqrt(large['clean_power'][idx]/data[2048]['clean_power'][idx])
        error=float(np.max(abs(clean_short-clean_long*scale)))
        assert error<8e-6,(idx,error)
        errors.append(error)
    report=dict(status='passed',paired_rows=len(data[2048]['y']),noise_reconstruction_checks=128,
        max_clean_pair_error=float(max(errors)),
        normalization='Separate clean-crop power in each arm; same target full-band noise variance.')
    (root/'noise_pairing_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
