"""Numerically verify Doppler PSD and stationary tap power for v2."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy import signal
from channel import gaussian_fir,scatter_tap


def main():
    rows=[]
    # 60 Hz sampling is enough for these Gaussian Doppler spectra and permits
    # long realizations. This also exercises the same polyphase interpolator.
    fs=60;length=fs*6000
    for i,spread in enumerate([.1,.5,1.,10.]):
        rate=max(30*spread,3)
        h=gaussian_fir(spread,rate)
        f,H=signal.freqz(h,worN=65536,whole=True,fs=rate)
        f=np.where(f>=rate/2,f-rate,f);p=np.abs(H)**2
        exact=2*np.sqrt(np.sum(f*f*p)/p.sum())
        tap=scatter_tap(length,fs,spread,np.random.default_rng(20261029+i))
        f,p=signal.welch(tap,fs,nperseg=32768,return_onesided=False,detrend=False)
        center=np.sum(f*p)/np.sum(p)
        empirical=2*np.sqrt(np.sum((f-center)**2*p)/p.sum())
        power=float(np.mean(np.abs(tap)**2))
        row=dict(requested_2sigma_hz=spread,fir_power_spectrum_2sigma_hz=float(exact),
                 empirical_output_psd_2sigma_hz=float(empirical),tap_mean_power=power,
                 empirical_duration_s=6000,empirical_fs_hz=fs,
                 passed=bool(abs(exact/spread-1)<.005 and abs(empirical/spread-1)<.15 and .75<power<1.25))
        rows.append(row)
    out=Path(__file__).parent/'channel_validation.json'
    out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2),flush=True)
    if not all(r['passed'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
