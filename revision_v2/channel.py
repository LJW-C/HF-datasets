"""Gaussian-PSD Watterson-type channel; spread is output PSD two-sigma."""
import math
import numpy as np
from scipy import signal

PROFILES = {'good':(.5,.1), 'moderate':(1.,.5), 'poor':(2.,1.), 'flutter':(.5,10.)}


def gaussian_fir(spread_hz, rate, attenuation_db=60.):
    if spread_hz<=0 or rate<=0:
        raise ValueError('Positive spread and filter rate required')
    sigma=spread_hz/2
    # The filter amplitude is the square root of the target Gaussian PSD.
    coefficient=4*math.pi**2*sigma**2
    half=int(math.ceil(math.sqrt(-math.log(10**(-attenuation_db/20))/coefficient)*rate))
    t=np.arange(-half,half+1)/rate
    h=np.exp(-coefficient*t*t)
    return h/np.sqrt(np.sum(h*h))


def scatter_tap(length,fs,spread_hz,rng):
    rate=max(30*spread_hz,3.)
    if not float(rate).is_integer():
        raise ValueError('This implementation requires an integer fade rate')
    rate=int(rate)
    h=gaussian_fir(spread_hz,rate)
    low_len=int(math.ceil(length*rate/fs))+32
    white=(rng.standard_normal(low_len+len(h)-1)+1j*rng.standard_normal(low_len+len(h)-1))/math.sqrt(2)
    low=signal.fftconvolve(white,h,mode='valid')
    gcd=math.gcd(fs,rate)
    high=signal.resample_poly(low,fs//gcd,rate//gcd)
    begin=int(round(16*fs/rate))
    return high[begin:begin+length].astype(np.complex64)


def apply_channel(x,fs,profile,seed):
    delay_ms,spread=PROFILES[profile]
    rng=np.random.default_rng(seed)
    h1=scatter_tap(len(x),fs,spread,rng)
    h2=scatter_tap(len(x),fs,spread,rng)
    delay=int(round(delay_ms*fs/1000))
    delayed=np.zeros_like(x)
    if delay: delayed[delay:]=x[:-delay]
    else: delayed[:]=x
    return ((h1*x+h2*delayed)/math.sqrt(2)).astype(np.complex64)
