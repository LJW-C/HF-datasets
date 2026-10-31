"""Build a new time-disjoint benchmark from readback-verified recordings."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent/'python_packages'))
import numpy as np
from scipy import signal
from scipy.io import wavfile
from numpy.lib.format import open_memmap
from channel import PROFILES,apply_channel
from source_checks import bandwidth_reference

SPLITS=['train','val','test']


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def read_audio(path):
    fs,a=wavfile.read(path)
    if np.issubdtype(a.dtype,np.integer):
        info=np.iinfo(a.dtype)
        a=(a.astype(np.float64)-128)/128 if a.dtype==np.uint8 else a.astype(np.float64)/max(abs(info.min),info.max)
    else: a=a.astype(np.float64)
    if a.ndim==2:a=a.mean(axis=1)
    if not np.isfinite(a).all(): raise ValueError('Nonfinite audio')
    return int(fs),a


def to_iq(audio,fs,output_fs,carrier):
    # Called separately inside each raw-time partition, so Hilbert and resampling
    # never use source samples from a different split.
    audio=audio-audio.mean()
    analytic=signal.hilbert(audio)
    analytic*=np.exp(-2j*np.pi*carrier*np.arange(len(audio))/fs)
    gcd=math.gcd(fs,output_fs)
    return signal.resample_poly(analytic,output_fs//gcd,fs//gcd).astype(np.complex64)


def raw_intervals(record,fs):
    start,stop=record['payload_interval_seconds']
    bounds=[round(start*fs),round((start+.7*(stop-start))*fs),round((start+.85*(stop-start))*fs),round(stop*fs)]
    return [(bounds[i],bounds[i+1]) for i in range(3)]


def build(args):
    recordings=[]
    for path in sorted(args.raw_root.glob('*/recording.json'), key=lambda p: p.parent.name.casefold()):
        r=json.loads(path.read_text(encoding='utf-8'))
        if not r['qc']['passed']:continue
        if args.modes and r['mode_name'] not in args.modes.split(','):continue
        if r['readback']['txid'] or r['readback']['rsid']:raise ValueError('RSID state is not disabled')
        if r['readback']['actual_modem']!=r['requested_modem']:raise ValueError('Modem identity mismatch')
        if not r.get('monitor') or any(m['modem']!=r['requested_modem'] or m['txid'] or str(m['trx']).lower()!='tx' or abs(m['carrier_hz']-r['readback']['carrier_hz'])>1 for m in r['monitor']):
            raise ValueError('Recorded TX state monitoring failed')
        reference=bandwidth_reference(r['requested_modem'],r['readback']['bandwidth_hz'])
        if reference>0 and not .45*reference<=r['qc']['obw99_hz']<=2.2*reference:
            raise ValueError('Spectral identity screen failed for '+r['mode_name'])
        r['record_metadata_file']=str(path.resolve());recordings.append(r)
    if not recordings:raise ValueError('No verified recordings')
    if args.expected_modes and len(recordings)!=args.expected_modes:raise ValueError('Verified mode count differs from expected count')
    # Claim a new output directory atomically. Never overwrite an old dataset.
    args.out_root.mkdir(parents=True,exist_ok=False)
    dump(args.out_root/'build_status.json',{'status':'building'})
    profiles=list(PROFILES)
    if args.vectors_per_mode_snr%80:raise ValueError('vectors-per-mode-snr must be divisible by 80 for 70/15/15 and four balanced profiles')
    quotas=[args.vectors_per_mode_snr*7//10,args.vectors_per_mode_snr*15//100,args.vectors_per_mode_snr*15//100]
    snrs=[25,20,15,10,5,0,-5,-10]
    total=len(recordings)*len(snrs)*sum(quotas)
    x=open_memmap(args.out_root/'X.npy',mode='w+',dtype=np.complex64,shape=(total,args.vector_len))
    label_arrays={k:open_memmap(args.out_root/(k+'.npy'),mode='w+',dtype=dtype,shape=(total,)) for k,dtype in
                  [('y','int64'),('snr_db','int16'),('channel_profile','int16'),('split_id','int8'),
                   ('source_id','int16'),('crop_start_iq','int64'),('raw_partition_start','int64'),
                   ('raw_partition_stop','int64'),('fading_seed','int64'),('impairment_seed','int64'),
                   ('cfo_hz','float32'),('phase_rad','float32'),('clean_power','float32'),('realized_snr_db','float32')]}
    indices={name:[] for name in SPLITS};partitions=[];idx=0
    for cls,record in enumerate(recordings):
        portable_source=Path(record['record_metadata_file']).parent/'source.wav'
        path=portable_source if portable_source.exists() else Path(record['wav'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Source checksum changed')
        fs,audio=read_audio(path)
        for split,(begin,end) in enumerate(raw_intervals(record,fs)):
            iq=to_iq(audio[begin:end],fs,args.fs,record['readback']['carrier_hz'])
            guard=max(args.vector_len,int(args.guard_sec*args.fs))
            if len(iq)<2*guard+args.vector_len:raise ValueError('Source partition too short; repetition is forbidden')
            # Reject empty windows based on clean pre-channel source power, not
            # noise power or instantaneous channel fading depth.
            cumulative=np.r_[0.,np.cumsum(np.abs(iq).astype(np.float64)**2)]
            energies=(cumulative[args.vector_len:]-cumulative[:-args.vector_len])/args.vector_len
            threshold=max(1e-9,float(np.percentile(energies,95))*args.activity_power_ratio)
            starts=np.flatnonzero(energies>threshold)
            starts=starts[(starts>=guard)&(starts+args.vector_len<=len(iq)-guard)]
            if not len(starts):raise ValueError('No active clean windows')
            partitions.append(dict(source_id=cls,mode=record['mode_name'],split=SPLITS[split],
                raw_start=begin,raw_stop=end,raw_fs=fs,iq_length=len(iq),guard_iq_samples=guard,
                activity_power_threshold=threshold,eligible_crop_starts=len(starts),
                no_source_overlap_across_splits=True))
            channels=[];seeds=[]
            for profile_id,profile in enumerate(profiles):
                fade_seed=args.seed+cls*10000+split*100+profile_id
                channels.append(apply_channel(iq,args.fs,profile,fade_seed));seeds.append(fade_seed)
            rng=np.random.default_rng(args.seed+cls*10000+split*100+50)
            for snr in snrs:
                # Balance profiles exactly in every configuration/SNR/split cell.
                profile_ids=np.repeat(np.arange(len(profiles)),quotas[split]//len(profiles))
                rng.shuffle(profile_ids)
                for profile_id in profile_ids:
                    imp_seed=int(rng.integers(1,2**62));local=np.random.default_rng(imp_seed)
                    start=int(local.choice(starts))
                    clean=channels[profile_id][start:start+args.vector_len].copy()
                    power=float(np.mean(np.abs(clean)**2))
                    if power<1e-16:raise ValueError('Degenerate channel output')
                    clean/=np.sqrt(power)
                    cfo=float(local.uniform(-args.freq_offset_hz,args.freq_offset_hz));phase=float(local.uniform(0,2*np.pi))
                    clean*=np.exp(1j*(2*np.pi*cfo*np.arange(args.vector_len)/args.fs+phase))
                    variance=10**(-snr/10)
                    noise=(np.sqrt(variance/2)*(local.standard_normal(args.vector_len)+1j*local.standard_normal(args.vector_len))).astype(np.complex64)
                    x[idx]=clean+noise
                    fields=dict(y=cls,snr_db=snr,channel_profile=profile_id,split_id=split,source_id=cls,
                        crop_start_iq=start,raw_partition_start=begin,raw_partition_stop=end,
                        fading_seed=seeds[profile_id],impairment_seed=imp_seed,cfo_hz=cfo,phase_rad=phase,
                        clean_power=power,realized_snr_db=10*np.log10(float(np.mean(abs(clean)**2))/float(np.mean(abs(noise)**2))))
                    for key,value in fields.items():label_arrays[key][idx]=value
                    indices[SPLITS[split]].append(idx);idx+=1
            print(record['mode_name'],SPLITS[split], 'done',idx,flush=True)
        x.flush()
        for array in label_arrays.values():array.flush()
    np.savez_compressed(args.out_root/'benchmark_split_indices.npz',**{k:np.array(v,dtype=np.int64) for k,v in indices.items()})
    dump(args.out_root/'mode_names.json',[r['mode_name'] for r in recordings])
    dump(args.out_root/'channel_profiles.json',profiles)
    dump(args.out_root/'source_recordings.json',recordings)
    dump(args.out_root/'source_partitions.json',partitions)
    code_files=['channel.py','build_verified_dataset.py','record_verified_sources.py','verify_dataset.py','source_checks.py']
    dump(args.out_root/'generation_code_sha256.json',{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in code_files})
    code_dir=args.out_root/'generation_code'
    code_dir.mkdir()
    for name in code_files:shutil.copy2(Path(__file__).parent/name,code_dir/name)
    versions={}
    for name in ['numpy','scipy','torch','scikit-learn','sounddevice','soundfile']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not_installed'
    dump(args.out_root/'generation_environment.json',dict(python=sys.version,packages=versions))
    dump(args.out_root/'meta.json',dict(dataset='HFDigiSet_revision_v2',num_vectors=total,vector_len=args.vector_len,
        fs_hz=args.fs,seed=args.seed,snr_db_values=snrs,profiles=PROFILES,spread_definition='output Gaussian Doppler PSD two_sigma_hz',
        fading_interpolation='polyphase resample with 16 low-rate-sample margins',
        split_protocol='raw_time_disjoint_before_hilbert_resampling_and_fading',split_counts={k:len(v) for k,v in indices.items()},
        vectors_per_mode_snr=args.vectors_per_mode_snr,profile_assignment='balanced_within_mode_snr_split',
        raw_source_count=len(recordings),source_recording_seconds=[r['duration_sec'] for r in recordings],
        guard_sec=args.guard_sec,activity_power_ratio=args.activity_power_ratio,
        random_frequency_offset_hz=[-args.freq_offset_hz,args.freq_offset_hz],phase_range_rad=[0,2*np.pi],
        snr_definition='clean signal power / complex AWGN power over full fs bandwidth',
        short_source_policy='reject',shared_fading='within mode/profile/split, never across splits',
        limitation='One recording per mode; partitions are time-disjoint, not recording-disjoint. Crops can overlap within one split.'))
    dump(args.out_root/'build_status.json',dict(status='complete',num_vectors=idx))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=Path(__file__).parent/'raw_verified')
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--modes',default='')
    p.add_argument('--expected-modes',type=int,default=17)
    p.add_argument('--vectors-per-mode-snr',type=int,default=1200)
    p.add_argument('--vector-len',type=int,default=2048)
    p.add_argument('--fs',type=int,default=6000)
    p.add_argument('--freq-offset-hz',type=float,default=250)
    p.add_argument('--guard-sec',type=float,default=.5)
    p.add_argument('--activity-power-ratio',type=float,default=.001)
    p.add_argument('--seed',type=int,default=20261029)
    build(p.parse_args())


if __name__=='__main__':main()
