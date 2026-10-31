"""Paired Olivia/Contestia duration study, separate from the 17-class benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from numpy.lib.format import open_memmap
from build_verified_dataset import read_audio, raw_intervals, to_iq, dump, SPLITS
from channel import PROFILES, apply_channel
from verify_dataset import verify

MODES = ['Contestia_4_250', 'Contestia_8_500', 'Olivia_4_250', 'Olivia_8_500']
LENGTHS = [2048, 8192]


def build(output, count, seed):
    here = Path(__file__).resolve().parent
    if count % 80:
        raise ValueError('Count must be divisible by 80')
    records = json.loads((here/'benchmark_17/source_recordings.json').read_text(encoding='utf-8'))
    records = [r for r in records if r['mode_name'] in MODES]
    assert [r['mode_name'] for r in records] == MODES
    output.mkdir(parents=True, exist_ok=False)
    n = 4*8*count
    quotas = [count*7//10, count*15//100, count*15//100]
    arrays = {}
    dtypes = dict(y='int64', snr_db='int16', channel_profile='int16', split_id='int8',
                  source_id='int16', crop_start_iq='int64', raw_partition_start='int64',
                  raw_partition_stop='int64', fading_seed='int64', impairment_seed='int64',
                  cfo_hz='float32', phase_rad='float32', clean_power='float32',
                  realized_snr_db='float32', paired_id='int64')
    for length in LENGTHS:
        root = output/str(length)
        root.mkdir()
        dump(root/'build_status.json', dict(status='building'))
        arrays[length] = {'X': open_memmap(root/'X.npy', mode='w+', dtype='complex64', shape=(n,length))}
        for key, dtype in dtypes.items():
            arrays[length][key] = open_memmap(root/(key+'.npy'), mode='w+', dtype=dtype, shape=(n,))
    partitions = []
    indices = {k: [] for k in SPLITS}
    snrs = [25,20,15,10,5,0,-5,-10]
    idx = 0
    for cls, record in enumerate(records):
        wav = here/'raw_verified'/record['mode_name']/'source.wav'
        assert hashlib.sha256(wav.read_bytes()).hexdigest() == record['sha256']
        fs, audio = read_audio(wav)
        for split,(begin,end) in enumerate(raw_intervals(record,fs)):
            iq = to_iq(audio[begin:end],fs,6000,record['readback']['carrier_hz'])
            guard = max(LENGTHS)
            starts = np.arange(guard,len(iq)-guard-max(LENGTHS)+1)
            cumulative = np.r_[0.,np.cumsum(abs(iq).astype(np.float64)**2)]
            thresholds = {}
            for length in LENGTHS:
                powers = (cumulative[length:]-cumulative[:-length])/length
                threshold = max(1e-9,.001*float(np.percentile(powers,95)))
                starts = starts[powers[starts] > threshold]
                thresholds[str(length)] = threshold
            if not len(starts):
                raise ValueError('No paired active crop fits the common guard')
            partitions.append(dict(source_id=cls,mode=record['mode_name'],split=SPLITS[split],
                raw_start=begin,raw_stop=end,raw_fs=fs,iq_length=len(iq),guard_iq_samples=guard,
                activity_power_threshold=thresholds,eligible_crop_starts=len(starts),
                no_source_overlap_across_splits=True))
            fade_seeds = [seed+cls*10000+split*100+p for p in range(4)]
            channels = [apply_channel(iq,6000,p,s) for p,s in zip(PROFILES,fade_seeds)]
            rng = np.random.default_rng(seed+cls*10000+split*100+50)
            for snr in snrs:
                profile_ids = np.repeat(np.arange(4),quotas[split]//4)
                rng.shuffle(profile_ids)
                for profile in profile_ids:
                    imp_seed = int(rng.integers(1,2**62))
                    local = np.random.default_rng(imp_seed)
                    start = int(local.choice(starts))
                    cfo = float(local.uniform(-250,250))
                    phase = float(local.uniform(0,2*np.pi))
                    # Draw one long noise realization; the short arm uses its prefix.
                    noise = (np.sqrt(10**(-snr/10)/2)*(local.standard_normal(8192)+
                             1j*local.standard_normal(8192))).astype(np.complex64)
                    clean = channels[profile][start:start+8192]
                    rotation = np.exp(1j*(2*np.pi*cfo*np.arange(8192)/6000+phase))
                    for length in LENGTHS:
                        power = float(np.mean(abs(clean[:length])**2))
                        signal = clean[:length]/np.sqrt(power)*rotation[:length]
                        noise_arm = noise[:length]
                        arrays[length]['X'][idx] = signal+noise_arm
                        values = dict(y=cls,snr_db=snr,channel_profile=profile,split_id=split,
                            source_id=cls,crop_start_iq=start,raw_partition_start=begin,
                            raw_partition_stop=end,fading_seed=fade_seeds[profile],
                            impairment_seed=imp_seed,cfo_hz=cfo,phase_rad=phase,clean_power=power,
                            realized_snr_db=10*np.log10(float(np.mean(abs(signal)**2))/float(np.mean(abs(noise_arm)**2))),
                            paired_id=idx)
                        for key,value in values.items():
                            arrays[length][key][idx] = value
                    indices[SPLITS[split]].append(idx)
                    idx += 1
            print(record['mode_name'],SPLITS[split],idx,flush=True)
    for length in LENGTHS:
        root = output/str(length)
        for array in arrays[length].values():
            array.flush()
        np.savez_compressed(root/'benchmark_split_indices.npz',**{k:np.array(v,dtype='int64') for k,v in indices.items()})
        dump(root/'mode_names.json',MODES)
        dump(root/'channel_profiles.json',list(PROFILES))
        dump(root/'source_recordings.json',records)
        dump(root/'source_partitions.json',partitions)
        dump(root/'meta.json',dict(dataset='HFDigiSet_paired_duration_diagnostic',num_vectors=n,
            vector_len=length,fs_hz=6000,seed=seed,split_protocol='raw_time_disjoint_paired_duration',
            vectors_per_mode_snr=count,profile_assignment='balanced_within_mode_snr_split',
            split_counts={k:len(v) for k,v in indices.items()},snr_db_values=snrs,
            paired_lengths=LENGTHS,common_guard_iq=8192,
            paired_factors='source/start/fading/CFO/phase; short noise is prefix of long noise',
            per_arm_power='Each clean crop normalized separately to one; full-band target SNR unchanged',
            limitation='Four-class diagnostic, not the 17-class benchmark; one recording per class.'))
        dump(root/'build_status.json',dict(status='complete',num_vectors=idx))
        verify(root)
    for key in dtypes:
        if key not in ['clean_power','realized_snr_db']:
            assert np.array_equal(arrays[2048][key],arrays[8192][key]),key
    shutil.copy2(__file__,output/Path(__file__).name)
    dump(output/'pairing_verification.json',dict(status='passed',paired_rows=idx,lengths=LENGTHS,
        matching_metadata=[k for k in dtypes if k not in ['clean_power','realized_snr_db']],
        note='Per-arm normalization differs by design; noisy arrays are not identical prefixes.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--vectors-per-mode-snr',type=int,default=400)
    p.add_argument('--seed',type=int,default=20261030)
    a=p.parse_args()
    build(a.out_root,a.vectors_per_mode_snr,a.seed)
