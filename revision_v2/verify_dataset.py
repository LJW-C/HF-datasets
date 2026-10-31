"""Check saved split provenance, profile quotas, SNR and vector validity."""
import argparse
import json
from pathlib import Path
import numpy as np


def verify(root):
    status=json.loads((root/'build_status.json').read_text())
    if status['status']!='complete':raise ValueError('Build is incomplete')
    meta=json.loads((root/'meta.json').read_text())
    partitions=json.loads((root/'source_partitions.json').read_text())
    arrays={key:np.load(root/(key+'.npy'),mmap_mode='r') for key in ['X','y','snr_db','channel_profile','split_id','source_id','raw_partition_start','raw_partition_stop','crop_start_iq','realized_snr_db']}
    count=len(arrays['y']);saved=np.load(root/'benchmark_split_indices.npz')
    merged=np.concatenate([saved[key] for key in ['train','val','test']])
    if not np.array_equal(np.sort(merged),np.arange(count)):raise ValueError('Split overlap or missing vectors')
    for split,key in enumerate(['train','val','test']):
        if not np.all(arrays['split_id'][saved[key]]==split):raise ValueError('Split labels disagree')
    for source in np.unique(arrays['source_id']):
        intervals=sorted((p['raw_start'],p['raw_stop']) for p in partitions if p['source_id']==source)
        if len(intervals)!=3 or any(a[1]>b[0] for a,b in zip(intervals,intervals[1:])):
            raise ValueError('Raw source partitions overlap')
    for p in partitions:
        split=['train','val','test'].index(p['split'])
        ix=np.flatnonzero((arrays['source_id']==p['source_id'])&(arrays['split_id']==split))
        if not np.all(arrays['raw_partition_start'][ix]==p['raw_start']) or not np.all(arrays['raw_partition_stop'][ix]==p['raw_stop']):raise ValueError('Provenance differs from partition record')
        starts=arrays['crop_start_iq'][ix]
        if np.any(starts<p['guard_iq_samples']) or np.any(starts+meta['vector_len']>p['iq_length']-p['guard_iq_samples']):raise ValueError('Crop crosses guard')
    for cls in np.unique(arrays['y']):
        for snr in np.unique(arrays['snr_db']):
            for split in range(3):
                ix=(arrays['y']==cls)&(arrays['snr_db']==snr)&(arrays['split_id']==split)
                counts=np.bincount(arrays['channel_profile'][ix],minlength=4)
                if len(set(counts.tolist()))!=1:raise ValueError('Unbalanced profile quotas')
    for start in range(0,count,2048):
        if not np.isfinite(arrays['X'][start:start+2048]).all():raise ValueError('Nonfinite vectors')
    error=np.asarray(arrays['realized_snr_db'])-arrays['snr_db']
    report=dict(status='passed',vectors=count,split_counts={key:len(saved[key]) for key in saved.files},
                raw_time_disjoint=True,balanced_profiles=True,finite_vectors=True,
                realized_snr_error_mean_db=float(error.mean()),realized_snr_error_std_db=float(error.std()),
                limitation='Time-disjoint partitions of one recording per mode; not independent recording sessions.')
    (root/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    verify(p.parse_args().root)
