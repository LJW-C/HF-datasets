"""Conventional spectral/statistical feature + Random Forest baseline."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent/'python_packages'))
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix
import joblib
from verify_dataset import verify


def features(x):
    x=x/np.sqrt(np.maximum(np.mean(abs(x)**2,axis=1,keepdims=True),1e-12))
    amplitude=np.abs(x)
    spectral=np.abs(np.fft.fftshift(np.fft.fft(x*np.hanning(x.shape[1]),axis=1),axes=1))**2
    spectral=spectral.reshape(len(x),64,-1).sum(axis=2)
    spectral/=np.maximum(spectral.sum(axis=1,keepdims=True),1e-12)
    rows=[np.log10(np.maximum(spectral,1e-12)),np.quantile(amplitude,[.1,.25,.5,.75,.9],axis=1).T,
          np.mean(amplitude**4,axis=1,keepdims=True)]
    for lag in [1,2,4,8,16,32,64,128]:
        rows.append(np.abs(np.mean(x[:,lag:]*x[:,:-lag].conj(),axis=1,keepdims=True)))
    return np.concatenate(rows,axis=1).astype(np.float32)


def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,default=Path(__file__).parent/'benchmark_17')
    p.add_argument('--out-dir',type=Path,default=Path(__file__).parent/'runs/seed_20261029/random_forest')
    p.add_argument('--seed',type=int,default=20261029)
    args=p.parse_args();verify(args.dataset_root)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    arrays={key:np.load(args.dataset_root/(key+'.npy'),mmap_mode='r') for key in ['X','y','snr_db','channel_profile']}
    x=arrays['X'];f=np.empty((len(x),78),np.float32)
    for begin in range(0,len(x),512):
        f[begin:begin+512]=features(np.asarray(x[begin:begin+512]))
    split=np.load(args.dataset_root/'benchmark_split_indices.npz')
    model=RandomForestClassifier(n_estimators=200,max_depth=24,n_jobs=4,random_state=args.seed)
    print('TRAIN random_forest',flush=True)
    model.fit(f[split['train']],arrays['y'][split['train']])
    val=model.predict(f[split['val']]);idx=split['test'];pred=model.predict(f[idx]);target=arrays['y'][idx]
    np.savez_compressed(args.out_dir/'test_predictions.npz',pred=pred,target=target,index=idx,
                        snr_db=arrays['snr_db'][idx],channel_profile=arrays['channel_profile'][idx])
    metrics=dict(model='random_forest',test_samples=len(idx),accuracy=float(accuracy_score(target,pred)),
        balanced_accuracy=float(balanced_accuracy_score(target,pred)),macro_f1=float(f1_score(target,pred,average='macro')),
        weighted_f1=float(f1_score(target,pred,average='weighted')),best_val_acc=float(accuracy_score(arrays['y'][split['val']],val)))
    (args.out_dir/'metrics.json').write_text(json.dumps(metrics,indent=2))
    names=json.loads((args.dataset_root/'mode_names.json').read_text(encoding='utf-8'))
    profiles=json.loads((args.dataset_root/'channel_profiles.json').read_text(encoding='utf-8'))
    (args.out_dir/'config.json').write_text(json.dumps(dict(model='random_forest',mode_names=names,channel_profiles=profiles,
        seed=args.seed,feature_count=78,n_estimators=200,max_depth=24,
        validation_usage='One evaluation of the fixed classifier, not checkpoint selection',
        features='64 log normalized spectral bins, five amplitude quantiles, amplitude fourth moment, eight autocorrelation magnitudes'),indent=2))
    for field,file in [('snr_db','metrics_by_snr.csv'),('channel_profile','metrics_by_channel.csv')]:
        rows=[];values=arrays[field][idx]
        for value in np.unique(values):
            mask=values==value
            row={field:int(value),'samples':int(mask.sum()),'accuracy':float(accuracy_score(target[mask],pred[mask])),
                 'macro_f1':float(f1_score(target[mask],pred[mask],average='macro',zero_division=0))}
            if field=='channel_profile':row['channel_name']=profiles[int(value)]
            rows.append(row)
        write_csv(args.out_dir/file,rows)
    joblib.dump(model,args.out_dir/'model.joblib')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':main()
