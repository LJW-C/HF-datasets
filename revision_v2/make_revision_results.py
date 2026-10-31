"""Export measured new results and recording QC; never substitutes old scores."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent/'python_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

DISPLAY={'iq_cnn':'IQ-CNN','conformer':'Conformer','resnet1d':'ResNet-1D','cldnn':'CLDNN',
         'tcn':'TCN','transformer':'Transformer','random_forest':'Random Forest'}


def write_csv(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,default=Path(__file__).parent/'raw_verified')
    p.add_argument('--runs',type=Path,default=Path(__file__).parent/'runs')
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    p.add_argument('--dataset-root',type=Path,default=Path(__file__).parent/'benchmark_17')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    recordings=[]
    for file in sorted(args.raw_root.glob('*/recording.json')):
        r=json.loads(file.read_text(encoding='utf-8'))
        recordings.append(dict(mode=r['mode_name'],actual_modem=r['readback']['actual_modem'],
            carrier_hz=r['readback']['carrier_hz'],txid=r['readback']['txid'],
            duration_s=r['duration_sec'],audio_fs=r['record_fs'],obw99_hz=r['qc']['obw99_hz'],
            passed=r['qc']['passed'],source_sha256=r['sha256']))
    write_csv(args.out/'recording_quality.csv',recordings)
    metrics=[];snrs=[];profiles=[];families=[];classes=[]
    for file in sorted(args.runs.glob('seed_*/*/metrics.json')):
        row=json.loads(file.read_text());row['seed']=file.parent.parent.name
        metrics.append(row)
        for name,target in [('metrics_by_snr.csv',snrs),('metrics_by_channel.csv',profiles)]:
            with (file.parent/name).open(encoding='utf-8') as f:
                for item in csv.DictReader(f):target.append(dict(model=file.parent.name,seed=row['seed'],**item))
        cfg=json.loads((file.parent/'config.json').read_text(encoding='utf-8'))
        names=cfg['mode_names'];predictions=np.load(file.parent/'test_predictions.npz')
        y=predictions['target'];pred=predictions['pred']
        def family(name):
            if 'PSK' in name:return 'PSK'
            if name.startswith('MFSK'):return 'MFSK'
            return name.split('_')[0]
        yf=np.array([family(names[int(i)]) for i in y])
        pf=np.array([family(names[int(i)]) for i in pred])
        for fam in sorted(set(yf)):
            ix=yf==fam
            families.append(dict(model=file.parent.name,seed=row['seed'],family=fam,samples=int(ix.sum()),
                family_accuracy=float(np.mean(yf[ix]==pf[ix])),fine_class_accuracy=float(np.mean(y[ix]==pred[ix]))))
        for cls,name in enumerate(names):
            ix=y==cls
            classes.append(dict(model=file.parent.name,seed=row['seed'],mode=name,samples=int(ix.sum()),recall=float(np.mean(y[ix]==pred[ix]))))
    write_csv(args.out/'model_metrics.csv',metrics)
    write_csv(args.out/'metrics_by_snr.csv',snrs)
    write_csv(args.out/'metrics_by_profile.csv',profiles)
    write_csv(args.out/'metrics_by_family.csv',families)
    write_csv(args.out/'metrics_by_class.csv',classes)
    if (args.dataset_root/'build_status.json').exists() and json.loads((args.dataset_root/'build_status.json').read_text())['status']=='complete':
        names=json.loads((args.dataset_root/'mode_names.json').read_text(encoding='utf-8'))
        arrays={key:np.load(args.dataset_root/(key+'.npy'),mmap_mode='r') for key in ['X','y','snr_db','channel_profile','split_id']}
        selected=['BPSK31','RTTY','MFSK16','DominoEX_8']
        fig,axes=plt.subplots(4,4,figsize=(17,12),constrained_layout=True)
        examples=[]
        for row,name in enumerate(selected):
            ix=np.flatnonzero((arrays['y']==names.index(name))&(arrays['snr_db']==25)&(arrays['channel_profile']==0)&(arrays['split_id']==2))
            index=int(ix[0]);x=arrays['X'][index];t=np.arange(len(x))/6000
            axes[row,0].plot(t,x.real,lw=.7,label='I');axes[row,0].plot(t,x.imag,lw=.7,label='Q');axes[row,0].set_ylabel(name+'\nAmplitude');axes[row,0].legend(fontsize=9)
            axes[row,1].scatter(x.real,x.imag,s=2,alpha=.35);axes[row,1].set_aspect('equal',adjustable='box')
            f,psd=signal.welch(x,6000,nperseg=1024,return_onesided=False)
            axes[row,2].plot(np.fft.fftshift(f),10*np.log10(np.maximum(np.fft.fftshift(psd),1e-20)))
            axes[row,2].set(xlim=(-3000,3000),ylabel='PSD (dB/Hz)')
            f,tt,z=signal.stft(x,6000,nperseg=512,noverlap=480,boundary=None,return_onesided=False)
            axes[row,3].pcolormesh(tt,np.fft.fftshift(f),20*np.log10(np.maximum(np.abs(np.fft.fftshift(z,axes=0)),1e-10)),shading='auto',rasterized=True)
            axes[row,3].set(ylim=(-700,700),ylabel='Frequency (Hz)')
            examples.append(dict(mode=name,index=index,split='test',profile='good',snr_db=25))
        for ax,title in zip(axes[0],['I/Q waveforms','Unsynchronized I/Q scatter','Full-band spectrum','Spectrogram: central +/-700 Hz']):ax.set_title(title,fontsize=12)
        for col,label in enumerate(['Time (s)','In-phase','Frequency (Hz)','Time (s)']):axes[-1,col].set_xlabel(label)
        fig.savefig(args.out/'verified_examples.png',dpi=180);fig.savefig(args.out/'verified_examples.pdf');plt.close(fig)
        (args.out/'verified_examples.json').write_text(json.dumps(examples,indent=2),encoding='utf-8')
    if snrs:
        fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
        for model in sorted(set(r['model'] for r in snrs)):
            rows=sorted([r for r in snrs if r['model']==model],key=lambda r:float(r['snr_db']))
            values=sorted(set(float(r['snr_db']) for r in rows))
            acc=[np.mean([float(r['accuracy']) for r in rows if float(r['snr_db'])==s])*100 for s in values]
            ax.plot(values,acc,marker='o',label=DISPLAY.get(model,model))
        ax.set(xlabel='Full-band SNR (dB)',ylabel='Test accuracy (%)',ylim=(0,100),xticks=[-10,-5,0,5,10,15,20,25])
        ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(14)
        ax.legend(fontsize=11);ax.grid(alpha=.25);ax.tick_params(labelsize=12)
        fig.savefig(args.out/'snr_accuracy.png',dpi=200);fig.savefig(args.out/'snr_accuracy.pdf');plt.close(fig)
    if metrics:
        best=max(metrics,key=lambda r:r['accuracy'])
        model_dir=args.runs/best['seed']/best['model']
        cfg=json.loads((model_dir/'config.json').read_text(encoding='utf-8'))
        names=cfg['mode_names'];p=np.load(model_dir/'test_predictions.npz')
        cm=np.zeros((len(names),len(names)),dtype=np.int64)
        np.add.at(cm,(p['target'],p['pred']),1)
        normalized=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
        fig,ax=plt.subplots(figsize=(12,10),constrained_layout=True)
        im=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
        fig.colorbar(im,ax=ax,label='Row-normalized proportion')
        ax.set_xticks(np.arange(len(names)));ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names,rotation=50,ha='right',fontsize=11)
        ax.set_yticklabels(names,fontsize=11)
        ax.set_xlabel('Predicted configuration',fontsize=14);ax.set_ylabel('True configuration',fontsize=14)
        ax.set_title(DISPLAY.get(best['model'],best['model'])+' on the time-disjoint test split',fontsize=15)
        for i in range(len(names)):
            for j in range(len(names)):
                value=normalized[i,j]
                if i==j or value>=.05:
                    ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=9,color='white' if value>.5 else 'black')
        fig.savefig(args.out/'confusion_matrix_best.png',dpi=200);fig.savefig(args.out/'confusion_matrix_best.pdf');plt.close(fig)
        np.savetxt(args.out/'confusion_matrix_best.csv',cm,fmt='%d',delimiter=',')
    summary=dict(recorded_modes=len(recordings),qc_passed=sum(r['passed'] for r in recordings),
                 finished_model_runs=len(metrics),metrics=metrics,
                 note='Only completed v2 runs are included. Sanity runs and historical scores are excluded.')
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
