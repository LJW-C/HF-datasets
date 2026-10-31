"""Export only completed and prediction-verified repeat/duration experiments."""
import csv
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
SEEDS=[20261029,20261030,20261031]
MODELS=['conformer','random_forest']


def family(name):
    if 'PSK' in name:return 'PSK'
    if name.startswith('MFSK'):return 'MFSK'
    return name.split('_')[0]


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def verified_row(run,dataset):
    metrics=json.loads((run/'metrics.json').read_text())
    cfg=json.loads((run/'config.json').read_text())
    pred=np.load(run/'test_predictions.npz')
    split=np.load(dataset/'benchmark_split_indices.npz')['test']
    key='index' if 'index' in pred.files else 'indices'
    assert np.array_equal(pred[key],split)
    y=np.load(dataset/'y.npy',mmap_mode='r')[split]
    assert np.array_equal(y,pred['target'])
    for metric,value in [('accuracy',accuracy_score(y,pred['pred'])),
                          ('macro_f1',f1_score(y,pred['pred'],average='macro',zero_division=0)),
                          ('balanced_accuracy',balanced_accuracy_score(y,pred['pred']))]:
        assert abs(value-metrics[metric])<1e-10,(run,metric)
    names=cfg['mode_names']
    yf=np.array([family(names[int(c)]) for c in y])
    pf=np.array([family(names[int(c)]) for c in pred['pred']])
    return dict(seed=cfg['seed'],accuracy=metrics['accuracy'],macro_f1=metrics['macro_f1'],
                family_accuracy=float(np.mean(yf==pf)),samples=len(y))


def aggregate(rows,keys):
    return {k+'_'+stat:float(getattr(np,stat)([r[k] for r in rows],**({'ddof':1} if stat=='std' else {})))
            for k in keys for stat in ['mean','std']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-only',action='store_true')
    args=parser.parse_args()
    out=HERE/'supplementary_experiments_20261030'
    out.mkdir(exist_ok=True)
    repeats=[];repeat_summary=[]
    for model in ([] if args.duration_only else MODELS):
        rows=[]
        for seed in SEEDS:
            base='runs' if seed==20261029 else 'repeat_runs_20261030'
            run=HERE/base/('seed_'+str(seed))/model
            row=dict(model=model,**verified_row(run,HERE/'benchmark_17'))
            rows.append(row);repeats.append(row)
        repeat_summary.append(dict(model=model,seeds=len(rows),**aggregate(rows,['accuracy','macro_f1','family_accuracy'])))
    if repeats:
        write(out/'repeat_metrics.csv',repeats)
        write(out/'repeat_summary.csv',repeat_summary)
    duration=[];duration_summary=[];snr_rows=[]
    for model in ['iq_cnn','random_forest']:
        for length in [2048,8192]:
            rows=[]
            for seed in SEEDS:
                run=HERE/'duration_runs_20261030'/str(length)/('seed_'+str(seed))/model
                row=dict(model=model,length=length,**verified_row(run,HERE/'paired_windows_20261030'/str(length)))
                rows.append(row);duration.append(row)
                prediction=np.load(run/'test_predictions.npz')
                names=json.loads((run/'config.json').read_text())['mode_names']
                true=prediction['target'];pred=prediction['pred']
                yf=np.array([family(names[int(i)]) for i in true])
                pf=np.array([family(names[int(i)]) for i in pred])
                for snr in sorted(set(prediction['snr_db'])):
                    mask=prediction['snr_db']==snr
                    snr_rows.append(dict(model=model,length=length,seed=seed,snr_db=int(snr),
                        samples=int(mask.sum()),accuracy=float(np.mean(true[mask]==pred[mask])),
                        family_accuracy=float(np.mean(yf[mask]==pf[mask]))))
            duration_summary.append(dict(model=model,length=length,seeds=len(rows),
                **aggregate(rows,['accuracy','macro_f1','family_accuracy'])))
    write(out/'duration_metrics.csv',duration)
    write(out/'duration_summary.csv',duration_summary)
    write(out/'duration_snr_metrics.csv',snr_rows)
    snr_summary=[]
    for model in ['iq_cnn','random_forest']:
        for length in [2048,8192]:
            for snr in [-10,-5,0,5,10,15,20,25]:
                subset=[r for r in snr_rows if r['model']==model and r['length']==length and r['snr_db']==snr]
                assert len(subset)==3
                snr_summary.append(dict(model=model,length=length,snr_db=snr,
                    **aggregate(subset,['accuracy','family_accuracy'])))
    write(out/'duration_snr_summary.csv',snr_summary)
    differences=[]
    for model in ['iq_cnn','random_forest']:
        a=[r for r in duration if r['model']==model and r['length']==2048]
        b=[r for r in duration if r['model']==model and r['length']==8192]
        deltas=[dict(seed=left['seed'],accuracy=right['accuracy']-left['accuracy'],
                     family_accuracy=right['family_accuracy']-left['family_accuracy']) for left,right in zip(a,b)]
        differences.append(dict(model=model,**aggregate(deltas,['accuracy','family_accuracy'])))
    write(out/'paired_duration_difference.csv',differences)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.2),layout='constrained')
    for ax,metric,title in zip(axes,['accuracy','family_accuracy'],['Four-configuration accuracy','Two-family accuracy']):
        for model,color,label in [('iq_cnn','#236078','IQ-CNN'),('random_forest','#b35240','Random Forest')]:
            rows=[r for r in duration_summary if r['model']==model]
            ax.errorbar([2048,8192],[100*r[metric+'_mean'] for r in rows],
                yerr=[100*r[metric+'_std'] for r in rows],marker='o',capsize=4,color=color,label=label)
        ax.set(title=title,xlabel='Samples at 6 kHz',ylabel='Accuracy (%)',ylim=(0,100),xticks=[2048,8192])
        ax.grid(alpha=.2);ax.axhline(25 if metric=='accuracy' else 50,color='gray',ls=':',lw=.8)
    axes[0].legend(fontsize=9)
    for ext in ['pdf','png']:fig.savefig(out/('paired_duration_results.'+ext),dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.3,3.2),layout='constrained',sharey=True)
    for ax,model,title in zip(axes,['iq_cnn','random_forest'],['IQ-CNN','Random Forest']):
        for length,color in [(2048,'#236078'),(8192,'#b35240')]:
            rows=[r for r in snr_summary if r['model']==model and r['length']==length]
            ax.errorbar([r['snr_db'] for r in rows],[100*r['accuracy_mean'] for r in rows],
                yerr=[100*r['accuracy_std'] for r in rows],marker='o',ms=3,capsize=2,
                color=color,label=str(length)+' points')
        ax.set(title=title,xlabel='Full-band SNR (dB)',ylim=(0,100),xticks=[-10,0,10,20])
        ax.grid(alpha=.2);ax.axhline(25,color='gray',ls=':',lw=.8)
    axes[0].set_ylabel('Four-configuration accuracy (%)');axes[0].legend(fontsize=9)
    for ext in ['pdf','png']:fig.savefig(out/('paired_duration_by_snr.'+ext),dpi=200)
    plt.close(fig)
    # This is a coarsening of the reference fine predictions, not a retrained head.
    run=HERE/'runs/seed_20261029/conformer'
    names=json.loads((run/'config.json').read_text())['mode_names']
    pred=np.load(run/'test_predictions.npz')
    families=sorted({family(n) for n in names})
    target=np.array([families.index(family(names[int(c)])) for c in pred['target']])
    predicted=np.array([families.index(family(names[int(c)])) for c in pred['pred']])
    cm=np.zeros((len(families),len(families)),dtype=int)
    np.add.at(cm,(target,predicted),1)
    proportions=cm/cm.sum(axis=1,keepdims=True)
    write(out/'reference_family_confusion_counts.csv',[
        dict(true_family=fam,**{f:int(v) for f,v in zip(families,row)}) for fam,row in zip(families,cm)])
    (out/'family_mapping.json').write_text(json.dumps(dict(mode_to_family={n:family(n) for n in names},
        reference_seed=20261029,family_accuracy=float(np.trace(cm)/cm.sum()),
        interpretation='Coarsened fine-model predictions, not a separately trained family classifier.'),indent=2))
    fig,ax=plt.subplots(figsize=(6.2,5.2),layout='constrained')
    image=ax.imshow(100*proportions,cmap='Blues',vmin=0,vmax=100)
    ax.set(xticks=range(len(families)),yticks=range(len(families)),xlabel='Predicted family',ylabel='True family')
    ax.set_xticklabels(families,rotation=45,ha='right');ax.set_yticklabels(families)
    for i in range(len(families)):
        for j in range(len(families)):
            if i==j or proportions[i,j]>=.03:
                ax.text(j,i,f'{100*proportions[i,j]:.1f}',ha='center',va='center',fontsize=9,
                    color='white' if proportions[i,j]>.5 else 'black')
    fig.colorbar(image,ax=ax,label='Row percentage (%)')
    ax.set_title('Conformer: coarsened reference predictions')
    for ext in ['pdf','png']:fig.savefig(out/('reference_family_confusion.'+ext),dpi=200)
    plt.close(fig)
    report=dict(status='duration_complete_repeats_pending' if args.duration_only else 'passed',
        repeat_runs_checked=len(repeats),duration_runs_checked=len(duration),
        seeds=SEEDS,sd_definition='Sample standard deviation across training seeds (ddof=1).',
        main_benchmark='Fixed 17-class raw-time-disjoint split; Conformer and Random Forest repeated only.',
        diagnostic='Separate paired four-class duration study, IQ-CNN and Random Forest, 12800 vectors per length, test n=1920.',
        limitations='Three seeds measure training variability, not independent recordings, payloads or channel populations. No architecture-wide significance claim.')
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(repeats=repeat_summary,duration=duration_summary,differences=differences),indent=2))


if __name__=='__main__':main()
