"""Train existing baselines only on a successfully verified predefined split."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from verify_dataset import verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,default=Path(__file__).parent/'runs')
    p.add_argument('--models',default='conformer,iq_cnn,resnet1d,cldnn,tcn,transformer')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--seeds',default='20261029')
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--device',choices=['cuda','cpu','auto'],default='cuda')
    args=p.parse_args()
    if args.jobs < 1: p.error('--jobs must be positive')
    verify(args.dataset_root)
    import torch
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'training_environment.json').write_text(json.dumps(dict(
        python=sys.version,pytorch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        jobs=args.jobs,omp_threads=4,seeds=args.seeds,epochs=args.epochs,requested_device=args.device,
        split='predefined source-time disjoint',dataset_root=str(args.dataset_root.resolve())),indent=2),encoding='utf-8')
    repository=Path(__file__).resolve().parents[1]
    if not (repository/'train_rf_classifier.py').is_file():
        repository=repository/'hf-digital-mode-iq-dataset-github'
    env=os.environ.copy()
    env['PYTHONPATH']=str(Path(__file__).parent/'python_packages')+os.pathsep+str(repository)
    env['MPLBACKEND']='Agg'
    env['PYTHONIOENCODING']='utf-8'
    env['OMP_NUM_THREADS']='4'
    env['MKL_NUM_THREADS']='4'
    tasks=[]
    for seed in args.seeds.split(','):
        out=args.out_dir/('seed_'+seed)
        for model in args.models.split(','):
            model_out=out/model
            if model_out.exists():raise FileExistsError('Refusing to overwrite a training run: '+str(model_out))
            out.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,str(repository/'train_rf_classifier.py'),
                     '--dataset-root',str(args.dataset_root.resolve()),'--out-dir',str(out.resolve()),
                     '--model',model,'--seed',seed,'--epochs',str(args.epochs),
                     '--device',args.device,'--batch-size','128','--num-workers','0']
            tasks.append((model,seed,out,command))
    def run(task):
        model,seed,out,command=task
        log_path=out/(model+'_console.log')
        try:
            with log_path.open('w',encoding='utf-8') as log:
                print('TRAIN',model,seed,flush=True)
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        except subprocess.CalledProcessError as exc:
            tail='\n'.join(log_path.read_text(encoding='utf-8',errors='replace').splitlines()[-35:])
            raise RuntimeError('Training failed; log: '+str(log_path)+'\n'+tail) from exc
        print('FINISHED',model,seed,flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(run,t) for t in tasks]
        for future in as_completed(futures):future.result()


if __name__=='__main__':main()
