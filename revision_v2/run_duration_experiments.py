"""Run fixed-protocol duration controls without overwriting previous experiments."""
from pathlib import Path
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
SEEDS = [20261029,20261030,20261031]


def classical():
    tasks=[]
    for seed in SEEDS[1:]:
        tasks.append((HERE/'benchmark_17',HERE/'repeat_runs_20261030'/('seed_'+str(seed))/'random_forest',seed))
    for length in [2048,8192]:
        for seed in SEEDS:
            tasks.append((HERE/'paired_windows_20261030'/str(length),
                          HERE/'duration_runs_20261030'/str(length)/('seed_'+str(seed))/'random_forest',seed))
    for dataset,out,seed in tasks:
        out.parent.mkdir(parents=True,exist_ok=True)
        with (out.parent/'random_forest_console.log').open('w',encoding='utf-8') as stream:
            print('RF',dataset.name,seed,flush=True)
            subprocess.run([sys.executable,str(HERE/'train_classical_baseline.py'),'--dataset-root',str(dataset),
                            '--out-dir',str(out),'--seed',str(seed)],stdout=stream,stderr=subprocess.STDOUT,check=True)


def neural():
    for length in [2048,8192]:
        subprocess.run([sys.executable,str(HERE/'train_suite.py'),
            '--dataset-root',str(HERE/'paired_windows_20261030'/str(length)),
            '--out-dir',str(HERE/'duration_runs_20261030'/str(length)),
            '--models','iq_cnn','--seeds',','.join(map(str,SEEDS)),'--jobs','1'],check=True)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(classical),pool.submit(neural)]
        for task in futures:
            task.result()
    (HERE/'duration_runs_20261030/experiment_status.json').write_text(json.dumps(dict(
        status='complete',seeds=SEEDS,lengths=[2048,8192],models=['iq_cnn','random_forest'],
        protocol='Paired four-class, 400 vectors per mode/SNR; 30 epochs for IQ-CNN.'),indent=2))
