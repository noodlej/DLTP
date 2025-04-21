#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
train_stage1.py: Stage‑1 모델을 GroupDRO로 학습합니다.
- model_bird: 새 레이블(y) 예측
- model_background: 배경(background) 예측
Usage:
    python train_stage1.py
'''
import os
import json
import subprocess
import argparse
import sys
import runpy
import importlib.util
import torch.utils.data.dataloader as _dataloader
import shutil
import time
from wilds import get_dataset
from collections import Counter
import numpy as np
import glob

# DomainBed 패키지를 찾기 위해 NoiseRobustDG 경로를 추가
sys.path.insert(0, os.path.abspath('NoiseRobustDG'))
sys.path.insert(0, os.path.abspath('.'))

# custom dataset adapter: DomainBed에 WILDSWaterbirdsBG 등록 (importlib 사용)
import domainbed.datasets as db_datasets  # noqa: E402
adapter_path = os.path.join(os.path.dirname(__file__), 'aida', 'datasets.py')
spec = importlib.util.spec_from_file_location('aida_datasets', adapter_path)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
db_datasets.WILDSWaterbirdsBG = adapter.WILDSWaterbirdsBG
# Bird prediction adapter: download=True를 설정한 WILDSWaterbirds also override
db_datasets.WILDSWaterbirds = adapter.WILDSWaterbirds

# CLI 인자 파싱: epochs(steps), learning rate, seed, output_root 추가
parser = argparse.ArgumentParser(description='Stage‑1 GroupDRO 학습 스크립트')
parser.add_argument('--epochs', type=int, default=2000, help='DomainBed 훈련 스텝 수 (기본: 2000)')
parser.add_argument('--lr', type=float, default=5e-5, help='학습률 override (기본: 5e-5)')
parser.add_argument('--seed', type=int, default=0, help='랜덤 시드 (기본: 0)')
parser.add_argument('--output_root', type=str, default='outputs', help='출력 디렉토리 루트 (기본: outputs)')
parser.add_argument('--trial_seed', type=int, default=0, help='Validation split 시드 고정 (기본: 0)')
parser.add_argument('--batch_size', type=int, default=32, help='학습 배치 사이즈 설정 (기본: 32)')
parser.add_argument('--groupdro_eta', type=float, default=None, help='GroupDRO η 하이퍼파라미터 override')
parser.add_argument('--weight_decay', type=float, default=None, help='Weight decay override')
args = parser.parse_args()
EPOCHS = args.epochs
LR = args.lr
SEED = args.seed
TRIAL_SEED = args.trial_seed
OUTPUT_ROOT = args.output_root
BATCH_SIZE = args.batch_size
GROUPDRO_ETA = args.groupdro_eta
WEIGHT_DECAY = args.weight_decay

# 설정
# aida.py를 통해 Waterbirds 데이터를 지정된 경로에 다운로드합니다.
DATA_ROOT = '/data2/yws_ids25/data/AIDA/data'
# DomainBed에는 root 디렉터리(aida/data)를 전달합니다
DATA_DIR = DATA_ROOT
DOMAINBED_TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), 'NoiseRobustDG', 'domainbed', 'scripts', 'train.py')
HOLDOUT_FRACTION = 0.2  # train:val = 8:2
TEST_ENVS = [4, 5]      # wilds의 validation(4) 및 test(5) 그룹
ALGORITHM = 'GroupDRO'
TASK = 'domain_generalization'

# MultiProcessingDataLoaderIter destructor override to ignore child process assertion
_dataloader._MultiProcessingDataLoaderIter.__del__ = lambda self: None

def select_best_model(output_dir: str):
    """Select best checkpoint by worst-group validation accuracy and save as model.pkl."""
    results_file = os.path.join(output_dir, 'results.jsonl')
    if not os.path.exists(results_file):
        print(f"[select_best_model] No results.jsonl found: {results_file}")
        return
    with open(results_file, 'r') as f:
        recs = [json.loads(line) for line in f if line.strip()]
    best_val = -1.0
    best_step = None
    for rec in recs:
        val_accs = [v for k, v in rec.items() if k.startswith('val') and k.endswith('_acc')]
        if not val_accs:
            continue
        worst = min(val_accs)
        if worst > best_val:
            best_val = worst
            best_step = rec.get('step')
    if best_step is None:
        print(f"[select_best_model] No valid val_acc in {output_dir}")
        return
    # 최종 model.pkl로 복사: model_best.pkl 우선, 그 외 model_step 파일
    dst = os.path.join(output_dir, 'model.pkl')
    best_model = os.path.join(output_dir, 'model_best.pkl')
    step_model = os.path.join(output_dir, f"model_step{best_step}.pkl")
    if os.path.exists(best_model):
        shutil.copy(best_model, dst)
        print(f"[select_best_model] Copied best-only checkpoint: {best_model}")
    elif os.path.exists(step_model):
        shutil.copy(step_model, dst)
        print(f"[select_best_model] Copied step checkpoint: {step_model}")
    else:
        print(f"[select_best_model] Warning: No checkpoint found for best step={best_step}")
        return
    # 중간 checkpoint 정리
    for ckpt in glob.glob(os.path.join(output_dir, 'model_*.pkl')):
        if ckpt != dst:
            os.remove(ckpt)

def run_training(dataset_name: str, output_dir: str):
    """
    DomainBed CLI를 호출하여 지정한 데이터셋을 학습합니다.
    """
    # train set의 subgroup 개수 출력 (y=bird, background)
    raw_ds = get_dataset(dataset='waterbirds', download=False, root_dir=DATA_ROOT)
    meta = raw_ds.metadata_array
    if hasattr(meta, 'numpy'):
        meta = meta.numpy()
    split_arr = raw_ds.split_array
    split_id = raw_ds.split_dict['train']
    split_np = split_arr.numpy() if hasattr(split_arr, 'numpy') else split_arr
    train_idx = np.where(split_np == split_id)[0]
    train_meta = meta[train_idx]
    y_i = raw_ds.metadata_fields.index('y')
    bg_i = raw_ds.metadata_fields.index('background')
    counts = Counter([(int(r[y_i]), int(r[bg_i])) for r in train_meta])
    print(f"[run_training][{dataset_name}] Train subgroup counts (y, background): {counts}")
    os.makedirs(output_dir, exist_ok=True)
    # 하이퍼파라미터 설정 (holdout_fraction, batch_size, learning rate)
    hparams = {"holdout_fraction": HOLDOUT_FRACTION, "batch_size": BATCH_SIZE}
    if LR is not None:
        hparams['lr'] = LR
    if GROUPDRO_ETA is not None:
        hparams['groupdro_eta'] = GROUPDRO_ETA
    if WEIGHT_DECAY is not None:
        hparams['weight_decay'] = WEIGHT_DECAY
    args_list = [
        '--data_dir', DATA_DIR,
        '--dataset', dataset_name,
        '--algorithm', ALGORITHM,
        '--task', TASK,
        '--hparams', json.dumps(hparams),
        '--hparams_seed', str(SEED),
        '--trial_seed', str(TRIAL_SEED),
        '--seed', str(SEED),
        '--test_envs', *[str(te) for te in TEST_ENVS],
        '--output_dir', output_dir,
        '--num_workers', '0',
        '--save_model_every_checkpoint',  # checkpoint마다 기록하지 않고 best만 저장
    ]
    if EPOCHS is not None:
        args_list.extend(['--steps', str(EPOCHS)])
    # AIDA 어댑터 등록 후 DomainBed train 실행(wrapper script via -m)
    project_root = os.path.abspath(os.path.dirname(__file__))
    noise_robust_dg_dir = os.path.join(project_root, 'NoiseRobustDG')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([noise_robust_dg_dir, project_root, env.get('PYTHONPATH', '')])
    cmd = [sys.executable, '-m', 'aida.train_domainbed', *args_list]
    print(f"[run_training] CMD >> {' '.join(cmd)} (cwd={project_root})")
    subprocess.run(cmd, cwd=project_root, env=env, check=True)
    # 도메인베드 스크립트 내부 best-only 로직에 따라 model_best.pkl이 생성됨
    # 최종 model.pkl 선택 및 전달
    select_best_model(output_dir)

def main():
    # 0) 데이터 다운로드
    subprocess.run(['python', 'aida.py'], check=True)
    # Stage‑1 학습: 시드 기반 경로에 저장
    bird_dir = os.path.join(OUTPUT_ROOT, f'model_bird_seed{SEED}')
    run_training('WILDSWaterbirds', bird_dir)
    bg_dir = os.path.join(OUTPUT_ROOT, f'model_background_seed{SEED}')
    run_training('WILDSWaterbirdsBG', bg_dir)

if __name__ == '__main__':
    main() 