#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage‑1 sweep script leveraging train_stage1.py"""

import os
import sys
import subprocess
import argparse
import json
import shutil
import time


def evaluate_worst_val(output_dir: str) -> float:
    """Compute worst‑group validation accuracy from results.jsonl."""
    results_file = os.path.join(output_dir, 'results.jsonl')
    if not os.path.exists(results_file):
        raise FileNotFoundError(f'Results file not found: {results_file}')
    with open(results_file, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f'No records found in {results_file}')
    latest = max(records, key=lambda r: r.get('step', -1))
    val_accs = [v for k, v in latest.items() if k.startswith('val') and k.endswith('_acc')]
    if not val_accs:
        raise ValueError(f'No validation group accuracies in record: {list(latest.keys())}')
    return min(val_accs)


def main():
    parser = argparse.ArgumentParser(description='Stage‑1 sweep using train_stage1.py')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0,1,2],
                        help='Random seeds for sweep')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training steps')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate override')
    parser.add_argument('--output_root', type=str, default='outputs',
                        help='Root directory for train_stage1.py outputs')
    parser.add_argument('--save_dir', type=str, default='/data2/yws_ids25/data/AIDA',
                        help='Directory to save best models')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='학습 배치 사이즈 설정(train_stage1.py로 전달)')
    parser.add_argument('--lrs', type=float, nargs='+', default=None,
                        help='학습률 리스트 (sweep 대상; 생략 시 --lr 사용)')
    parser.add_argument('--etas', type=float, nargs='+', default=None,
                        help='groupDRO_eta 리스트 (sweep 대상; 생략 시 기본값)')
    args = parser.parse_args()

    # build hyperparam lists for sweep
    lr_list = args.lrs if args.lrs is not None else [args.lr]
    eta_list = args.etas if args.etas is not None else [None]

    script_dir = os.path.abspath(os.path.dirname(__file__))
    train_script = os.path.join(script_dir, 'train_stage1.py')

    os.makedirs(args.save_dir, exist_ok=True)
    best_bird = -1.0
    best_bg = -1.0

    # sweep over seeds, learning rates, and groupDRO_eta
    for seed in args.seeds:
        for lr in lr_list:
            for eta in eta_list:
                combo_root = os.path.join(args.output_root, f'seed{seed}_lr{lr}_eta{eta}')
                os.makedirs(combo_root, exist_ok=True)
                # launch training
                cmd = [
                    sys.executable, train_script,
                    '--seed', str(seed),
                    '--epochs', str(args.epochs),
                    '--lr', str(lr),
                    '--batch_size', str(args.batch_size),
                    '--output_root', combo_root
                ]
                if eta is not None:
                    cmd += ['--groupdro_eta', str(eta)]
                print(f'Running sweep: seed={seed}, lr={lr}, eta={eta}')
                print('CMD:', ' '.join(cmd))
                subprocess.run(cmd, check=True)

                # evaluate bird model
                bird_dir = os.path.join(combo_root, f'model_bird_seed{seed}')
                results_bird = os.path.join(bird_dir, 'results.jsonl')
                if not os.path.exists(results_bird):
                    raise FileNotFoundError(f'No results.jsonl for bird at {results_bird}')
                bird_val = evaluate_worst_val(bird_dir)
                print(f'bird-best seed={seed}, lr={lr}, eta={eta}, acc={bird_val:.4f}')
                if bird_val > best_bird:
                    best_bird = bird_val
                    shutil.copy(
                        os.path.join(bird_dir, 'model.pkl'),
                        os.path.join(args.save_dir, 'model_bird.pkl')
                    )
                    print("-------------------------------------------------------------------------")
                    print(f'Updated best bird: {best_bird:.4f} (seed={seed}, lr={lr}, eta={eta})')
                    print("-------------------------------------------------------------------------")

                # evaluate background model
                bg_dir = os.path.join(combo_root, f'model_background_seed{seed}')
                results_bg = os.path.join(bg_dir, 'results.jsonl')
                if not os.path.exists(results_bg):
                    raise FileNotFoundError(f'No results.jsonl for background at {results_bg}')
                bg_val = evaluate_worst_val(bg_dir)
                print(f'bg-best seed={seed}, lr={lr}, eta={eta}, acc={bg_val:.4f}')
                if bg_val > best_bg:
                    best_bg = bg_val
                    shutil.copy(
                        os.path.join(bg_dir, 'model.pkl'),
                        os.path.join(args.save_dir, 'model_background.pkl')
                    )
                    print("-------------------------------------------------------------------------")
                    print(f'Updated best background: {best_bg:.4f} (seed={seed}, lr={lr}, eta={eta})')
                    print("-------------------------------------------------------------------------")
                # remove temporary outputs
                shutil.rmtree(combo_root, ignore_errors=True)


if __name__ == '__main__':
    main() 