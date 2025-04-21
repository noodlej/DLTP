#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_hparam_sweep.py: Randomly sample hyperparameters from specified distributions
and print out train_stage1_sweep.py commands without executing.
"""
import argparse
import random
import os


def sample_lrs(n: int):
    """Sample learning rates from 10^Uniform(-5, -3.5)"""
    return [10 ** random.uniform(-5, -3.5) for _ in range(n)]


def sample_etas(n: int):
    """Sample groupDRO_eta from 10^Uniform(-3, -1)"""
    return [10 ** random.uniform(-3, -1) for _ in range(n)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate sweep commands for train_stage1_sweep.py')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                        help='Random seeds for sweep')
    parser.add_argument('--num-lrs', type=int, default=2,
                        help='Number of learning rates to sample')
    parser.add_argument('--num-etas', type=int, default=2,
                        help='Number of groupDRO_eta to sample')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # Sample hyperparameters
    lr_list = sample_lrs(args.num_lrs)
    eta_list = sample_etas(args.num_etas)

    # Format for CLI
    lr_strs = [f'{lr:.1e}' for lr in lr_list]
    eta_strs = [f'{eta:.1e}' for eta in eta_list]
    seed_strs = [str(s) for s in args.seeds]

    cmd = ['python', 'train_stage1_sweep.py',
           '--seeds', *seed_strs,
           '--lrs', *lr_strs,
           '--etas', *eta_strs]

    print('Generated command:')
    print(' '.join(cmd)) 