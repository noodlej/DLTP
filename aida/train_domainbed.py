#!/usr/bin/env python3
import os
import sys
import runpy

# Ensure project root on sys.path to import aida.datasets
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# Ensure NoiseRobustDG on sys.path to import domainbed package
noise_robust_dg_dir = os.path.join(project_root, 'NoiseRobustDG')
sys.path.insert(0, noise_robust_dg_dir)

# Register AIDA adapters into domainbed.datasets
import domainbed.datasets as db_datasets  # noqa: E402
from aida.datasets import WILDSWaterbirdsBG, WILDSWaterbirds  # noqa: E402

db_datasets.WILDSWaterbirdsBG = WILDSWaterbirdsBG
db_datasets.WILDSWaterbirds = WILDSWaterbirds

if __name__ == '__main__':
    # Delegate to DomainBed's train module
    runpy.run_module('domainbed.scripts.train', run_name='__main__') 