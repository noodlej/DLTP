#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_stage1.py: Stage-1 학습된 모델을 불러와 Waterbirds test set 각 서브그룹별 정확도를 계산합니다.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from torchvision import transforms
from wilds import get_dataset

# DomainBed 알고리즘 import을 위해 NoiseRobustDG 경로 설정
sys.path.insert(0, os.path.abspath("NoiseRobustDG"))
from domainbed import algorithms

def load_model(model_path: str, device: torch.device):
    """저장된 모델을 불러와 Algorithm 객체로 반환합니다."""
    save_dict = torch.load(model_path, map_location=device)
    alg_name = save_dict["args"]["algorithm"]
    input_shape = save_dict["model_input_shape"]
    num_classes = save_dict["model_num_classes"]
    num_domains = save_dict["model_num_domains"]
    hparams = save_dict["model_hparams"]
    AlgClass = algorithms.get_algorithm_class(alg_name)
    model = AlgClass(input_shape, num_classes, num_domains, hparams)
    # GroupDRO buffer 'q' may mismatch shape; remove before loading
    state_dict = save_dict["model_dict"].copy()
    if "q" in state_dict:
        state_dict.pop("q")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def evaluate_subgroup(model, device, raw_ds, indices, label_type: str):
    """
    특정 서브그룹 인덱스(indices)에 대해 모델의 정확도를 계산합니다.
    label_type: "y" 또는 "background"
    """
    meta = raw_ds.metadata_array
    if hasattr(meta, "numpy"):
        meta = meta.numpy()
    field_idx = raw_ds.metadata_fields.index(label_type)
    labels = meta[:, field_idx].astype(int)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    correct = 0
    total = len(indices)
    with torch.no_grad():
        for idx in indices:
            img = raw_ds.get_input(idx)
            x = preprocess(img).unsqueeze(0).to(device)
            logits = model.predict(x)
            pred = logits.argmax(dim=1).item()
            if pred == labels[idx]:
                correct += 1
    return correct / total if total > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(
        description="Stage-1 학습된 모델을 불러와 Waterbirds test set 서브그룹별 평가를 수행합니다."
    )
    parser.add_argument(
        "--data_root", type=str,
        default="/data2/yws_ids25/data/AIDA/data",
        help="Waterbirds 데이터의 루트 디렉토리"
    )
    parser.add_argument(
        "--model_dir", type=str,
        default="/data2/yws_ids25/data/AIDA",
        help="model_bird.pkl 및 model_background.pkl이 있는 디렉토리"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Raw Waterbirds 데이터셋 로드
    raw_ds = get_dataset(dataset="waterbirds", download=False, root_dir=args.data_root)

    # 테스트 인덱스 추출
    split_arr = raw_ds.split_array
    if hasattr(split_arr, "numpy"):
        split_arr = split_arr.numpy()
    test_idx = np.where(split_arr == raw_ds.split_dict["test"])[0]

    # 메타데이터 추출 (y, background)
    meta = raw_ds.metadata_array
    if hasattr(meta, "numpy"):
        meta = meta.numpy()
    y_all = meta[:, raw_ds.metadata_fields.index("y")].astype(int)
    bg_all = meta[:, raw_ds.metadata_fields.index("background")].astype(int)

    # 서브그룹 정의
    subgroups = {
        "landbird_land": (0, 0),
        "landbird_water": (0, 1),
        "waterbird_land": (1, 0),
        "waterbird_water": (1, 1),
    }
    subgroup_indices = {}
    for name, (y_val, bg_val) in subgroups.items():
        inds = [i for i in test_idx if y_all[i] == y_val and bg_all[i] == bg_val]
        subgroup_indices[name] = inds

    # 모델 로드
    bird_model = load_model(os.path.join(args.model_dir, "model_bird.pkl"), device)
    bg_model   = load_model(os.path.join(args.model_dir, "model_background.pkl"), device)

    # 평가 및 결과 출력
    results = {}
    for name in subgroups.keys():
        inds = subgroup_indices[name]
        bird_acc = evaluate_subgroup(bird_model, device, raw_ds, inds, "y")
        bg_acc   = evaluate_subgroup(bg_model, device, raw_ds, inds, "background")
        results[name] = {"bird_acc": bird_acc, "bg_acc": bg_acc}
        print(f"{name}: bird_acc={bird_acc:.4f}, background_acc={bg_acc:.4f}")

    # 전체 결과 저장
    with open(os.path.join(args.model_dir, "test_stage1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
