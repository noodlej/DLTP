#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIDA (Attribute Invariant Data Augmentation) 파이프라인:
1) Waterbirds trainset에서 (waterbird, land background) 그룹 제외
2) GroupDRO로 model_bird, model_background 학습
3) 샘플을 이용한 saliency map 계산
4) PuzzleMixup을 이용하여 synthetic images 생성
5) synthetic 이미지를 새로운 그룹 (waterbird, land background)으로 라벨링하여 trainset에 추가하고 GroupDRO 학습
6) test 데이터의 각 subgroup별 정확도 계산
"""
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import torchvision
import torchvision.transforms as T
from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
from PIL import Image

# 이미지 변환 정의 (PIL->Tensor)
img_transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
])

# Waterbirds raw dataset을 Torch Dataset으로 래핑하여 Tensor를 반환
class WaterbirdsTorchDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, indices, transform, mode='bird'):
        self.raw = raw_dataset
        self.indices = list(indices)
        self.transform = transform
        self.mode = mode
        self.bg_i = self.raw.metadata_fields.index('background')
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x = self.raw.get_input(real_idx)
        if isinstance(x, np.ndarray):
            x = Image.fromarray(x)
        x = self.transform(x)
        if not isinstance(x, torch.Tensor):
            x = T.ToTensor()(x)
        if self.mode == 'bird':
            y = self.raw.y_array[real_idx]
        else:
            y = self.raw.metadata_array[real_idx, self.bg_i]
        # label을 Tensor로 변환
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# PuzzleMixup 경로 설정
sys.path.append(os.path.join(os.path.dirname(__file__), 'PuzzleMix'))
from mixup import mixup_graph

# 설정 영역: 환경에 맞게 수정 (Windows 호환 동적 경로 설정)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, 'data', 'waterbirds')  # Waterbirds 원본 데이터 경로
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE= 32
LR         = 1e-4
DRO_ETA    = 1.0
N_STEPS    = 100   # 각 GroupDRO 학습 스텝 수

# PuzzleMix 하이퍼파라미터
BETA       = 0.5
GAMMA      = 0.5
ETA        = 1.0
NEIGH_SIZE = 2
N_LABELS   = 2
TRANSPORT  = False
T_EPS      = 10.0
T_SIZE     = 16

# 모델 정의: ResNet50 마지막 fc 수정
def get_resnet(num_classes):
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# GroupDRO 학습 함수
def train_groupdro(model, loaders, n_steps, eta, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    n_groups = len(loaders)
    # q를 uniform 초기화
    q = torch.ones(n_groups, device=device) / n_groups
    iters = [iter(dl) for dl in loaders]
    for step in range(n_steps):
        losses = []
        for i in range(n_groups):
            try:
                x, y = next(iters[i])
            except StopIteration:
                iters[i] = iter(loaders[i])
                x, y = next(iters[i])
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            losses.append(loss)
            # q 업데이트
            q[i] = q[i] * torch.exp(eta * loss.detach())
        q = q / q.sum()
        total_loss = sum(q[i] * losses[i] for i in range(n_groups))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return model

# Saliency map 계산 함수
def compute_saliency(model, x, target, device):
    model.eval()
    x = x.to(device).unsqueeze(0)
    x.requires_grad_()
    out = model(x)
    loss = F.cross_entropy(out, torch.tensor([target], device=device))
    model.zero_grad()
    loss.backward()
    sal = x.grad.abs().squeeze(0).sum(0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal.cpu().numpy()

# 메인 파이프라인
def main():
    # 1) 데이터 로드 및 그룹 제외
    dataset    = WaterbirdsDataset(root_dir=DATA_ROOT, download=True)
    splits     = dataset.split_array
    train_idx  = np.where(splits == 0)[0]
    val_idx    = np.where(splits == 1)[0]
    test_idx   = np.where(splits == 2)[0]
    y_arr      = dataset.y_array
    bg_i       = dataset.metadata_fields.index('background')
    bg_arr     = dataset.metadata_array[:, bg_i]
    g_arr      = bg_arr * 2 + y_arr
    # (waterbird, land background) => y=1, bg=0 -> g=1
    train_idx  = train_idx[g_arr[train_idx] != 1]

    # 2) GroupDRO 학습을 위한 DataLoader 준비
    groups     = sorted(np.unique(g_arr[train_idx]))
    bird_loads = []
    for g in groups:
        idxs = train_idx[g_arr[train_idx] == g]
        ds   = WaterbirdsTorchDataset(dataset, idxs, img_transform, mode='bird')
        bird_loads.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True))
    # model_bird 학습
    model_bird = get_resnet(2)
    model_bird = train_groupdro(model_bird, bird_loads, N_STEPS, DRO_ETA, DEVICE)

    # model_background 학습 (타겟=background)
    bg_loads = []
    for g in groups:
        idxs = train_idx[g_arr[train_idx] == g]
        ds   = WaterbirdsTorchDataset(dataset, idxs, img_transform, mode='background')
        bg_loads.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True))
    model_bg = get_resnet(2)
    model_bg = train_groupdro(model_bg, bg_loads, N_STEPS, DRO_ETA, DEVICE)

    # 3) 샘플로 saliency map 계산 (Tensor 변환 후)
    # landbird-land (g=0)
    idx0     = np.random.choice(train_idx[g_arr[train_idx] == 0])
    x0,_     = dataset[idx0]
    x0       = img_transform(x0)
    t0       = int(bg_arr[idx0])
    sal_land = compute_saliency(model_bg, x0, t0, DEVICE)
    # waterbird-water (g=3)
    idx3     = np.random.choice(train_idx[g_arr[train_idx] == 3])
    x3,_     = dataset[idx3]
    x3       = img_transform(x3)
    t3       = int(y_arr[idx3])
    sal_feat = compute_saliency(model_bird, x3, t3, DEVICE)

    # 4) PuzzleMixup으로 synthetic 이미지 생성
    block_num = 4
    inp1      = x0.unsqueeze(0).to(DEVICE)
    inp2      = x3.unsqueeze(0).to(DEVICE)
    synth, _  = mixup_graph(inp1, None, np.array([0]), block_num,
                            alpha=0.5, beta=BETA, gamma=GAMMA,
                            eta=ETA, neigh_size=NEIGH_SIZE,
                            n_labels=N_LABELS, mean=None,
                            std=None, transport=TRANSPORT,
                            t_eps=T_EPS, t_size=T_SIZE,
                            noise=None, adv_mask1=0,
                            adv_mask2=0, device=DEVICE,
                            mp=None)
    synth     = synth.squeeze(0).cpu()
    os.makedirs('aug', exist_ok=True)
    T.ToPILImage()(synth).save('aug/syn.png')

    # 5) synthetic 데이터 라벨링 및 최종 GroupDRO 학습 (50 step마다 테스트 정확도 출력)
    syn_y    = 1; syn_bg = 0; syn_g = 1
    syn_ds   = TensorDataset(synth.unsqueeze(0), torch.tensor([syn_y]))
    orig_ds  = WaterbirdsTorchDataset(dataset, train_idx, img_transform, mode='bird')
    final_ds = ConcatDataset([orig_ds, syn_ds])
    final_g = np.concatenate([g_arr[train_idx], np.array([syn_g])])
    final_loads = []
    for g in sorted(np.unique(final_g)):
        idxs = np.where(final_g == g)[0]
        sub  = ConcatDataset([orig_ds, syn_ds])
        # 해당 subgroup 인덱스만 선택한 서브셋 생성
        ds   = torch.utils.data.Subset(sub, idxs)
        final_loads.append(DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True))
    final_model = get_resnet(2)
    final_model.to(DEVICE)
    optimizer_final = torch.optim.Adam(final_model.parameters(), lr=LR)
    criterion_final = nn.CrossEntropyLoss()
    n_groups_final = len(final_loads)
    q_final = torch.ones(n_groups_final, device=DEVICE) / n_groups_final
    iters_final = [iter(dl) for dl in final_loads]

    # 테스트 환경별 DataLoader 준비
    test_group_loaders = {}
    test_g = g_arr[test_idx]
    for g in sorted(np.unique(test_g)):
        idxs_g = test_idx[test_g == g]
        ds_g = WaterbirdsTorchDataset(dataset, idxs_g, img_transform, mode='bird')
        test_group_loaders[g] = DataLoader(ds_g, batch_size=BATCH_SIZE, shuffle=False)

    final_model.train()
    for step in range(1, N_STEPS + 1):
        losses = []
        for i in range(n_groups_final):
            try:
                xb, yb = next(iters_final[i])
            except StopIteration:
                iters_final[i] = iter(final_loads[i])
                xb, yb = next(iters_final[i])
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = final_model(xb)
            loss_i = criterion_final(logits, yb)
            losses.append(loss_i)
            q_final[i] = q_final[i] * torch.exp(DRO_ETA * loss_i.detach())
        q_final = q_final / q_final.sum()
        total_loss = sum(q_final[i] * losses[i] for i in range(n_groups_final))
        optimizer_final.zero_grad()
        total_loss.backward()
        optimizer_final.step()

        # 50 step마다 테스트 정확도 평가
        if step % 50 == 0:
            final_model.eval()
            print(f"=== Step {step} Test Accuracy per Subgroup ===")
            for g, loader_g in test_group_loaders.items():
                correct, total = 0, 0
                for xt, yt in loader_g:
                    xt, yt = xt.to(DEVICE), yt.to(DEVICE)
                    preds = final_model(xt).argmax(dim=1)
                    correct += (preds == yt).sum().item()
                    total += yt.size(0)
                acc = correct / total if total > 0 else 0
                print(f"Subgroup g={g}: {acc:.4f}")
            final_model.train()
    print("=== Final Training Complete ===")

if __name__ == '__main__':
    main() 