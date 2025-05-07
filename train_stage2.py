import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class SyntheticReplacementDataset(Dataset):
    """
    base_dataset의 replace_indices 위치 샘플을 synth_files로 modulo 매핑
    """
    def __init__(self, base_dataset, synth_files, replace_indices):
        self.base = base_dataset
        self.transform = getattr(base_dataset, 'transform', None)
        self.synth_files = synth_files
        self.replace_list = list(replace_indices)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x_orig, y, metadata = self.base[idx]
        if idx in self.replace_list:
            i = self.replace_list.index(idx)
            path = self.synth_files[i % len(self.synth_files)]
            img = Image.open(path).convert('RGB')
            x = self.transform(img) if self.transform else transforms.ToTensor()(img)
        else:
            x = x_orig
        return x, y, metadata


class SyntheticOnlyDataset(Dataset):
    """
    synth_files의 모든 이미지를 한 번씩 평가에 사용
    """
    def __init__(self, synth_files, transform, y_label, bg_value):
        self.synth_files = synth_files
        self.transform = transform
        self.y_label = y_label
        self.bg_value = bg_value

    def __len__(self):
        return len(self.synth_files)

    def __getitem__(self, idx):
        path = self.synth_files[idx]
        img = Image.open(path).convert('RGB')
        x = self.transform(img)
        y = torch.tensor(self.y_label, dtype=torch.long)
        metadata = torch.tensor([self.bg_value, self.y_label], dtype=torch.long)
        return x, y, metadata


def evaluate_group_acc(model, loader, device, n_groups, n_backgrounds, bg_idx):
    model.eval()
    correct = torch.zeros(n_groups, device=device)
    total = torch.zeros(n_groups, device=device)
    with torch.no_grad():
        for x, y, metadata in loader:
            x, y = x.to(device), y.to(device)
            bg = metadata[:, bg_idx].long().to(device)
            groups = y.long() * n_backgrounds + bg
            preds = model(x).argmax(dim=1)
            for g in range(n_groups):
                mask = (groups == g)
                if mask.any():
                    correct[g] += (preds[mask] == y[mask]).sum()
                    total[g]   += mask.sum()
    return (correct / total).cpu().numpy()


def evaluate_group_acc_combined(model, orig_loader, synth_loader,
                                device, n_groups, n_backgrounds, bg_idx,
                                synthetic_group_idx):
    model.eval()
    group_correct = torch.zeros(n_groups, device=device)
    group_total = torch.zeros(n_groups, device=device)

    with torch.no_grad():
        # 원본 val 데이터로 g != synthetic_group 처리
        for x, y, metadata in orig_loader:
            x, y = x.to(device), y.to(device)
            bg = metadata[:, bg_idx].long().to(device)
            groups = y.long() * n_backgrounds + bg
            preds = model(x).argmax(dim=1)
            for g in range(n_groups):
                if g == synthetic_group_idx:
                    continue
                mask = groups == g
                if mask.any():
                    group_correct[g] += (preds[mask] == y[mask]).sum()
                    group_total[g]   += mask.sum()
        # synthetic-only 데이터로 g == synthetic_group
        for x, y, metadata in synth_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            g = synthetic_group_idx
            group_correct[g] += (preds == y).sum()
            group_total[g]   += preds.size(0)

    return (group_correct / group_total).cpu().numpy()


def main():
    device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs   = 200
    batch_size   = 32
    lr           = 0.00005
    weight_decay = 5e-3
    eta          = 0.05
    patience     = 3

    # 1) Wilds Waterbirds 로드
    dataset = get_dataset(dataset="waterbirds", download=False)
    transforms_train = get_transforms()
    transforms_eval  = get_transforms()

    train_base = dataset.get_subset("train", transform=transforms_train)
    val_base   = dataset.get_subset("val",   transform=transforms_eval)
    test_data  = dataset.get_subset("test",  transform=transforms_eval)

    # 2) metadata_fields 인덱스
    fields = dataset.metadata_fields
    bg_idx = fields.index('background')
    y_idx  = fields.index('y')

    # 3) (waterbird, land) 그룹 인덱스 추출
    train_meta = train_base.metadata_array
    val_meta   = val_base.metadata_array
    train_replace = np.where((train_meta[:, y_idx] == 1) & (train_meta[:, bg_idx] == 0))[0]
    val_replace   = np.where((val_meta[:,   y_idx] == 1) & (val_meta[:,   bg_idx] == 0))[0]

    # 4) synth_train, synth_val 디렉토리 설정
    synth_train_dir = "/data2/yws_ids25/data/AIDA/data/puzzled_data/train"
    synth_val_dir   = "/data2/yws_ids25/data/AIDA/data/puzzled_data/validation"

    synth_train = sorted([
        os.path.join(synth_train_dir, f)
        for f in os.listdir(synth_train_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
           and os.path.isfile(os.path.join(synth_train_dir, f))
    ])
    synth_val = sorted([
        os.path.join(synth_val_dir, f)
        for f in os.listdir(synth_val_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
           and os.path.isfile(os.path.join(synth_val_dir, f))
    ])

    # synthetic_group_idx 계산 (y=1, bg=0)
    n_backgrounds = len(np.unique(train_meta[:, bg_idx]))
    synthetic_group_idx = 1 * n_backgrounds + 0

    # 5) Dataset 래핑
    train_data = SyntheticReplacementDataset(train_base, synth_train, train_replace)
    val_loader_orig = get_eval_loader("standard", val_base,
                                      batch_size=batch_size, num_workers=4)
    val_loader_synth = DataLoader(
        SyntheticOnlyDataset(synth_val, transforms_eval, y_label=1, bg_value=0),
        batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = get_eval_loader("standard", test_data,
                                  batch_size=batch_size, num_workers=4)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # 6) 그룹 수
    n_classes = dataset.n_classes
    n_groups  = n_classes * n_backgrounds

    # 7) 모델/옵티마이저/손실
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    criterion = nn.CrossEntropyLoss(reduction="none")

    # 8) q 초기화
    q = torch.ones(n_groups, device=device) / n_groups

    # 9) 학습 및 검증
    worst_group_val_acc = 0
    is_best = False
    patience_counter = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, labels, metadata in train_loader:
            images, labels = images.to(device), labels.to(device)
            bg = metadata[:, bg_idx].long().to(device)
            groups = labels.long() * n_backgrounds + bg

            logits = model(images)
            losses = criterion(logits, labels)
            group_losses = torch.zeros(n_groups, device=device)
            for g in range(n_groups):
                mask = (groups == g)
                if mask.any():
                    group_losses[g] = losses[mask].mean()

            loss = torch.dot(q, group_losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                q = q * torch.exp(eta * group_losses)
                q = q / q.sum()

        val_acc = evaluate_group_acc_combined(
            model,
            val_loader_orig,
            val_loader_synth,
            device,
            n_groups,
            n_backgrounds,
            bg_idx,
            synthetic_group_idx
        )
        print(f"Epoch {epoch:03d}")
        print(f"val group acc: {val_acc}")
        if val_acc.min() > worst_group_val_acc:
            worst_group_val_acc = val_acc.min()
            best_val_acc = val_acc
            is_best = True
            patience_counter = 0
        else:
            patience_counter += 1
        with torch.no_grad():
            # 10) 최종 테스트 — 원본 test 데이터로만 그룹별 accuracy 계산
            test_acc = evaluate_group_acc(
                model,
                test_loader,
                device,
                n_groups,
                n_backgrounds,
                bg_idx
            )
            if is_best:
                best_test_acc = test_acc
            print("test group acc:", test_acc)
            print("------------------------------------------------------------------------------------------------")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("best val group acc:", best_val_acc)
    print("best test group acc:", best_test_acc)
if __name__ == "__main__":
    main()