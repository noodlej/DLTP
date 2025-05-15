def train_model(lr, weight_decay, eta, 
                num_epochs=200, batch_size=32, patience=3,
                synth_train_dir="/data/users/jnoodle/saliency/puzzled_data/train",
                synth_val_dir="/data/users/jnoodle/saliency/puzzled_data1/validation"):

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

    class SyntheticReplacementDataset(Dataset):
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
            for x, y, metadata in synth_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                g = synthetic_group_idx
                group_correct[g] += (preds == y).sum()
                group_total[g]   += preds.size(0)

        return (group_correct / group_total).cpu().numpy()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset(dataset="waterbirds", download=False)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_base = dataset.get_subset("train", transform=transform)
    val_base = dataset.get_subset("val", transform=transform)
    test_data = dataset.get_subset("test", transform=transform)

    fields = dataset.metadata_fields
    bg_idx = fields.index('background')
    y_idx = fields.index('y')

    train_meta = train_base.metadata_array
    val_meta = val_base.metadata_array
    train_replace = np.where((train_meta[:, y_idx] == 1) & (train_meta[:, bg_idx] == 0))[0]
    val_replace = np.where((val_meta[:, y_idx] == 1) & (val_meta[:, bg_idx] == 0))[0]

    synth_train = sorted([
        os.path.join(synth_train_dir, f)
        for f in os.listdir(synth_train_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    synth_val = sorted([
        os.path.join(synth_val_dir, f)
        for f in os.listdir(synth_val_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    n_backgrounds = len(np.unique(train_meta[:, bg_idx]))
    synthetic_group_idx = 1 * n_backgrounds + 0

    train_data = SyntheticReplacementDataset(train_base, synth_train, train_replace)
    val_loader_orig = get_eval_loader("standard", val_base, batch_size=batch_size, num_workers=4)
    val_loader_synth = DataLoader(SyntheticOnlyDataset(synth_val, transform, 1, 0),
                                  batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = get_eval_loader("standard", test_data, batch_size=batch_size, num_workers=4)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    n_classes = dataset.n_classes
    n_groups = n_classes * n_backgrounds

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    criterion = nn.CrossEntropyLoss(reduction="none")

    q = torch.ones(n_groups, device=device) / n_groups

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

        val_acc = evaluate_group_acc_combined(model, val_loader_orig, val_loader_synth, device,
                                              n_groups, n_backgrounds, bg_idx, synthetic_group_idx)
        # val_acc = evaluate_group_acc(model, val_loader_orig, device, n_groups, n_backgrounds, bg_idx)
        if val_acc.min() > worst_group_val_acc:
            worst_group_val_acc = val_acc.min()
            best_val_acc = val_acc
            is_best = True
            patience_counter = 0
        else:
            is_best = False
            patience_counter += 1

        test_acc = evaluate_group_acc(model, test_loader, device, n_groups, n_backgrounds, bg_idx)
        if is_best:
            best_test_acc = test_acc

        if patience_counter >= patience:
            break

    return best_val_acc, best_test_acc, epoch
