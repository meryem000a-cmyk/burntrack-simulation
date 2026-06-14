#!/usr/bin/env python3
"""
Kaggle Notebook: Train dryness CNN on 2x T4 GPUs.
"""

import random
import time
from pathlib import Path

import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# ── Paths ──
DATA = Path("/kaggle/input/datasets/anwarmounir67/vlm-nano/balanced")
WORKING = Path("/kaggle/working")
CKPT_DIR = WORKING / "dryness_ckpts"
CKPT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 200
PATIENCE_VAL_ACC = 15
PATIENCE_VAL_LOSS = 5


class DrynessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x).flatten(1)
        return self.fc(x).squeeze(1)


class DrynessDataset(Dataset):
    def __init__(self, data_dir, split="train", val_ratio=0.15):
        self.split = split

        # Build sample list and pre-decode to memmap
        samples = []
        for cls_dir in sorted(Path(data_dir).iterdir()):
            if not cls_dir.is_dir():
                continue
            if cls_dir.name.endswith("_not_dry"):
                label = 0.0
            elif cls_dir.name.endswith("_dry"):
                label = 1.0
            else:
                continue
            for f in cls_dir.glob("*.jpg"):
                samples.append((str(f), label))

        random.shuffle(samples)
        split_idx = int(len(samples) * (1 - val_ratio))
        if split == "train":
            samples = samples[:split_idx]
        else:
            samples = samples[split_idx:]

        self.labels = [s[1] for s in samples]
        N = len(samples)
        tag = "train" if split == "train" else "val"
        mmap_path = Path(f"/kaggle/working/dryness_{tag}.mmap")

        resize = transforms.Resize((IMG_SIZE, IMG_SIZE))
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if not mmap_path.exists():
            print(f"  Pre-decoding {N} {tag} images to memmap...", end=" ", flush=True)
            t0 = time.time()
            mmap = np.memmap(mmap_path, dtype="float16", mode="write", shape=(N, 3, IMG_SIZE, IMG_SIZE))
            for i, (path, _) in enumerate(samples):
                img = Image.open(path).convert("RGB")
                img = resize(img)
                img = to_tensor(img)
                img = normalize(img)
                mmap[i] = img.numpy().astype("float16")
            mmap.flush()
            del mmap
            print(f"{time.time()-t0:.0f}s", flush=True)
        else:
            print(f"  Loading cached {tag} memmap...", flush=True)

        self.mmap = np.memmap(mmap_path, dtype="float16", mode="readonly", shape=(N, 3, IMG_SIZE, IMG_SIZE))
        self.N = N

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = torch.from_numpy(self.mmap[idx].copy()).float()
        if self.split == "train":
            if torch.rand(1).item() > 0.5:
                img = img.flip(-1)
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.7, 1.0), ratio=(1.0, 1.0))
            img = F.interpolate(img[:, i:i+h, j:j+w].unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode="bilinear").squeeze(0)
            img = img + torch.randn_like(img) * 0.02
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"  Device: {device} ({n_gpus} GPUs)", flush=True)

    print(f"\n  Loading data...", flush=True)
    train_ds = DrynessDataset(DATA, "train")
    val_ds = DrynessDataset(DATA, "val")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}", flush=True)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    acc_no_improve = 0
    loss_no_improve = 0
    ckpt_counter = 0

    model = DrynessCNN()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\n  {'Ep':>3} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7} {'Best':>7} {'Time':>6}", flush=True)
    print(f"  {'-'*3} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*6}", flush=True)

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()

        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters() if not isinstance(model, nn.DataParallel) else model.module.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            tr_correct += (preds == labels).sum().item()
            tr_total += len(labels)
            if (batch_idx + 1) % max(1, n_train_batches // 10) == 0 or batch_idx == n_train_batches - 1:
                pct = 100 * (batch_idx + 1) / n_train_batches
                curr_acc = 100 * tr_correct / tr_total
                print(f"    [{batch_idx+1}/{n_train_batches}] loss: {loss.item():.4f}  acc: {curr_acc:.1f}%  {pct:.0f}%", end="\r", flush=True)
        print(flush=True)
        tr_loss /= n_train_batches
        tr_acc = 100 * tr_correct / tr_total

        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                vl_loss += criterion(logits, labels).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                vl_correct += (preds == labels).sum().item()
                vl_total += len(labels)
                if (batch_idx + 1) % max(1, n_val_batches // 5) == 0 or batch_idx == n_val_batches - 1:
                    pct = 100 * (batch_idx + 1) / n_val_batches
                    curr_acc = 100 * vl_correct / vl_total
                    print(f"    val [{batch_idx+1}/{n_val_batches}] acc: {curr_acc:.1f}%  {pct:.0f}%", end="\r", flush=True)
        print(flush=True)
        vl_loss /= n_val_batches
        vl_acc = 100 * vl_correct / vl_total

        elapsed = time.time() - t0

        save_model = model.module if isinstance(model, nn.DataParallel) else model
        state = save_model.state_dict()

        # Save last 20 checkpoints
        ckpt_counter += 1
        torch.save(state, CKPT_DIR / f"epoch_{epoch+1:03d}.pt")
        if ckpt_counter > 20:
            oldest = epoch + 1 - 20
            old_path = CKPT_DIR / f"epoch_{oldest:03d}.pt"
            if old_path.exists():
                old_path.unlink()

        # Track best
        is_best = False
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            acc_no_improve = 0
            is_best = True
        else:
            acc_no_improve += 1

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            loss_no_improve = 0
        else:
            loss_no_improve += 1

        if is_best:
            torch.save(state, WORKING / "best_dryness_cnn.pt")

        mark = " ⭐" if is_best else ""
        print(f"  {epoch+1:>3} {tr_loss:>8.4f} {tr_acc:>6.2f}% {vl_loss:>8.4f} {vl_acc:>6.2f}% {best_val_acc:>6.2f}% {elapsed:>5.0f}s{mark}", flush=True)

        # Early stopping
        if loss_no_improve >= PATIENCE_VAL_LOSS:
            print(f"  ⏹ Val loss not improving for {PATIENCE_VAL_LOSS} epochs. Stopping.", flush=True)
            break
        if acc_no_improve >= PATIENCE_VAL_ACC:
            print(f"  ⏹ Val acc not improving for {PATIENCE_VAL_ACC} epochs. Stopping.", flush=True)
            break

    print(f"\n  Best val accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"  Best checkpoint: {WORKING / 'best_dryness_cnn.pt'}", flush=True)
    print(f"  Last 20 checkpoints: {CKPT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
