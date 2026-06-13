#!/usr/bin/env python3
"""
Train 5-layer CNN for dry/not-dry on CPU until 99.5% train / 98% val.
"""

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

IMG_SIZE = 224
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cpu"

TRAIN_TARGET = 99.5
VAL_TARGET = 98.0
MAX_EPOCHS = 200
PATIENCE = 1


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
        if split == "train":
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        self.samples = []
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
                self.samples.append((str(f), label))

        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * (1 - val_ratio))
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def main():
    data_dir = "datasets/balanced"

    print(f"  Data: {data_dir}")
    print(f"  Device: {DEVICE}")
    print()

    train_ds = DrynessDataset(data_dir, "train")
    val_ds = DrynessDataset(data_dir, "val")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    dry_ratio = sum(l for _, l in train_ds.samples) / len(train_ds)
    print(f"  Dry in train: {dry_ratio:.0%}")
    print(f"  Targets: {TRAIN_TARGET}% train, {VAL_TARGET}% val")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False, num_workers=0)

    model = DrynessCNN()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params:,}")
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = 0.0
    no_improve = 0
    print(f"\n  {'Ep':>3} {'TrLoss':>8} {'TrAcc':>7} {'VlLoss':>8} {'VlAcc':>7} {'Time':>6}")
    print(f"  {'-'*3} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()

        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        n_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            tr_correct += (preds == labels).sum().item()
            tr_total += len(labels)
            if (batch_idx + 1) % max(1, n_batches // 10) == 0 or batch_idx == n_batches - 1:
                pct = 100 * (batch_idx + 1) / n_batches
                curr_acc = 100 * tr_correct / tr_total
                print(f"    [{batch_idx+1}/{n_batches}] loss: {loss.item():.4f}  acc: {curr_acc:.1f}%  {pct:.0f}%", end="\r", flush=True)
        print()
        tr_loss /= n_batches
        tr_acc = 100 * tr_correct / tr_total

        model.eval()
        vl_loss, vl_correct, vl_total = 0.0, 0, 0
        n_val_batches = len(val_loader)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                vl_loss += criterion(logits, labels).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                vl_correct += (preds == labels).sum().item()
                vl_total += len(labels)
                if (batch_idx + 1) % max(1, n_val_batches // 5) == 0 or batch_idx == n_val_batches - 1:
                    pct = 100 * (batch_idx + 1) / n_val_batches
                    curr_acc = 100 * vl_correct / vl_total
                    print(f"    val [{batch_idx+1}/{n_val_batches}] acc: {curr_acc:.1f}%  {pct:.0f}%", end="\r", flush=True)
        print()
        vl_loss /= n_val_batches
        vl_acc = 100 * vl_correct / vl_total

        elapsed = time.time() - t0
        print(f"  {epoch+1:>3} {tr_loss:>8.4f} {tr_acc:>6.2f}% {vl_loss:>8.4f} {vl_acc:>6.2f}% {elapsed:>5.0f}s")

        if vl_acc >= VAL_TARGET and tr_acc >= TRAIN_TARGET:
            torch.save(model.state_dict(), "best_dryness_cnn.pt")
            print(f"\n  ✅ Targets met! Train: {tr_acc:.1f}%, Val: {vl_acc:.1f}%")
            print(f"  Saved: best_dryness_cnn.pt")
            return

        if vl_acc > best_val:
            best_val = vl_acc
            no_improve = 0
            torch.save(model.state_dict(), "best_dryness_cnn.pt")
            print(f"  ⭐ Best val: {vl_acc:.2f}%")
        else:
            no_improve += 1

        if no_improve >= PATIENCE and tr_acc >= TRAIN_TARGET:
            print(f"\n  ⏹ Val stopped improving ({PATIENCE} epochs). Best: {best_val:.2f}%")
            return


if __name__ == "__main__":
    main()
