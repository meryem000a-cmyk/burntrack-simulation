#!/usr/bin/env python3
"""
Species Classifier — Fast CPU Training
========================================
Phase 1: Uses pre-extracted crops from distill_species_classifier.py
Phase 2: Two-stage fine-tuning of MobileNetV3-Small:
  Stage 1: Freeze backbone, train only classifier head (fast, ~1 min/epoch)
  Stage 2: Unfreeze all, fine-tune with low LR (slower, ~5 min/epoch)

Usage:
    ./flora_env/bin/python train_species_fast.py
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

CROP_DIR = Path("datasets/species_cls/cropped")
MODEL_DIR = Path("datasets/species_cls/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 128        # Larger batch for CPU efficiency
NUM_WORKERS = 4
DEVICE = "cpu"

# Two-stage training
STAGE1_EPOCHS = 8       # Head-only training (fast)
STAGE2_EPOCHS = 12      # Full fine-tune (slower but crucial)
STAGE1_LR = 0.01
STAGE2_LR = 0.0003

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

SPECIES = [
    "adansonia", "acacia", "vachellia", "senegalia", "combretum",
    "brachystegia", "colophospermum", "ficus", "khaya", "macaranga",
    "euphorbia", "aloe", "protea", "erica", "themeda", "andropogon", "tamarix",
]


def get_dataloaders():
    """Create train/val dataloaders with augmentation and balanced sampling."""

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(str(CROP_DIR / "train"), transform=train_transform)
    val_ds = ImageFolder(str(CROP_DIR / "val"), transform=val_transform)

    # Balanced sampler
    class_counts = defaultdict(int)
    for _, label in train_ds.samples:
        class_counts[label] += 1

    n_classes = len(train_ds.classes)
    total = sum(class_counts.values())
    weights = {k: total / (n_classes * v) for k, v in class_counts.items() if v > 0}
    sample_weights = [weights.get(lbl, 1.0) for _, lbl in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_loader, val_loader, train_ds.classes, class_counts


def build_model(num_classes):
    """Build MobileNetV3-Small with custom head."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Replace classifier
    in_feat = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    return model


def freeze_backbone(model):
    """Freeze everything except classifier."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze everything."""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return (loss, accuracy)."""
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    return running_loss / total, 100 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate, return (loss, accuracy, per_class_correct, per_class_total)."""
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    per_cls_correct = defaultdict(int)
    per_cls_total = defaultdict(int)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        loss = criterion(out, labels)

        running_loss += loss.item() * images.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

        for p, t in zip(pred, labels):
            per_cls_total[t.item()] += 1
            if p == t:
                per_cls_correct[t.item()] += 1

    return (
        running_loss / total,
        100 * correct / total,
        per_cls_correct,
        per_cls_total,
    )


def main():
    print("=" * 65)
    print("  MobileNetV3-Small Species Classifier (Fast CPU Training)")
    print("=" * 65)

    # Check crops exist
    train_count = sum(1 for _ in (CROP_DIR / "train").rglob("*.jpg"))
    val_count = sum(1 for _ in (CROP_DIR / "val").rglob("*.jpg"))
    print(f"  Dataset: {train_count:,} train / {val_count:,} val")

    if train_count < 100:
        print("  ❌ Not enough crops! Run distill_species_classifier.py first.")
        return

    train_loader, val_loader, classes, class_counts = get_dataloaders()
    num_classes = len(classes)
    print(f"  Classes ({num_classes}): {classes}")

    # Build model
    model = build_model(num_classes)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: MobileNetV3-Small ({total_params:,} params = {total_params/1e6:.1f}M)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0
    best_epoch = 0

    # ──────────────────────────────────────────────────────────
    # Stage 1: Head-only (frozen backbone)
    # ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Stage 1: Training classifier head only ({STAGE1_EPOCHS} epochs)")
    print(f"{'─'*65}")

    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} (backbone frozen)")

    model = model.to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=STAGE1_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS, eta_min=1e-5)

    print(f"  {'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'VlLoss':>7}  {'VlAcc':>6}  {'Time':>5}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*5}")

    for epoch in range(STAGE1_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        mark = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch, "stage": 1,
                "model_state_dict": model.state_dict(),
                "val_acc": vl_acc, "classes": classes,
            }, MODEL_DIR / "best_species_classifier.pt")
            mark = " ⭐"

        print(f"  {epoch+1:>3}  {tr_loss:>7.3f}  {tr_acc:>5.1f}%  {vl_loss:>7.3f}  {vl_acc:>5.1f}%  {elapsed:>4.0f}s{mark}")

    # ──────────────────────────────────────────────────────────
    # Stage 2: Full fine-tune (all layers unfrozen)
    # ──────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Stage 2: Full fine-tune ({STAGE2_EPOCHS} epochs)")
    print(f"{'─'*65}")

    unfreeze_all(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,} (all unfrozen)")

    # Use different LR for backbone vs head
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": STAGE2_LR * 0.1},  # Lower LR for pretrained backbone
        {"params": head_params, "lr": STAGE2_LR},
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS, eta_min=1e-6)

    print(f"  {'Ep':>3}  {'TrLoss':>7}  {'TrAcc':>6}  {'VlLoss':>7}  {'VlAcc':>6}  {'Time':>5}")
    print(f"  {'─'*3}  {'─'*7}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*5}")

    no_improve = 0
    for epoch in range(STAGE2_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc, per_cls_c, per_cls_t = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        mark = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = STAGE1_EPOCHS + epoch
            no_improve = 0
            torch.save({
                "epoch": best_epoch, "stage": 2,
                "model_state_dict": model.state_dict(),
                "val_acc": vl_acc, "classes": classes,
            }, MODEL_DIR / "best_species_classifier.pt")
            mark = " ⭐"
        else:
            no_improve += 1

        print(f"  {STAGE1_EPOCHS+epoch+1:>3}  {tr_loss:>7.3f}  {tr_acc:>5.1f}%  {vl_loss:>7.3f}  {vl_acc:>5.1f}%  {elapsed:>4.0f}s{mark}")

        if no_improve >= 8:
            print(f"\n  ⏹ Early stopping (no improvement for 8 epochs)")
            break

    # ──────────────────────────────────────────────────────────
    # Export & Summary
    # ──────────────────────────────────────────────────────────

    # Reload best weights
    ckpt = torch.load(MODEL_DIR / "best_species_classifier.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Final per-class eval
    _, final_acc, per_cls_c, per_cls_t = validate(model, val_loader, criterion, DEVICE)

    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS (Best epoch {best_epoch+1})")
    print(f"{'='*65}")
    print(f"  Val accuracy: {best_val_acc:.1f}%")
    print(f"  Model size:   {total_params:,} params ({total_params*4/1024/1024:.1f} MB FP32)")
    print(f"\n  {'Species':<20} {'Correct':>7} {'Total':>6} {'Acc':>7}")
    print(f"  {'─'*20} {'─'*7} {'─'*6} {'─'*7}")

    for idx, cls_name in enumerate(classes):
        c = per_cls_c.get(idx, 0)
        t = per_cls_t.get(idx, 0)
        acc = 100 * c / t if t > 0 else 0
        bar = "█" * int(acc / 5)
        print(f"  {cls_name:<20} {c:>7} {t:>6} {acc:>5.1f}% {bar}")

    # Export ONNX
    try:
        example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        torch.onnx.export(
            model, example,
            str(MODEL_DIR / "species_classifier.onnx"),
            input_names=["image"], output_names=["species"],
            dynamic_axes={"image": {0: "batch"}, "species": {0: "batch"}},
            opset_version=13,
        )
        onnx_mb = (MODEL_DIR / "species_classifier.onnx").stat().st_size / 1024 / 1024
        print(f"\n  ✅ ONNX exported: {onnx_mb:.1f} MB")
    except Exception as e:
        print(f"\n  ⚠️ ONNX export failed: {e}")

    # Save class map
    with open(MODEL_DIR / "species_classes.json", "w") as f:
        json.dump({i: name for i, name in enumerate(classes)}, f, indent=2)

    print(f"\n  📁 All models saved to: {MODEL_DIR}")
    print(f"  🏁 Done!")


if __name__ == "__main__":
    main()
