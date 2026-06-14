#!/usr/bin/env python3
"""
Kaggle Training Script: MobileNetV3-Small Species Classifier
============================================================
This script is designed to run in a Kaggle Notebook with a GPU.
It takes the existing YOLO flora dataset (which has 34 classes),
extracts cropped plant regions using the bounding boxes, maps 
the classes down to 17 species, and trains a fast MobileNetV3-Small.

Data Path:
/kaggle/input/datasets/anwarmounir67/flora-is-cool/content/yolo_flora

Output:
/kaggle/working/models/best_species_classifier.pt
"""

import json
import os
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
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

# Kaggle dataset paths
DATASET_BASE = Path("/kaggle/input/datasets/anwarmounir67/flora-is-cool/content/yolo_flora")
IMAGES_DIR = DATASET_BASE / "images"
LABELS_DIR = DATASET_BASE / "labels"

# Working directories (Kaggle allows writing to /kaggle/working)
WORK_DIR = Path("/kaggle/working")
CROP_DIR = WORK_DIR / "cropped"
MODEL_DIR = WORK_DIR / "models"

# Ensure output directories exist
CROP_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 4  # Adjust based on Kaggle instance (usually 2 or 4)
# Ensure Kaggle GPU is turned on
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA is not available! Please turn on the GPU in Kaggle (Settings -> Accelerator -> GPU T4 x2)")

# Strict GPU configuration
torch.backends.cudnn.benchmark = True  # Accelerates training on static image sizes
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

STAGE1_EPOCHS = 5    # Head only
STAGE2_EPOCHS = 45   # Full fine-tune (50 total epochs)
STAGE1_LR = 0.01
STAGE2_LR = 0.0005

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

SPECIES = [
    "adansonia", "acacia", "vachellia", "senegalia", "combretum",
    "brachystegia", "colophospermum", "ficus", "khaya", "macaranga",
    "euphorbia", "aloe", "protea", "erica", "themeda", "andropogon", "tamarix",
]
NUM_SPECIES = len(SPECIES)

# ──────────────────────────────────────────────────────────────────────
# Phase 1: Dynamic Cropping (Extract Dataset)
# ──────────────────────────────────────────────────────────────────────

def extract_crops():
    print(f"==================================================")
    print(f"  Phase 1: Extracting Crops from YOLO dataset")
    print(f"==================================================")
    
    for split in ["train", "val"]:
        split_img_dir = IMAGES_DIR / split
        split_lbl_dir = LABELS_DIR / split
        
        if not split_img_dir.exists() or not split_lbl_dir.exists():
            print(f"  ⚠️ Missing {split} directory, skipping...")
            continue
            
        print(f"\n  Processing {split} split...")
        img_files = list(split_img_dir.glob("*.jpg"))
        
        for sp in SPECIES:
            (CROP_DIR / split / sp).mkdir(parents=True, exist_ok=True)
            
        processed = 0
        skipped = 0
        
        for img_path in img_files:
            lbl_path = split_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                skipped += 1
                continue
                
            try:
                lbl_text = lbl_path.read_text().strip()
                if not lbl_text:
                    skipped += 1
                    continue
                    
                parts = lbl_text.split("\n")[0].split()
                if len(parts) < 5:
                    skipped += 1
                    continue
                    
                class_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                species_id = class_id // 2
                if species_id >= NUM_SPECIES:
                    continue
                    
                species_name = SPECIES[species_id]
                out_path = CROP_DIR / split / species_name / f"{img_path.stem}.jpg"
                
                if out_path.exists():
                    processed += 1
                    continue
                
                img = Image.open(img_path).convert("RGB")
                img_w, img_h = img.size
                
                # BBox coords
                x1 = max(0, int((cx - w / 2) * img_w))
                y1 = max(0, int((cy - h / 2) * img_h))
                x2 = min(img_w, int((cx + w / 2) * img_w))
                y2 = min(img_h, int((cy + h / 2) * img_h))
                
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    skipped += 1
                    continue
                    
                cropped = img.crop((x1, y1, x2, y2))
                cropped = cropped.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                cropped.save(out_path, "JPEG", quality=85)
                
                processed += 1
                if processed % 1000 == 0:
                    print(f"    {processed} images cropped...")
            except Exception as e:
                skipped += 1
                continue
                
        print(f"  ✅ {split} complete! Processed: {processed}, Skipped: {skipped}")

# ──────────────────────────────────────────────────────────────────────
# Phase 2: Dataloaders & Modeling
# ──────────────────────────────────────────────────────────────────────

def get_dataloaders():
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

    import io
    from torch.utils.data import Dataset

    class InMemoryDataset(Dataset):
        """Loads all images into RAM as compressed bytes to bypass Kaggle's slow disk I/O."""
        def __init__(self, image_folder, transform=None):
            self.samples = image_folder.samples
            self.classes = image_folder.classes
            self.transform = transform
            self.images = []
            
            print(f"\n  📥 Loading {len(self.samples)} images into RAM for fast training...")
            for path, target in self.samples:
                with open(path, "rb") as f:
                    self.images.append((f.read(), target))
                
        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img_data, target = self.images[idx]
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, target

    base_train_ds = ImageFolder(str(CROP_DIR / "train"))
    base_val_ds = ImageFolder(str(CROP_DIR / "val"))
    
    train_ds = InMemoryDataset(base_train_ds, transform=train_transform)
    val_ds = InMemoryDataset(base_val_ds, transform=val_transform)

    # Class balancing
    class_counts = defaultdict(int)
    for _, lbl in train_ds.samples:
        class_counts[lbl] += 1

    n_classes = len(train_ds.classes)
    total = sum(class_counts.values())
    weights = {k: total / (n_classes * v) for k, v in class_counts.items() if v > 0}
    sample_weights = [weights.get(lbl, 1.0) for _, lbl in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, val_loader, train_ds.classes

def build_model(num_classes):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_feat = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model

def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
    model.eval()
    running_loss, correct, total = 0, 0, 0
    per_cls_c = defaultdict(int)
    per_cls_t = defaultdict(int)
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(images)
        loss = criterion(out, labels)
        running_loss += loss.item() * images.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        for p, t in zip(pred, labels):
            per_cls_t[t.item()] += 1
            if p == t: per_cls_c[t.item()] += 1
    return running_loss / total, 100 * correct / total, per_cls_c, per_cls_t

# ──────────────────────────────────────────────────────────────────────
# Phase 3: Main Training Loop
# ──────────────────────────────────────────────────────────────────────

def main():
    print(f"==================================================")
    print(f"  Kaggle MobileNetV3-Small Training")
    print(f"  Device: {DEVICE}")
    print(f"==================================================")

    # 1. Extract Crops (Skip if already done)
    existing_crops = sum(1 for _ in CROP_DIR.rglob("*.jpg"))
    if existing_crops > 10000:
        print(f"\n  ✅ Found {existing_crops} cropped images in {CROP_DIR}. Skipping extraction.")
    else:
        extract_crops()

    # 2. Prepare Data
    train_loader, val_loader, classes = get_dataloaders()
    num_classes = len(classes)
    print(f"\n  Classes ({num_classes}): {classes}")

    # 3. Build Model
    model = build_model(num_classes)
    
    # Use DataParallel if multiple GPUs are available (Kaggle T4 x2)
    if torch.cuda.device_count() > 1:
        print(f"\n  🚀 Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
        
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_val_acc = 0

    # STAGE 1: Head Only
    print(f"\n  [Stage 1] Training classifier head ({STAGE1_EPOCHS} epochs)")
    freeze_backbone(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=STAGE1_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS, eta_min=1e-5)

    for epoch in range(STAGE1_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc, _, _ = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        mark = " ⭐" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"model_state_dict": state_to_save, "classes": classes}, MODEL_DIR / "best.pt")
            
        print(f"  Ep {epoch+1:>2}/{STAGE1_EPOCHS} | TrAcc: {tr_acc:>5.1f}% | VlAcc: {vl_acc:>5.1f}% | Time: {time.time()-t0:>3.0f}s{mark}")

    # STAGE 2: Full Fine-tune
    print(f"\n  [Stage 2] Full fine-tune ({STAGE2_EPOCHS} epochs)")
    unfreeze_all(model)
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": STAGE2_LR * 0.1},
        {"params": head_params, "lr": STAGE2_LR},
    ], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE2_EPOCHS, eta_min=1e-6)

    no_improve = 0
    for epoch in range(STAGE2_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc, per_c, per_t = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        mark = " ⭐" if vl_acc > best_val_acc else ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve = 0
            state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({"model_state_dict": state_to_save, "classes": classes}, MODEL_DIR / "best.pt")
        else:
            no_improve += 1
            
        print(f"  Ep {epoch+1:>2}/{STAGE2_EPOCHS} | TrAcc: {tr_acc:>5.1f}% | VlAcc: {vl_acc:>5.1f}% | Time: {time.time()-t0:>3.0f}s{mark}")
        
        if no_improve >= 12:
            print("  ⏹ Early stopping.")
            break

    # 4. Final Evaluation & Export
    final_model = model.module if isinstance(model, nn.DataParallel) else model
    final_model.load_state_dict(torch.load(MODEL_DIR / "best.pt", weights_only=False)["model_state_dict"])
    final_model.eval()
    _, final_acc, per_c, per_t = validate(model, val_loader, criterion, DEVICE)

    print(f"\n==================================================")
    print(f"  FINAL RESULTS")
    print(f"==================================================")
    print(f"  Best Val Accuracy: {best_val_acc:.1f}%")
    
    for idx, cls in enumerate(classes):
        c, t = per_c.get(idx, 0), per_t.get(idx, 0)
        acc = 100 * c / t if t > 0 else 0
        print(f"  {cls:<20} {c:>4}/{t:<4} ({acc:>5.1f}%)")

    # Export ONNX
    try:
        dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        torch.onnx.export(
            final_model, dummy_input, str(MODEL_DIR / "species_classifier.onnx"),
            input_names=["image"], output_names=["species"],
            dynamic_axes={"image": {0: "batch"}, "species": {0: "batch"}},
            opset_version=13,
        )
        print(f"\n  ✅ ONNX model saved to {MODEL_DIR / 'species_classifier.onnx'}")
    except Exception as e:
        print(f"\n  ⚠️ ONNX export failed: {e}")

    # Save class map
    with open(MODEL_DIR / "classes.json", "w") as f:
        json.dump({i: name for i, name in enumerate(classes)}, f, indent=2)

    print(f"  📁 All files saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
