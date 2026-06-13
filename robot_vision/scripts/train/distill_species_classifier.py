#!/usr/bin/env python3
"""
Species Classifier Distillation Pipeline
==========================================
Extracts cropped plant regions from the annotated YOLO dataset,
maps the 34 detection classes → 17 species-only classes,
and trains a MobileNetV3-Small classifier (~2.5M params).

The idea: YOLO handles bbox + dryness (NDVI can do dryness too),
this small model handles species identification on cropped regions.

Usage:
    ./flora_env/bin/python distill_species_classifier.py
"""

import gc
import os
import random
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

ZIP_PATH = Path.home() / "Downloads" / "yolo_flora_annotated.zip"
WORK_DIR = Path("datasets/species_cls")
CROP_DIR = WORK_DIR / "cropped"       # ImageNet-style: cropped/{split}/{species}/img.jpg
MODEL_DIR = WORK_DIR / "models"

# Species list (17 species, mapped from 34 detection classes)
# class_id // 2 = species_id
SPECIES = [
    "adansonia",       # 0,1
    "acacia",          # 2,3
    "vachellia",       # 4,5
    "senegalia",       # 6,7
    "combretum",       # 8,9
    "brachystegia",    # 10,11
    "colophospermum",  # 12,13
    "ficus",           # 14,15
    "khaya",           # 16,17
    "macaranga",       # 18,19
    "euphorbia",       # 20,21
    "aloe",            # 22,23
    "protea",          # 24,25
    "erica",           # 26,27
    "themeda",         # 28,29
    "andropogon",      # 30,31
    "tamarix",         # 32,33
]

NUM_SPECIES = len(SPECIES)

# Training config
IMG_SIZE = 224           # MobileNetV3 default
BATCH_SIZE = 64
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
NUM_WORKERS = 4
DEVICE = "cpu"           # Use CPU for local training
MIN_SAMPLES_PER_CLASS = 10  # Skip classes with too few samples

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Extract & Crop
# ──────────────────────────────────────────────────────────────────────

def extract_and_crop():
    """Extract images from zip, crop using bbox labels, save as ImageNet-style dataset."""
    
    print("=" * 65)
    print("  Phase 1: Extracting & Cropping Plant Regions")
    print("=" * 65)
    
    # Create output dirs
    for split in ("train", "val"):
        for sp in SPECIES:
            (CROP_DIR / split / sp).mkdir(parents=True, exist_ok=True)
    
    stats = defaultdict(int)
    
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        # Index all files in the zip
        all_names = zf.namelist()
        
        # Separate images and labels
        img_files = {}
        lbl_files = {}
        
        for name in all_names:
            if "/images/" in name and name.endswith(".jpg"):
                # Extract split and basename
                if "/train/" in name:
                    split = "train"
                elif "/val/" in name:
                    split = "val"
                else:
                    continue
                stem = Path(name).stem
                img_files[(split, stem)] = name
            elif "/labels/" in name and name.endswith(".txt"):
                if "/train/" in name:
                    split = "train"
                elif "/val/" in name:
                    split = "val"
                else:
                    continue
                stem = Path(name).stem
                lbl_files[(split, stem)] = name
        
        print(f"  Found {len(img_files)} images, {len(lbl_files)} labels")
        
        # Process each image that has a matching label
        total = len(img_files)
        processed = 0
        start_time = time.time()
        
        for (split, stem), img_name in img_files.items():
            key = (split, stem)
            if key not in lbl_files:
                stats["no_label"] += 1
                continue
            
            # Check if already processed
            # We need to read the label first to know the species
            lbl_name = lbl_files[key]
            
            try:
                lbl_text = zf.read(lbl_name).decode("utf-8").strip()
                if not lbl_text:
                    stats["empty_label"] += 1
                    continue
                
                # Parse first line: class_id cx cy w h
                parts = lbl_text.split("\n")[0].split()
                if len(parts) < 5:
                    stats["bad_label"] += 1
                    continue
                
                class_id = int(parts[0])
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # Map detection class → species
                species_id = class_id // 2
                if species_id >= NUM_SPECIES:
                    stats["bad_class"] += 1
                    continue
                species_name = SPECIES[species_id]
                
                # Check if output already exists
                out_path = CROP_DIR / split / species_name / f"{stem}.jpg"
                if out_path.exists():
                    stats["already_done"] += 1
                    processed += 1
                    continue
                
                # Read image
                img_data = zf.read(img_name)
                img = Image.open(__import__("io").BytesIO(img_data))
                img = img.convert("RGB")
                
                img_w, img_h = img.size
                
                # Convert normalized YOLO bbox to pixel coords
                x1 = max(0, int((cx - w / 2) * img_w))
                y1 = max(0, int((cy - h / 2) * img_h))
                x2 = min(img_w, int((cx + w / 2) * img_w))
                y2 = min(img_h, int((cy + h / 2) * img_h))
                
                # Ensure minimum crop size
                crop_w = x2 - x1
                crop_h = y2 - y1
                if crop_w < 20 or crop_h < 20:
                    stats["too_small"] += 1
                    continue
                
                # Crop and resize
                cropped = img.crop((x1, y1, x2, y2))
                cropped = cropped.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                cropped.save(str(out_path), "JPEG", quality=85)
                
                stats[f"{split}_{species_name}"] += 1
                stats[f"{split}_total"] += 1
                processed += 1
                
            except Exception as e:
                stats["errors"] += 1
                continue
            
            # Progress
            if processed % 500 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  {processed:>6}/{total} ({rate:.0f} img/s)", end="\r", flush=True)
    
    print(f"\n\n  ✅ Cropping complete!")
    print(f"  Train: {stats.get('train_total', 0):,} crops")
    print(f"  Val:   {stats.get('val_total', 0):,} crops")
    print(f"  Skipped: {stats.get('no_label', 0)} no label, "
          f"{stats.get('empty_label', 0)} empty, "
          f"{stats.get('too_small', 0)} too small, "
          f"{stats.get('errors', 0)} errors, "
          f"{stats.get('already_done', 0)} already done")
    
    # Print per-species distribution
    print(f"\n  {'Species':<20} {'Train':>8} {'Val':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*6}")
    for sp in SPECIES:
        tr = stats.get(f"train_{sp}", 0)
        vl = stats.get(f"val_{sp}", 0)
        # Count existing files if we skipped (already_done)
        tr_actual = len(list((CROP_DIR / "train" / sp).glob("*.jpg")))
        vl_actual = len(list((CROP_DIR / "val" / sp).glob("*.jpg")))
        print(f"  {sp:<20} {tr_actual:>8} {vl_actual:>6}")


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Train MobileNetV3-Small
# ──────────────────────────────────────────────────────────────────────

def train_mobilenet():
    """Train MobileNetV3-Small as a species classifier."""
    
    print("\n" + "=" * 65)
    print("  Phase 2: Training MobileNetV3-Small Species Classifier")
    print("=" * 65)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    train_dataset = ImageFolder(str(CROP_DIR / "train"), transform=train_transform)
    val_dataset = ImageFolder(str(CROP_DIR / "val"), transform=val_transform)
    
    print(f"  Train: {len(train_dataset):,} images")
    print(f"  Val:   {len(val_dataset):,} images")
    print(f"  Classes: {train_dataset.classes}")
    
    num_classes = len(train_dataset.classes)
    
    # Handle class imbalance with WeightedRandomSampler
    class_counts = defaultdict(int)
    for _, label in train_dataset.samples:
        class_counts[label] += 1
    
    # Print class distribution
    print(f"\n  Class distribution:")
    for cls_idx, cls_name in enumerate(train_dataset.classes):
        count = class_counts.get(cls_idx, 0)
        bar = "█" * min(50, count // 50)
        print(f"    {cls_name:<20} {count:>6}  {bar}")
    
    # Compute sample weights for balanced sampling
    total_samples = sum(class_counts.values())
    class_weights = {k: total_samples / (num_classes * v) for k, v in class_counts.items() if v > 0}
    sample_weights = [class_weights.get(label, 1.0) for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    
    # Build model
    print(f"\n  📦 Loading MobileNetV3-Small (pretrained)...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    # Replace classifier head for our number of species
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    model = model.to(DEVICE)
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    print(f"\n  🏋️ Training for {NUM_EPOCHS} epochs on {DEVICE}...")
    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'LR':>10}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*10}")
    
    for epoch in range(NUM_EPOCHS):
        # ── Train ──
        model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / total
        train_acc = 100 * correct / total
        
        # ── Validate ──
        model.eval()
        val_loss_sum = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss_sum += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss_sum / val_total if val_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        
        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": train_dataset.classes,
                "num_classes": num_classes,
            }, MODEL_DIR / "best_species_classifier.pt")
            marker = " ⭐ BEST"
        
        print(f"  {epoch+1:>5}  {train_loss:>10.4f}  {train_acc:>8.1f}%  {val_loss:>8.4f}  {val_acc:>6.1f}%  {lr:>10.6f}{marker}")
        
        # Early stopping
        if epoch - best_epoch > 15:
            print(f"\n  ⏹ Early stopping at epoch {epoch+1} (no improvement for 15 epochs)")
            break
    
    # ── Save final model ──
    # Also save as standalone for easy loading
    model.load_state_dict(
        torch.load(MODEL_DIR / "best_species_classifier.pt", weights_only=False)["model_state_dict"]
    )
    
    # Export as TorchScript for deployment
    model.eval()
    example_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    try:
        scripted = torch.jit.trace(model, example_input)
        scripted.save(str(MODEL_DIR / "species_classifier_scripted.pt"))
        print(f"\n  ✅ TorchScript model saved: {MODEL_DIR / 'species_classifier_scripted.pt'}")
    except Exception as e:
        print(f"\n  ⚠️ TorchScript export failed: {e}")
    
    # Export ONNX
    try:
        torch.onnx.export(
            model, example_input,
            str(MODEL_DIR / "species_classifier.onnx"),
            input_names=["image"],
            output_names=["species"],
            dynamic_axes={"image": {0: "batch"}, "species": {0: "batch"}},
            opset_version=13,
        )
        onnx_size = (MODEL_DIR / "species_classifier.onnx").stat().st_size / (1024 * 1024)
        print(f"  ✅ ONNX model saved: {MODEL_DIR / 'species_classifier.onnx'} ({onnx_size:.1f} MB)")
    except Exception as e:
        print(f"  ⚠️ ONNX export failed: {e}")
    
    # ── Summary ──
    print(f"\n{'='*65}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*65}")
    print(f"  Best epoch:        {best_epoch + 1}")
    print(f"  Best val accuracy: {best_val_acc:.1f}%")
    print(f"  Model params:      {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Model size:        ~{total_params * 4 / (1024*1024):.1f} MB (FP32)")
    
    # Save class mapping
    class_map_path = MODEL_DIR / "species_classes.txt"
    with open(class_map_path, "w") as f:
        for i, cls in enumerate(train_dataset.classes):
            f.write(f"{i}: {cls}\n")
    print(f"  Class map saved:   {class_map_path}")
    
    return best_val_acc, history


# ──────────────────────────────────────────────────────────────────────
# Phase 3: Evaluate on fresh images  
# ──────────────────────────────────────────────────────────────────────

def evaluate_on_fresh():
    """Quick evaluation of the trained classifier on the fresh eval images."""
    
    fresh_dir = Path("datasets/fresh_eval/images")
    labels_dir = Path("datasets/fresh_eval/labels")
    
    if not fresh_dir.exists():
        print("\n  ⚠️ No fresh eval images found, skipping")
        return
    
    print(f"\n{'='*65}")
    print(f"  Phase 3: Evaluating on Fresh Images")
    print(f"{'='*65}")
    
    # Load model
    checkpoint = torch.load(
        MODEL_DIR / "best_species_classifier.pt",
        weights_only=False, map_location=DEVICE
    )
    classes = checkpoint["classes"]
    num_classes = checkpoint["num_classes"]
    
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(DEVICE)
    
    transform = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    correct = 0
    total = 0
    
    for img_path in sorted(fresh_dir.glob("*.jpg")):
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        
        # Read GT
        parts = label_path.read_text().strip().split()
        if len(parts) < 5:
            continue
        gt_class_id = int(parts[0])
        gt_species_id = gt_class_id // 2
        if gt_species_id >= NUM_SPECIES:
            continue
        gt_species = SPECIES[gt_species_id]
        
        # Crop using bbox
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        img = Image.open(str(img_path)).convert("RGB")
        img_w, img_h = img.size
        x1 = max(0, int((cx - w / 2) * img_w))
        y1 = max(0, int((cy - h / 2) * img_h))
        x2 = min(img_w, int((cx + w / 2) * img_w))
        y2 = min(img_h, int((cy + h / 2) * img_h))
        cropped = img.crop((x1, y1, x2, y2))
        
        # Classify
        input_tensor = transform(cropped).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = output.max(1)
        
        pred_species = classes[pred_idx.item()]
        total += 1
        match = (pred_species == gt_species)
        if match:
            correct += 1
        
        status = "✅" if match else "❌"
        conf = torch.softmax(output, dim=1)[0][pred_idx].item()
        print(f"  {status} {img_path.name:<35} GT: {gt_species:<18} Pred: {pred_species:<18} ({conf:.2f})")
    
    if total > 0:
        acc = 100 * correct / total
        print(f"\n  Species accuracy on fresh images: {correct}/{total} ({acc:.1f}%)")
    else:
        print("  No images evaluated")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    # Phase 1: Extract & crop
    crop_train_dir = CROP_DIR / "train"
    existing = sum(1 for _ in crop_train_dir.rglob("*.jpg")) if crop_train_dir.exists() else 0
    
    if existing >= 1000:
        print(f"✅ Found {existing:,} existing crops, skipping extraction")
        print("   (Delete datasets/species_cls/cropped/ to re-extract)")
    else:
        extract_and_crop()
    
    # Phase 2: Train
    train_mobilenet()
    
    # Phase 3: Quick eval on fresh images
    evaluate_on_fresh()
    
    print(f"\n🏁 Done! Model saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
