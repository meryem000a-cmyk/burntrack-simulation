#!/usr/bin/env python3
"""
Kaggle Training Script: ConvNeXt-Small African Flora Classifier
================================================================
Trains a high-capacity ConvNeXt-Small (50M params) on 17 African plant
species using the existing Kaggle 'flora-is-cool' dataset.

WHY ConvNeXt-Small:
  - 50M parameters → serious capacity for fine-grained botanical species
  - ImageNet top-1: 83.6% → strong pretrained features, modern CNN architecture
  - Pure convolutional → deploys cleanly to TFLite INT8 / ONNX (no transformer quirks)
  - Designed with transformer-era tricks (larger kernels, LayerNorm, GELU)

KEY DESIGN DECISIONS:
  1. Maps 34 classes (17 species × dry/not_dry) → 17 pure species.
     The dry/not_dry labels were Gemini auto-generated and unreliable.
  2. Uses full images, NOT crops — because 88% of bounding boxes were
     dummy (0.5 0.5 1.0 1.0), so cropping gives the same image.
  3. CutMix + MixUp augmentation for 2-4% accuracy boost.
  4. Two-stage training: frozen backbone → full fine-tune with differential LR.

Data Path on Kaggle:
  /kaggle/input/flora-is-cool/content/yolo_flora

Output:
  /kaggle/working/models/best_classifier.pt
  /kaggle/working/models/classifier.onnx

ALTERNATIVE BACKBONES (edit MODEL_NAME below to switch):
  - 'convnext_small.fb_in22k_ft_in1k'  ← DEFAULT (50M, 83.6% IN1k)
  - 'convnext_base.fb_in22k_ft_in1k'   ← Bigger (89M, 85.8% IN1k)
  - 'swin_s3_small_224.ms_in1k'        ← Swin Transformer (50M, 83.7%)
  - 'vit_small_patch14_dinov2.lvd142m'  ← DINOv2 ViT-S (22M, best features)
"""

import io
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image, ImageFile
import warnings

# Handle massive high-res images and prevent DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

# Kaggle/Local dataset paths auto-detection
import sys
import zipfile

DATASET_BASE = None

# Check if GCS dataset is passed via command line arguments (for Vertex AI)
gcs_data_uri = None
for i, arg in enumerate(sys.argv):
    if arg == "--data" and i + 1 < len(sys.argv):
        gcs_data_uri = sys.argv[i + 1]
        break

if gcs_data_uri and gcs_data_uri.startswith("gs://"):
    from google.cloud import storage
    print(f"📥 Downloading dataset from GCS: {gcs_data_uri}...")
    parts = gcs_data_uri.replace("gs://", "").split("/")
    bucket_name = parts[0]
    blob_name = "/".join(parts[1:])
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    os.makedirs("/tmp", exist_ok=True)
    zip_path = "/tmp/dataset.zip"
    blob.download_from_filename(zip_path)
    
    DATASET_BASE = Path("/tmp/yolo_flora")
    DATASET_BASE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATASET_BASE)
    print(f"  ✅ Dataset extracted to: {DATASET_BASE}")
else:
    potential_bases = [
        Path("/kaggle/input/africanplantclass/yolo_flora"),
        Path("/kaggle/input/african-plant-class/yolo_flora"),
        Path("/kaggle/input/datasets/anwarmounir67/africanplantclass/yolo_flora"),
        Path("/kaggle/input/flora-is-cool/content/yolo_flora"),
        Path("/kaggle/input/yolo_flora"),
        Path("./yolo_flora"),
        Path("../yolo_flora"),
        Path("/home/anwar/Documents/Vision/datasets/yolo_flora")
    ]

    # Quick search to verify which path actually exists
    for base in potential_bases:
        if (base / "images" / "train").exists():
            DATASET_BASE = base
            break

    if DATASET_BASE is None:
        # Broad search in /kaggle/input for 'yolo_flora'
        import glob
        matching_dirs = glob.glob("/kaggle/input/**/yolo_flora", recursive=True)
        if matching_dirs:
            DATASET_BASE = Path(matching_dirs[0])

    if DATASET_BASE is None:
        # Bulletproof fallback: search for 'images/train' folder anywhere in /kaggle/input
        import glob
        matching_train_dirs = glob.glob("/kaggle/input/**/images/train", recursive=True)
        if matching_train_dirs:
            # Set base to the parent of images
            DATASET_BASE = Path(matching_train_dirs[0]).parent.parent
            print(f"🎯 Auto-detected dataset layout parent at: {DATASET_BASE}")

    if DATASET_BASE is None:
        # Default fallback
        DATASET_BASE = Path("/kaggle/input/africanplantclass/yolo_flora")

print(f"📂 Resolved DATASET_BASE to: {DATASET_BASE}")

IMAGES_TRAIN = DATASET_BASE / "images" / "train"
IMAGES_VAL = DATASET_BASE / "images" / "val"
LABELS_TRAIN = DATASET_BASE / "labels" / "train"
LABELS_VAL = DATASET_BASE / "labels" / "val"

# Output directory setup
if os.path.exists("/kaggle/working"):
    WORK_DIR = Path("/kaggle/working")
else:
    WORK_DIR = Path("./output")

MODEL_DIR = WORK_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
print(f"💾 Saving outputs to: {MODEL_DIR}")

# The 17 African plant species (mapping class_id // 2 → species)
# Original: 0=adansonia_not_dry, 1=adansonia_dry, 2=acacia_not_dry, ...
SPECIES = [
    "adansonia", "acacia", "vachellia", "senegalia", "combretum",
    "brachystegia", "colophospermum", "ficus", "khaya", "macaranga",
    "euphorbia", "aloe", "protea", "erica", "themeda", "andropogon", "tamarix",
]
NUM_SPECIES = len(SPECIES)  # 17

# Training hyperparameters
IMG_SIZE = 224          # ConvNeXt native resolution
MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"  # 89M params, 85.8% ImageNet — max accuracy
BATCH_SIZE = 32         # Reduced to 32 to guarantee no VRAM OOM
NUM_WORKERS = 2         # Reduced to 2 to prevent System RAM OOM from multiprocessing
STAGE1_EPOCHS = 5       # Head-only warmup
STAGE2_EPOCHS = 60      # Full fine-tune
STAGE1_LR = 0.005
STAGE2_LR = 0.0003
WEIGHT_DECAY = 0.02
LABEL_SMOOTHING = 0.1
CUTMIX_ALPHA = 1.0
MIXUP_ALPHA = 0.2
PATIENCE = 15           # Early stopping patience
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("⚠️  WARNING: No GPU detected! Training will be extremely slow.")
    print("   In Kaggle: Settings → Accelerator → GPU T4 x2")


# ──────────────────────────────────────────────────────────────────────
# Dataset: Read images + map 34 YOLO classes → 17 species
# ──────────────────────────────────────────────────────────────────────

class FloraSpeciesDataset(Dataset):
    """
    Reads images from the YOLO flora dataset and maps the 34-class
    (species × dry/not_dry) labels down to 17 pure species.

    Since 88% of bounding boxes are dummy (full-image), we skip cropping
    entirely and use the full image for classification.
    """

    def __init__(self, images_dir, labels_dir, transform=None):
        self.transform = transform
        self.samples = []  # list of (image_path, species_id)

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)

        if not images_dir.exists():
            print(f"⚠️  WARNING: Images directory does not exist: {images_dir}")
            return
        if not labels_dir.exists():
            print(f"⚠️  WARNING: Labels directory does not exist: {labels_dir}")
            return

        for img_path in sorted(images_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue

            lbl_path = labels_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            try:
                text = lbl_path.read_text().strip()
                if not text:
                    continue
                class_id = int(text.split("\n")[0].split()[0])
                species_id = class_id // 2  # 0,1→0  2,3→1  etc.
                if 0 <= species_id < NUM_SPECIES:
                    self.samples.append((str(img_path), species_id))
            except (ValueError, IndexError):
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            # Immediately shrink massive images before PyTorch allocates huge tensors
            if img.width > 1024 or img.height > 1024:
                img.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
        except Exception:
            # If an image is corrupt, return a blank black image so training doesn't crash
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────────
# CutMix / MixUp Augmentation
# ──────────────────────────────────────────────────────────────────────

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def apply_cutmix_or_mixup(images, labels, num_classes):
    """Randomly apply CutMix or MixUp to a batch. Returns mixed images
    and soft label targets (one-hot)."""

    batch_size = images.size(0)
    # Convert labels to one-hot with label smoothing
    targets = torch.zeros(batch_size, num_classes, device=images.device)
    targets.scatter_(1, labels.unsqueeze(1), 1.0)
    # Apply label smoothing
    targets = targets * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / num_classes

    # 50% chance CutMix, 30% chance MixUp, 20% chance neither
    r = random.random()
    if r < 0.5:
        # CutMix
        lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
        rand_index = torch.randperm(batch_size, device=images.device)
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
        # Adjust lambda based on actual area ratio
        lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        targets_b = torch.zeros(batch_size, num_classes, device=images.device)
        targets_b.scatter_(1, labels[rand_index].unsqueeze(1), 1.0)
        targets_b = targets_b * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / num_classes
        targets = lam * targets + (1 - lam) * targets_b
    elif r < 0.8:
        # MixUp
        lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
        rand_index = torch.randperm(batch_size, device=images.device)
        images = lam * images + (1 - lam) * images[rand_index]
        targets_b = torch.zeros(batch_size, num_classes, device=images.device)
        targets_b.scatter_(1, labels[rand_index].unsqueeze(1), 1.0)
        targets_b = targets_b * (1.0 - LABEL_SMOOTHING) + LABEL_SMOOTHING / num_classes
        targets = lam * targets + (1 - lam) * targets_b

    return images, targets


# ──────────────────────────────────────────────────────────────────────
# Model: EfficientNet-V2-S via timm
# ──────────────────────────────────────────────────────────────────────

def build_model(num_classes):
    """Build ConvNeXt-Small (or alternative backbone) with a custom classification head."""
    try:
        import timm
    except ImportError:
        os.system("pip install -q timm")
        import timm

    print(f"  Loading backbone: {MODEL_NAME}")
    try:
        model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=0,  # Remove original head
        )
    except Exception as e:
        print(f"⚠️  WARNING: Failed to load pretrained weights for {MODEL_NAME}: {e}")
        print("   If you are running in Kaggle, please make sure to toggle 'Internet On' in the right sidebar settings panel.")
        print("   Attempting to initialize model with random weights (no pretraining)...")
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False,
            num_classes=0,  # Remove original head
        )

    # Get feature dimension from the backbone
    with torch.no_grad():
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        feat_dim = model(dummy).shape[1]

    # Custom head with dropout for regularization
    classifier = nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.GELU(),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes),
    )

    return model, classifier, feat_dim


class FullModel(nn.Module):
    """Wraps backbone + classifier into a single nn.Module for clean
    training, export, and DataParallel support."""

    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ──────────────────────────────────────────────────────────────────────
# Training & Validation Loops
# ──────────────────────────────────────────────────────────────────────

# Pre-defined tensors for GPU normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def train_one_epoch(model, loader, optimizer, device, num_classes, use_cutmix=True):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    num_batches = len(loader)
    
    # GPU-accelerated Random Erasing
    gpu_erasing = transforms.RandomErasing(p=0.15, scale=(0.02, 0.15))

    for i, (images, labels) in enumerate(loader):
        # 1. Move to GPU immediately
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # 2. Fast GPU Normalization (relieves CPU)
        mean = MEAN.to(device)
        std = STD.to(device)
        images = (images - mean) / std
        
        # 3. Apply GPU Erasing
        images = gpu_erasing(images)

        if use_cutmix:
            images, soft_targets = apply_cutmix_or_mixup(images, labels, num_classes)
            out = model(images)
            log_probs = torch.nn.functional.log_softmax(out, dim=1)
            loss = -(soft_targets * log_probs).sum(dim=1).mean()
            _, pred = out.max(1)
            _, hard = soft_targets.max(1)
            correct += pred.eq(hard).sum().item()
        else:
            out = model(images)
            loss = nn.functional.cross_entropy(out, labels, label_smoothing=LABEL_SMOOTHING)
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        # Print progress every 50 batches
        if (i + 1) % 50 == 0 or (i + 1) == num_batches:
            current_loss = running_loss / total
            current_acc = 100.0 * correct / total
            print(f"    Batch {i+1:>3}/{num_batches} | Loss: {current_loss:.4f} | Acc: {current_acc:.1f}%", end="\r", flush=True)
    
    print(" " * 60, end="\r")  # Clear the batch progress line
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    per_cls_correct = defaultdict(int)
    per_cls_total = defaultdict(int)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Fast GPU Normalization
        mean = MEAN.to(device)
        std = STD.to(device)
        images = (images - mean) / std
        
        out = model(images)
        loss = nn.functional.cross_entropy(out, labels)
        running_loss += loss.item() * images.size(0)
        _, pred = out.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

        for p, t in zip(pred, labels):
            per_cls_total[t.item()] += 1
            if p == t:
                per_cls_correct[t.item()] += 1

    return running_loss / total, 100.0 * correct / total, per_cls_correct, per_cls_total


# ──────────────────────────────────────────────────────────────────────
# Data Loaders
# ──────────────────────────────────────────────────────────────────────

def get_dataloaders():
    # CPU does only the bare minimum: Resize, Flip, and ByteTensor conversion
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)), # Fixed resize is much faster than RandomResizedCrop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Note: Normalization and Erasing moved to GPU in train_one_epoch!
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    # Build base datasets (reads file list + labels)
    print("\n  📂 Scanning dataset directories...")
    base_train = FloraSpeciesDataset(IMAGES_TRAIN, LABELS_TRAIN)
    base_val = FloraSpeciesDataset(IMAGES_VAL, LABELS_VAL)

    print(f"  Found {len(base_train)} train / {len(base_val)} val samples across {NUM_SPECIES} species")

    # Print class distribution
    train_dist = defaultdict(int)
    for _, sid in base_train.samples:
        train_dist[sid] += 1
    print("\n  Class distribution (train):")
    for sid in range(NUM_SPECIES):
        name = SPECIES[sid]
        count = train_dist.get(sid, 0)
        bar = "█" * (count // 100)
        print(f"    {name:<20} {count:>5}  {bar}")

    # Assign transforms directly to the disk datasets
    train_ds = base_train
    train_ds.transform = train_transform
    
    val_ds = base_val
    val_ds.transform = val_transform

    # Weighted sampling to handle class imbalance
    class_counts = defaultdict(int)
    for _, lbl in train_ds.samples:
        class_counts[lbl] += 1
    total = sum(class_counts.values())
    weights = {k: total / (NUM_SPECIES * v) for k, v in class_counts.items() if v > 0}
    sample_weights = [weights.get(lbl, 1.0) for _, lbl in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────
# Main Training Pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print(f"  {MODEL_NAME} African Flora Classifier")
    print(f"  Device: {DEVICE}  |  Species: {NUM_SPECIES}  |  Image Size: {IMG_SIZE}")
    print("=" * 65)

    # 1. Data
    train_loader, val_loader = get_dataloaders()

    # 2. Model
    print(f"\n  🏗️  Building {MODEL_NAME} backbone...")
    backbone, classifier, feat_dim = build_model(NUM_SPECIES)
    model = FullModel(backbone, classifier)

    # DataParallel for multi-GPU (Kaggle T4 x2)
    if torch.cuda.device_count() > 1:
        print(f"  🚀 Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    model = model.to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  ✅ Model loaded: {param_count:.1f}M parameters (feat_dim={feat_dim})")

    best_val_acc = 0.0
    best_epoch = -1

    # ── STAGE 1: Head-only warmup ──────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  [Stage 1] Head-only warmup ({STAGE1_EPOCHS} epochs, LR={STAGE1_LR})")
    print(f"  {'─'*55}")

    # Freeze backbone
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    for param in base_model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS, eta_min=1e-5)

    for epoch in range(STAGE1_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, DEVICE, NUM_SPECIES, use_cutmix=False)
        vl_loss, vl_acc, _, _ = validate(model, val_loader, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        mark = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = epoch
            state = base_model.state_dict()
            torch.save({"model_state_dict": state, "species": SPECIES, "val_acc": vl_acc, "epoch": epoch}, MODEL_DIR / "best_classifier.pt")
            mark = " ⭐"

        print(f"  Ep {epoch+1:>2}/{STAGE1_EPOCHS} | TrLoss: {tr_loss:.4f} TrAcc: {tr_acc:>5.1f}% | VlLoss: {vl_loss:.4f} VlAcc: {vl_acc:>5.1f}% | {elapsed:>3.0f}s{mark}")

    # ── STAGE 2: Full fine-tune ────────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  [Stage 2] Full fine-tune ({STAGE2_EPOCHS} epochs, LR={STAGE2_LR})")
    print(f"  CutMix α={CUTMIX_ALPHA}  MixUp α={MIXUP_ALPHA}")
    print(f"  {'─'*55}")

    # Unfreeze everything
    for param in base_model.parameters():
        param.requires_grad = True

    # Differential learning rate: backbone 10x lower than head
    backbone_params = list(base_model.backbone.parameters())
    head_params = list(base_model.classifier.parameters())
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": STAGE2_LR * 0.1},
        {"params": head_params, "lr": STAGE2_LR},
    ], weight_decay=WEIGHT_DECAY)

    # Cosine annealing with 3-epoch warmup
    def lr_lambda(epoch):
        warmup_epochs = 3
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, STAGE2_EPOCHS - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    no_improve = 0
    for epoch in range(STAGE2_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, DEVICE, NUM_SPECIES, use_cutmix=True)
        vl_loss, vl_acc, per_c, per_t = validate(model, val_loader, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        mark = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_epoch = STAGE1_EPOCHS + epoch
            no_improve = 0
            state = base_model.state_dict()
            torch.save({"model_state_dict": state, "species": SPECIES, "val_acc": vl_acc, "epoch": best_epoch}, MODEL_DIR / "best_classifier.pt")
            mark = " ⭐"
        else:
            no_improve += 1

        current_lr = optimizer.param_groups[1]["lr"]
        print(f"  Ep {epoch+1:>2}/{STAGE2_EPOCHS} | TrLoss: {tr_loss:.4f} TrAcc: {tr_acc:>5.1f}% | VlLoss: {vl_loss:.4f} VlAcc: {vl_acc:>5.1f}% | LR: {current_lr:.6f} | {elapsed:>3.0f}s{mark}")

        # Early stopping
        if no_improve >= PATIENCE:
            print(f"  ⏹ Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

    # ── Final Evaluation ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL RESULTS")
    print(f"{'='*65}")

    # Reload best checkpoint
    ckpt = torch.load(MODEL_DIR / "best_classifier.pt", weights_only=False)
    base_model.load_state_dict(ckpt["model_state_dict"])
    _, final_acc, per_c, per_t = validate(model, val_loader, DEVICE)

    print(f"  Best Val Accuracy: {best_val_acc:.1f}% (epoch {best_epoch+1})")
    print(f"\n  Per-species breakdown:")
    print(f"  {'Species':<20} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
    print(f"  {'─'*47}")
    for idx in range(NUM_SPECIES):
        c = per_c.get(idx, 0)
        t = per_t.get(idx, 0)
        acc = 100 * c / t if t > 0 else 0
        bar = "█" * int(acc / 5)
        print(f"  {SPECIES[idx]:<20} {c:>7} {t:>7} {acc:>7.1f}%  {bar}")

    # ── Export ONNX ────────────────────────────────────────────────
    print(f"\n  📦 Exporting to ONNX...")
    try:
        base_model.eval()
        base_model_cpu = base_model.cpu()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        onnx_path = str(MODEL_DIR / "classifier.onnx")
        torch.onnx.export(
            base_model_cpu, dummy, onnx_path,
            input_names=["image"], output_names=["species"],
            dynamic_axes={"image": {0: "batch"}, "species": {0: "batch"}},
            opset_version=13,
        )
        print(f"  ✅ ONNX saved: {onnx_path}")
    except Exception as e:
        print(f"  ⚠️ ONNX export failed: {e}")

    # ── Save class mapping ─────────────────────────────────────────
    class_map = {i: name for i, name in enumerate(SPECIES)}
    with open(MODEL_DIR / "species_map.json", "w") as f:
        json.dump(class_map, f, indent=2)
    print(f"  ✅ Class map saved: {MODEL_DIR / 'species_map.json'}")

    print(f"\n{'='*65}")
    print(f"  🎉 DONE — All artifacts in {MODEL_DIR}")
    print(f"  Download best_classifier.pt and classifier.onnx for deployment.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
