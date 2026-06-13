#!/usr/bin/env python3
"""
Kaggle Notebook: Train NanoFloraVLM on 2x T4 GPUs
====================================================
Paste this entire file into a Kaggle notebook cell.

Required Kaggle Dataset: flora_balanced_vlm.zip (balanced/ + vlm_distill/ + nano_flora_vlm.py)
"""

import json, random, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Paths ──
WORKING = Path("/kaggle/working")
DATA = Path("/kaggle/input/datasets/anwarmounir67/vlm-nano")
BALANCED_DIR = DATA / "balanced"
VLM_DISTILL_DIR = WORKING / "vlm_distill"
MODEL_DIR = WORKING / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VLM_DISTILL_DIR.mkdir(parents=True, exist_ok=True)

# Copy tokenizer files from read-only input to writable working
import shutil
for f in (DATA / "vlm_distill").iterdir():
    if f.is_file():
        shutil.copy2(f, VLM_DISTILL_DIR / f.name)

# ── Config ──
PROMPTS = [
    "What species of plant is this?",
    "What plant is this?",
    "Identify this plant.",
    "Is this plant dry or alive?",
    "What is the condition of this plant?",
    "Identify this plant and its condition.",
    "What species is this and is it dry?",
    "Name this plant species.",
]
BATCH_SIZE = 64
NUM_WORKERS = 0

MAX_SEQ_LEN = 32
STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 40
STAGE1_LR = 1e-3
STAGE2_LR = 5e-5
WEIGHT_DECAY = 0.01
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

# ── Import NanoFloraVLM ──
import importlib.util
spec = importlib.util.spec_from_file_location("nano_flora_vlm", DATA / "nano_flora_vlm.py")
vlm_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vlm_mod)
NanoFloraVLM = vlm_mod.NanoFloraVLM
FloraTokenizer = vlm_mod.FloraTokenizer
DEFAULT_CONFIG = vlm_mod.DEFAULT_CONFIG

# ── Build tokenizer ──
print("\n[1/4] Building tokenizer...")
src = (DATA / "build_flora_tokenizer.py").read_text()
src = src.replace('OUTPUT_DIR = Path("datasets/vlm_distill")', f'OUTPUT_DIR = Path("{VLM_DISTILL_DIR}")')
exec(src)

# ── Generate pairs from balanced folder ──
print("\n[2/4] Generating training pairs from balanced dataset...")
tokenizer = FloraTokenizer(tokenizer_dir=str(VLM_DISTILL_DIR))

pairs = []
for cls_dir in sorted(BALANCED_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue
    parts = cls_dir.name.rsplit("_", 1)
    if len(parts) != 2 or parts[1] not in ("dry", "not_dry"):
        continue
    species, dryness = parts[0], parts[1]
    answer = f"species: {species}, dry: {'yes' if dryness == 'dry' else 'no'}"
    for img_path in cls_dir.glob("*.jpg"):
        prompt = random.choice(PROMPTS)
        pairs.append({"image_path": str(img_path), "prompt": prompt, "answer": answer})

print(f"  {len(pairs)} pairs generated")

# Split
random.shuffle(pairs)
split = int(len(pairs) * 0.9)
train_pairs, val_pairs = pairs[:split], pairs[split:]

# ── Dataset (pre-decoded via memmap — zero CPU during training) ──
class VLMDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len, augment=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment
        self.texts = []

        mmap_path = WORKING / "vlm_distill" / f"images_{'train' if augment else 'val'}.mmap"
        N = len(pairs)

        if mmap_path.exists():
            print(f"  Loading cached {mmap_path.name}...", flush=True)
        else:
            print(f"  Pre-decoding {N} images to memmap (one-time CPU cost)...", end=" ", flush=True)
            t0 = time.time()
            mmap = np.memmap(mmap_path, dtype="float16", mode="write", shape=(N, 3, 224, 224))
            for i, pair in enumerate(pairs):
                img = Image.open(pair["image_path"]).convert("RGB")
                img = img.resize((224, 224))
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                mmap[i] = img.numpy().astype("float16")
            mmap.flush()
            del mmap
            print(f"done ({time.time()-t0:.0f}s)")

        self.mmap = np.memmap(mmap_path, dtype="float16", mode="readonly", shape=(N, 3, 224, 224))

        for pair in pairs:
            prompt_ids = tokenizer.encode(pair["prompt"], add_special=False)
            answer_ids = tokenizer.encode(pair["answer"], add_special=False)
            full_ids = [tokenizer.bos_id] + prompt_ids + answer_ids + [tokenizer.eos_id]
            n_prompt = 1 + len(prompt_ids)
            labels = [-100] * n_prompt + answer_ids + [tokenizer.eos_id]
            full_ids = tokenizer.pad_sequence(full_ids, max_len)
            labels = (labels + [-100] * max_len)[:max_len]
            self.texts.append((full_ids, labels))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.mmap[idx].copy()).float()
        if self.augment:
            if torch.rand(1).item() > 0.5:
                img = img.flip(-1)
            i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(0.8, 1.0), ratio=(1.0, 1.0))
            img = F.interpolate(img[:, i:i+h, j:j+w].unsqueeze(0), size=(224, 224), mode="bilinear").squeeze(0)
            img = img + torch.randn_like(img) * 0.02
        return img, torch.tensor(self.texts[idx][0]), torch.tensor(self.texts[idx][1])

train_ds = VLMDataset(train_pairs, tokenizer, MAX_SEQ_LEN, augment=True)
val_ds = VLMDataset(val_pairs, tokenizer, MAX_SEQ_LEN, augment=False)
print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# ── Build model ──
print("\n[3/4] Building NanoFloraVLM...")
config = dict(DEFAULT_CONFIG)
config["vocab_size"] = tokenizer.vocab_size
config["max_seq_len"] = MAX_SEQ_LEN + config["num_visual_tokens"] + 8

model = NanoFloraVLM(config)
counts = model.count_parameters()
print(f"  Params: {counts['total']:,} ({counts['total_M']:.1f}M)")

# DataParallel for multi-GPU
if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

# ── Training helpers ──
def get_model(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def train_one_epoch(model, loader, criterion, optimizer, scheduler, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_total = len(loader)
    t0 = time.time()

    for batch_idx, (images, tokens, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, tokens)
        n_vis = get_model(model).n_visual_tokens
        shift_logits = logits[:, n_vis:-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)

        loss = criterion(shift_logits, shift_labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % max(1, n_total // 20) == 0:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed
            print(f"    [{batch_idx+1}/{n_total}] loss: {total_loss/n_batches:.4f}  "
                  f"{rate:.0f} batch/s  ETA: {(n_total-batch_idx-1)/rate:.0f}s", end="\r")

    print()
    return total_loss / n_batches if n_batches else 0

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    n_batches = 0

    for images, tokens, labels in loader:
        images = images.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, tokens)
        n_vis = get_model(model).n_visual_tokens
        shift_logits = logits[:, n_vis:-1, :].reshape(-1, logits.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)

        loss = criterion(shift_logits, shift_labels)
        total_loss += loss.item()

        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        correct += (preds[mask] == shift_labels[mask]).sum().item()
        total += mask.sum().item()
        n_batches += 1

    return total_loss / n_batches, 100 * correct / total if total else 0

# ── Train ──
print("\n[4/4] Training...")
best_val_loss = float("inf")

# Stage 1: Freeze vision encoder
print("\n  Stage 1: Frozen vision encoder")
for p in get_model(model).vision_encoder.parameters():
    p.requires_grad = False
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable: {trainable:,} params")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=STAGE1_LR, weight_decay=WEIGHT_DECAY,
)
total_steps = len(train_loader) * STAGE1_EPOCHS
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

print(f"\n  {'Ep':>3} {'TrLoss':>8} {'VlLoss':>8} {'VlAcc':>7} {'LR':>10} {'Time':>6}")
for epoch in range(STAGE1_EPOCHS):
    t0 = time.time()
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, STAGE1_EPOCHS)
    vl_loss, vl_acc = validate(model, val_loader, criterion)
    lr = optimizer.param_groups[0]["lr"]
    mark = ""
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        torch.save({"model_state_dict": get_model(model).state_dict(), "config": config},
                   MODEL_DIR / "best_nano_vlm.pt")
        mark = " ⭐"
    print(f"  {epoch+1:>3} {tr_loss:>8.4f} {vl_loss:>8.4f} {vl_acc:>6.1f}% {lr:>10.6f} {time.time()-t0:>5.0f}s{mark}")

# Stage 2: Full fine-tune
print("\n  Stage 2: Full fine-tune")
for p in get_model(model).vision_encoder.parameters():
    p.requires_grad = True
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable: {trainable:,} params")

optimizer = optim.AdamW([
    {"params": [p for n, p in get_model(model).named_parameters() if "vision_encoder" in n], "lr": STAGE2_LR * 0.1},
    {"params": [p for n, p in get_model(model).named_parameters() if "projector" in n], "lr": STAGE2_LR},
    {"params": [p for n, p in get_model(model).named_parameters() if "decoder" in n], "lr": STAGE2_LR},
], weight_decay=WEIGHT_DECAY)

total_steps = len(train_loader) * STAGE2_EPOCHS
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

no_improve = 0
for epoch in range(STAGE2_EPOCHS):
    t0 = time.time()
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, epoch, STAGE2_EPOCHS)
    vl_loss, vl_acc = validate(model, val_loader, criterion)
    lr = optimizer.param_groups[0]["lr"]
    mark = ""
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        no_improve = 0
        torch.save({"model_state_dict": get_model(model).state_dict(), "config": config},
                   MODEL_DIR / "best_nano_vlm.pt")
        mark = " ⭐"
    else:
        no_improve += 1
    print(f"  {STAGE1_EPOCHS+epoch+1:>3} {tr_loss:>8.4f} {vl_loss:>8.4f} {vl_acc:>6.1f}% {lr:>10.8f} {time.time()-t0:>5.0f}s{mark}")
    if no_improve >= 10:
        print("  ⏹ Early stopping")
        break

# ── Export ──
print(f"\n  Best val loss: {best_val_loss:.4f}")
print(f"  Model saved: {MODEL_DIR / 'best_nano_vlm.pt'}")

# Quick generation test
raw_model = get_model(model)
raw_model.load_state_dict(torch.load(MODEL_DIR / "best_nano_vlm.pt", weights_only=False)["model_state_dict"])
raw_model.eval()

dummy_img = torch.randn(1, 3, 224, 224).to(device)
prompt = "What plant is this?"
prompt_ids = tokenizer.encode(prompt, add_special=True)
result = raw_model.generate(dummy_img.cpu(), prompt_ids, tokenizer, max_new_tokens=20, temperature=0.1)
print(f"  Prompt: '{prompt}'")
print(f"  Output: '{result}'")

print("\n✅ Training complete! Download /kaggle/working/models/best_nano_vlm.pt")
