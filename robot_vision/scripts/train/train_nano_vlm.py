#!/usr/bin/env python3
"""
Train NanoFloraVLM — CPU-Optimized Training Pipeline
======================================================
Trains the ~25M param micro-VLM on your laptop (Ryzen 5 4500U).

Strategy:
  Stage 1: Freeze vision encoder, train projector + decoder (10 epochs)
  Stage 2: Unfreeze all, lower LR, full fine-tune (40 epochs)

Usage:
    ./flora_env/bin/python train_nano_vlm.py
"""

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from nano_flora_vlm import NanoFloraVLM, FloraTokenizer, DEFAULT_CONFIG

# ── Config ──
PAIRS_FILE = Path("datasets/vlm_distill/pairs.jsonl")
MODEL_DIR = Path("datasets/vlm_distill/models")

BATCH_SIZE = 8
GRAD_ACCUM = 8              # Effective batch = 64
NUM_WORKERS = 4
DEVICE = "cpu"
MAX_SEQ_LEN = 32            # Max answer tokens (structured answers are short)

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 40
STAGE1_LR = 1e-3
STAGE2_LR = 5e-5
WEIGHT_DECAY = 0.01

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset ──

class VLMDistillDataset(Dataset):
    """Dataset of (image, prompt, answer) triplets for VLM training."""

    def __init__(self, pairs_file, tokenizer, split="train", val_ratio=0.1):
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)) if split == "train"
            else transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if split == "train"
            else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05) if split == "train"
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Load all pairs
        all_pairs = []
        with open(pairs_file) as f:
            for line in f:
                d = json.loads(line)
                if Path(d["image_path"]).exists():
                    all_pairs.append(d)

        # Shuffle and split
        random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * (1 - val_ratio))
        self.pairs = all_pairs[:split_idx] if split == "train" else all_pairs[split_idx:]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load image
        try:
            img = Image.open(pair["image_path"]).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, 224, 224)

        # Tokenize: [BOS] prompt [SEP] answer [EOS]
        prompt_ids = self.tokenizer.encode(pair["prompt"], add_special=False)
        answer_ids = self.tokenizer.encode(pair["answer"], add_special=False)

        # Full sequence: BOS + prompt + answer + EOS
        full_ids = [self.tokenizer.bos_id] + prompt_ids + answer_ids + [self.tokenizer.eos_id]

        # Create labels: -100 for prompt tokens (don't train on questions), real IDs for answer
        # The model should predict the answer tokens given visual + prompt tokens
        n_prompt = 1 + len(prompt_ids)  # BOS + prompt
        labels = [-100] * n_prompt + answer_ids + [self.tokenizer.eos_id]

        # Pad/truncate
        full_ids = self.tokenizer.pad_sequence(full_ids, MAX_SEQ_LEN)
        labels = labels[:MAX_SEQ_LEN]
        labels = labels + [-100] * (MAX_SEQ_LEN - len(labels))

        return img, torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ── Training ──

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, grad_accum):
    model.train()
    total_loss, total_tokens, n_batches = 0.0, 0, 0
    optimizer.zero_grad()

    for batch_idx, (images, tokens, labels) in enumerate(loader):
        images = images.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device)

        logits = model(images, tokens)

        # Shift logits and labels for next-token prediction
        # logits: [B, n_vis + seq_len, vocab]
        # We only care about the text portion
        n_vis = model.n_visual_tokens
        text_logits = logits[:, n_vis:, :]      # [B, seq_len, vocab]
        shift_logits = text_logits[:, :-1, :]    # [B, seq_len-1, vocab]
        shift_labels = labels[:, 1:]             # [B, seq_len-1]

        loss = criterion(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        loss = loss / grad_accum
        loss.backward()

        if (batch_idx + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * grad_accum
        valid_tokens = (shift_labels != -100).sum().item()
        total_tokens += valid_tokens
        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    correct, total = 0, 0

    for images, tokens, labels in loader:
        images = images.to(device)
        tokens = tokens.to(device)
        labels = labels.to(device)

        logits = model(images, tokens)
        n_vis = model.n_visual_tokens
        text_logits = logits[:, n_vis:, :]
        shift_logits = text_logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        loss = criterion(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        total_loss += loss.item()
        n_batches += 1

        # Token accuracy (on non-masked positions)
        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        correct += (preds[mask] == shift_labels[mask]).sum().item()
        total += mask.sum().item()

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    accuracy = 100 * correct / total if total > 0 else 0
    return avg_loss, accuracy


def main():
    print("=" * 60)
    print("  NanoFloraVLM Training Pipeline")
    print("=" * 60)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    print("\n  Loading tokenizer...")
    tokenizer = FloraTokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Update config with actual vocab size
    config = dict(DEFAULT_CONFIG)
    config["vocab_size"] = tokenizer.vocab_size
    config["max_seq_len"] = MAX_SEQ_LEN + config["num_visual_tokens"] + 8  # safety margin

    # Load datasets
    print("  Loading dataset...")
    train_ds = VLMDistillDataset(PAIRS_FILE, tokenizer, split="train")
    val_ds = VLMDistillDataset(PAIRS_FILE, tokenizer, split="val")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Build model
    print("\n  Building NanoFloraVLM...")
    model = NanoFloraVLM(config)
    counts = model.count_parameters()
    print(f"  Total parameters: {counts['total']:,} ({counts['total_M']:.1f}M)")
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    best_val_loss = float("inf")

    # ── Stage 1: Freeze vision, train head ──
    print(f"\n  [Stage 1] Frozen vision encoder ({STAGE1_EPOCHS} epochs)")
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR, weight_decay=WEIGHT_DECAY,
    )
    total_steps_s1 = (len(train_loader) // GRAD_ACCUM) * STAGE1_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_s1, eta_min=1e-5)

    print(f"\n  {'Ep':>3} {'TrLoss':>8} {'VlLoss':>8} {'VlAcc':>7} {'LR':>10} {'Time':>6}")
    print(f"  {'-'*3} {'-'*8} {'-'*8} {'-'*7} {'-'*10} {'-'*6}")

    for epoch in range(STAGE1_EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE, GRAD_ACCUM)
        vl_loss, vl_acc = validate(model, val_loader, criterion, DEVICE)
        lr = optimizer.param_groups[0]["lr"]
        mark = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                         "tokenizer_dir": str(tokenizer.tokenizer_dir)},
                        MODEL_DIR / "best_nano_vlm.pt")
            mark = " ⭐"
        elapsed = time.time() - t0
        print(f"  {epoch+1:>3} {tr_loss:>8.4f} {vl_loss:>8.4f} {vl_acc:>6.1f}% {lr:>10.6f} {elapsed:>5.0f}s{mark}")

    # ── Stage 2: Full fine-tune ──
    print(f"\n  [Stage 2] Full fine-tune ({STAGE2_EPOCHS} epochs)")
    for p in model.vision_encoder.parameters():
        p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "vision_encoder" in n], "lr": STAGE2_LR * 0.1},
        {"params": [p for n, p in model.named_parameters() if "projector" in n], "lr": STAGE2_LR},
        {"params": [p for n, p in model.named_parameters() if "decoder" in n], "lr": STAGE2_LR},
    ], weight_decay=WEIGHT_DECAY)

    total_steps_s2 = (len(train_loader) // GRAD_ACCUM) * STAGE2_EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps_s2, eta_min=1e-7)

    no_improve = 0
    for epoch in range(STAGE2_EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, DEVICE, GRAD_ACCUM)
        vl_loss, vl_acc = validate(model, val_loader, criterion, DEVICE)
        lr = optimizer.param_groups[0]["lr"]
        mark = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                         "tokenizer_dir": str(tokenizer.tokenizer_dir)},
                        MODEL_DIR / "best_nano_vlm.pt")
            mark = " ⭐"
        else:
            no_improve += 1
        elapsed = time.time() - t0
        print(f"  {STAGE1_EPOCHS+epoch+1:>3} {tr_loss:>8.4f} {vl_loss:>8.4f} {vl_acc:>6.1f}% {lr:>10.8f} {elapsed:>5.0f}s{mark}")
        if no_improve >= 10:
            print("  ⏹ Early stopping.")
            break

    # ── Save final ──
    print(f"\n  ✅ Best val loss: {best_val_loss:.4f}")
    print(f"  📁 Model saved: {MODEL_DIR / 'best_nano_vlm.pt'}")

    # Quick generation test
    print(f"\n  Generation test:")
    checkpoint = torch.load(MODEL_DIR / "best_nano_vlm.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_img = torch.randn(1, 3, 224, 224)
    prompt = "What plant is this?"
    prompt_ids = tokenizer.encode(prompt, add_special=True)
    result = model.generate(dummy_img, prompt_ids, tokenizer, max_new_tokens=20, temperature=0.1)
    print(f"  Prompt: '{prompt}'")
    print(f"  Output: '{result}'")
    print(f"\n🏁 Training complete!")


if __name__ == "__main__":
    main()
