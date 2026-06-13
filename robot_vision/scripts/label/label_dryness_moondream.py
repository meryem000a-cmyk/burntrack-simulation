#!/usr/bin/env python3
"""Label eval_images as dry/alive using Moondream2, save to CSV."""

import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

EVAL_DIR = Path("eval_images")
LABELS_CSV = Path("eval_images/dryness_labels.csv")
PROMPT = "Is this plant dry and dead, or green and alive? Answer 'dry' or 'alive'."

BATCH_SIZE = 1  # Moondream encodes one image at a time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"  Loading Moondream2 on {DEVICE}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", trust_remote_code=True, revision="2024-08-26"
    ).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-08-26")
    model.eval()
    return model, tokenizer


def load_existing_labels():
    labels = {}
    if LABELS_CSV.exists():
        with open(LABELS_CSV) as f:
            for row in csv.DictReader(f):
                labels[row["path"]] = row["label"]
    return labels


def save_labels(labels, mode="a"):
    with open(LABELS_CSV, mode, newline="") as f:
        w = csv.writer(f)
        if mode == "w":
            w.writerow(["path", "species", "label"])
        for path, label in labels.items():
            w.writerow([path, path.split("/")[1], label])


def main():
    print("=" * 60, flush=True)
    print("  Label eval images with Moondream2", flush=True)
    print("=" * 60, flush=True)

    model, tokenizer = load_model()
    existing = load_existing_labels()
    print(f"  Already labeled: {len(existing)}", flush=True)

    images = sorted(Path(EVAL_DIR).rglob("*.jpg"))
    to_label = [str(p) for p in images if str(p) not in existing]
    print(f"  Total images: {len(images)}", flush=True)
    print(f"  Need to label: {len(to_label)}", flush=True)

    if not to_label:
        print("  Nothing to do.", flush=True)
        return

    new_labels = {}
    for i, path in enumerate(to_label):
        try:
            img = Image.open(path).convert("RGB")
            enc = model.encode_image(img)
            answer = model.answer_question(enc, PROMPT, tokenizer).strip().lower()
        except Exception as e:
            print(f"  [{i+1}/{len(to_label)}] ERROR {path}: {e}", flush=True)
            answer = "error"

        label = "dry" if "dry" in answer else "alive"
        new_labels[path] = label

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(to_label)}] {label:5s}  {path}", flush=True)

        if (i + 1) % 200 == 0:
            save_labels(new_labels, mode="a")
            new_labels.clear()
            print(f"  Saved checkpoint at {i+1}/{len(to_label)}", flush=True)

    if new_labels:
        save_labels(new_labels, mode="a")

    all_labels = load_existing_labels()
    dry_count = sum(1 for v in all_labels.values() if v == "dry")
    print(f"\n  Done. {len(all_labels)} labeled ({dry_count} dry, {len(all_labels) - dry_count} alive)", flush=True)


if __name__ == "__main__":
    main()
