#!/usr/bin/env python3
"""
Create a balanced dataset:
1. Subsample yolo_flora_cls to median class count.
2. Use Moondream to label unlabeled images → fill classes below target.
Output: datasets/balanced/{class}/
"""

import random, shutil, json, time
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image

TARGET = Path("datasets/balanced")
CLS_DIR = Path("datasets/yolo_flora_cls/train")
SOURCES = [
    Path("datasets/yolo_flora/images/train"),
    Path("datasets/boost_staging"),
    Path("datasets/fresh_eval/images"),
]
SPECIES = [
    "acacia", "adansonia", "aloe", "andropogon", "baobab",
    "brachystegia", "colophospermum", "combretum", "erica",
    "euphorbia", "ficus", "khaya", "macaranga", "protea",
    "senegalia", "tamarix", "themeda", "vachellia",
]


def get_classes():
    return sorted([d.name for d in CLS_DIR.iterdir() if d.is_dir()])


def subsample(classes, target):
    for cls in classes:
        imgs = list((CLS_DIR / cls).glob("*.jpg"))
        take = min(len(imgs), target)
        if take == 0:
            continue
        (TARGET / cls).mkdir(parents=True, exist_ok=True)
        for img in random.sample(imgs, take):
            shutil.copy2(img, TARGET / cls / img.name)
        if take < target:
            print(f"  ⚠  {cls:<30} only {take} existing, need {target - take} more")


def load_moondream():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "vikhyatk/moondream2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=True,
        device_map="cpu", torch_dtype=torch.float32,
    )
    model.eval()
    return model, tokenizer


def query(model, tokenizer, path, prompt):
    img = Image.open(path).convert("RGB")
    gen = model.query(img, prompt, tokenizer)["answer"]
    return "".join(gen).strip()


def classify_species(model, tokenizer, path):
    a = query(model, tokenizer, path, "What species of plant is this? Answer with one word.")
    return a.lower()


def classify_dryness(model, tokenizer, path):
    a = query(model, tokenizer, path, "Is this plant dry, dead, or brown? Answer YES or NO.")
    return "yes" if "YES" in a.upper() else "no"


def extract_species(answer):
    for s in SPECIES:
        if s in answer:
            return s
    return None


def main():
    print("=" * 60)
    print("  Balance — Subsample + Moondream Label")
    print("=" * 60)

    classes = get_classes()
    counts = {c: len(list((CLS_DIR / c).glob("*.jpg"))) for c in classes}
    target = sorted(counts.values())[len(counts) // 2]
    print(f"\n  Target per class: {target} (median)")

    if TARGET.exists():
        shutil.rmtree(TARGET)

    subsample(classes, target)

    deficit = {c: max(0, target - len(list((TARGET / c).glob("*.jpg")))) for c in classes}
    total_needed = sum(deficit.values())
    if total_needed == 0:
        print(f"\n  ✅ Balanced from existing data only. Done!")
        return

    print(f"\n  Loading Moondream...")
    model, tokenizer = load_moondream()

    existing = {f.name for c in classes for f in (TARGET / c).glob("*.jpg")}
    candidates = []
    for src in SOURCES:
        if src.exists():
            candidates.extend(src.glob("*.jpg"))
    random.shuffle(candidates)
    candidates = [c for c in candidates if c.name not in existing]

    added = 0
    t0 = time.time()
    for i, p in enumerate(candidates):
        if sum(deficit.values()) == 0:
            break
        sp = classify_species(model, tokenizer, p)
        dry = classify_dryness(model, tokenizer, p)
        cls = f"{sp}_{'dry' if dry == 'yes' else 'not_dry'}"
        if cls in deficit and deficit[cls] > 0:
            (TARGET / cls).mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, TARGET / cls / p.name)
            deficit[cls] -= 1
            added += 1
        if (i + 1) % 10 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"  [{i+1}] +{added} labeled, {sum(deficit.values())} left, {r:.1f} img/s", end="\r")

    print(f"\n\n  ✅ Added {added} Moondream-labeled images")
    for c in sorted(classes):
        n = len(list((TARGET / c).glob("*.jpg")))
        print(f"    {c:<30} {n}")


if __name__ == "__main__":
    main()
