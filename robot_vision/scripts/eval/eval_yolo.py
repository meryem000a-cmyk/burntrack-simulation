#!/usr/bin/env python3
"""Evaluate YOLO classification models on eval_images/(species)/"""

from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

EVAL_DIR = Path("../eval_images")
MODELS = [
    ("best(3).pt", "best"),
    ("last(1).pt", "last"),
]

for model_path, label in MODELS:
    print(f"\n{'='*60}")
    print(f"  Model: {label} ({model_path})")
    print(f"{'='*60}")

    model = YOLO(str(model_path))

    correct = 0
    total = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    for cls_dir in sorted(EVAL_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        gt = cls_dir.name
        for img_path in sorted(cls_dir.glob("*.jpg")):
            results = model(str(img_path), verbose=False)
            pred = results[0].names[results[0].probs.top1]
            per_class[gt]["total"] += 1
            total += 1
            if pred == gt:
                per_class[gt]["correct"] += 1
                correct += 1

    print(f"\n  {'Species':20s} {'Correct':>8} {'Total':>6} {'Acc':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*6}")
    for cls_name in sorted(per_class.keys()):
        c = per_class[cls_name]["correct"]
        t = per_class[cls_name]["total"]
        acc = 100 * c / t if t else 0
        print(f"  {cls_name:20s} {c:>8} {t:>6} {acc:>5.1f}%")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*6}")
    print(f"  {'Total':20s} {correct:>8} {total:>6} {100*correct/total:>5.1f}%")
