#!/usr/bin/env python3
"""
Evaluate best_nano_vlm.pt on unseen images.

Usage:
    python3 eval_nano_vlm.py --checkpoint /path/to/best_nano_vlm.pt [--data-dir eval_images]

Directory structure expected:
    eval_images/
        acacia/
            img_0001.jpg
            ...
        aloe/
            img_0001.jpg
            ...

Reports per-species accuracy and confusion matrix for species + dryness.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from nano_flora_vlm import NanoFloraVLM, FloraTokenizer, DEFAULT_CONFIG

SPECIES = [
    "acacia", "aloe", "baobab", "cactus", "colophospermum",
    "combretum", "commiphora", "convolvulus", "croton", "euphorbia",
    "grewia", "lycium", "olea", "panicum", "protea",
    "rhigozum", "ziziphus",
]

PROMPTS = [
    "What species of plant is this?",
    "What plant is this?",
    "Identify this plant.",
    "What type of plant do you see?",
    "Is this plant dry or alive?",
    "What is the condition of this plant?",
    "Identify this plant and its condition.",
    "What species is this and is it dry?",
    "Name this plant species.",
    "Describe this plant.",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def parse_prediction(text):
    species = None
    dry = None
    try:
        text = text.replace("\u0120", " ")
        parts = text.replace("\n", " ").split(",")
        for p in parts:
            p = p.strip()
            if "species" in p.lower() and ":" in p:
                val = p.split(":", 1)[1].strip().lower()
                species = val.split()[0] if val.split() else None
            elif "dry" in p.lower() and ":" in p:
                val = p.split(":", 1)[1].strip().lower()
                dry = "yes" if val in ("yes", "dry", "true", "1") else "no"
    except Exception:
        pass
    return species, dry


@torch.no_grad()
def evaluate(checkpoint_path, data_dir, num_prompts=3):
    print(f"  Device: {DEVICE}")
    print(f"  Checkpoint: {checkpoint_path}")
    print()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    config["vocab_size"] = max(config["vocab_size"], max(checkpoint["model_state_dict"].get("decoder.lm_head.weight", torch.zeros(1)).shape[0], 346))

    tokenizer = FloraTokenizer(config.get("tokenizer_dir", "datasets/vlm_distill"))
    model = NanoFloraVLM(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")
    print()

    data_dir = Path(data_dir)
    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not class_dirs:
        print(f"  No class directories found in {data_dir}")
        print(f"  Expected: {data_dir}/acacia/, {data_dir}/aloe/, ...")
        return

    results = defaultdict(lambda: {"correct": 0, "total": 0, "species_correct": 0, "species_total": 0})
    confusion = defaultdict(lambda: defaultdict(int))
    total_time = 0

    for cls_dir in class_dirs:
        gt_species = cls_dir.name.lower()
        img_paths = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.jpeg")) + sorted(cls_dir.glob("*.png"))
        if not img_paths:
            continue
        print(f"  {gt_species}: {len(img_paths)} images")

        for img_idx, img_path in enumerate(img_paths):
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            all_preds = []
            selected_prompts = PROMPTS[:num_prompts]
            for prompt in selected_prompts:
                prompt_ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_special=False)
                t0 = time.time()
                result = model.generate(
                    img_tensor, prompt_ids, tokenizer,
                    max_new_tokens=20, temperature=0.1,
                )
                total_time += time.time() - t0
                all_preds.append(result)

            species_votes = defaultdict(int)
            dry_votes = defaultdict(int)
            for r in all_preds:
                sp, dr = parse_prediction(r)
                if sp:
                    species_votes[sp] += 1
                if dr:
                    dry_votes[dr] += 1
            pred_species = max(species_votes, key=species_votes.get) if species_votes else None
            pred_dry = max(dry_votes, key=dry_votes.get) if dry_votes else None

            results[gt_species]["total"] += 1
            results[gt_species]["species_total"] += 1
            if pred_species == gt_species:
                results[gt_species]["correct"] += 1
                results[gt_species]["species_correct"] += 1

            if pred_species is None:
                pred_species = "???"
            confusion[gt_species][pred_species] += 1

            if img_idx == 0:
                print(f"    {img_path.name}: pred=({pred_species}, {pred_dry})  (first sample)")

    total_imgs = sum(v["total"] for v in results.values())
    total_correct = sum(v["correct"] for v in results.values())
    total_species_correct = sum(v["species_correct"] for v in results.values())

    print(f"\n{'='*60}")
    print(f"  Results Summary")
    print(f"{'='*60}")
    print(f"  Total images:  {total_imgs}")
    print(f"  Avg inference: {total_time / max(total_imgs, 1) * 1000:.1f}ms per image")
    print(f"\n  Species Accuracy: {total_species_correct}/{total_imgs} ({100*total_species_correct/max(total_imgs,1):.1f}%)")
    print(f"\n  Per-Species Accuracy:")
    print(f"  {'Species':<20} {'Correct':>8} {'Total':>6} {'Acc':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*6}")
    species_accs = []
    for cls_name in sorted(results.keys()):
        r = results[cls_name]
        acc = 100 * r["correct"] / r["total"] if r["total"] else 0
        species_accs.append(acc)
        print(f"  {cls_name:<20} {r['correct']:>8} {r['total']:>6} {acc:>5.1f}%")
    print(f"  {'-'*20} {'-'*8} {'-'*6} {'-'*6}")
    mean_acc = sum(species_accs) / len(species_accs) if species_accs else 0
    print(f"  {'Mean':<20} {total_correct:>8} {total_imgs:>6} {mean_acc:>5.1f}%")

    print(f"\n  Confusion Matrix (rows=ground truth, cols=predicted):")
    all_predicted_classes = set()
    for row in confusion:
        all_predicted_classes.update(confusion[row].keys())
    classes = sorted(set(list(confusion.keys()) + list(all_predicted_classes)))
    cell_w = max(16, max(len(c) for c in classes) + 2)
    header = " " * cell_w + "".join(c.rjust(cell_w) for c in classes)
    print(f"\n  {header}")
    for row in classes:
        if row not in confusion:
            continue
        line = row.rjust(cell_w)
        for col in classes:
            val = confusion[row].get(col, 0)
            line += str(val).rjust(cell_w)
        print(f"  {line}")

    print(f"\n  Tips:")
    print(f"    Species accuracy is more reliable than dryness.")
    print(f"    Dryness accuracy requires ground-truth dryness labels.")
    print(f"    Use --num-prompts N to average over more prompts for better accuracy.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate NanoFloraVLM")
    parser.add_argument("--checkpoint", default="best_nano_vlm.pt",
                        help="Path to best_nano_vlm.pt")
    parser.add_argument("--data-dir", default="eval_images",
                        help="Directory with eval images")
    parser.add_argument("--num-prompts", type=int, default=3,
                        help="Number of prompts to average per image")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data_dir, args.num_prompts)
