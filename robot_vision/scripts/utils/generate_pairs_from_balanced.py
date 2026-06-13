#!/usr/bin/env python3
"""Generate teacher pairs from datasets/balanced/ folder structure."""

import json, random
from pathlib import Path

BALANCED = Path("datasets/balanced")
OUTPUT = Path("datasets/vlm_distill/pairs.jsonl")
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

random.seed(42)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pairs = []
for cls_dir in sorted(BALANCED.iterdir()):
    if not cls_dir.is_dir():
        continue
    parts = cls_dir.name.rsplit("_", 1)
    if len(parts) != 2 or parts[1] not in ("dry", "not_dry"):
        continue
    species, dryness = parts[0], parts[1]
    dry_str = "yes" if dryness == "dry" else "no"
    answer = f"species: {species}, dry: {dry_str}"
    for img_path in cls_dir.glob("*.jpg"):
        prompt = random.choice(PROMPTS)
        pairs.append({"image_path": str(img_path), "prompt": prompt, "answer": answer})

with open(OUTPUT, "w") as f:
    for p in pairs:
        f.write(json.dumps(p) + "\n")

print(f"  Generated {len(pairs)} pairs from {BALANCED}")
