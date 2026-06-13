#!/usr/bin/env python3
"""Evaluate CNN dryness + YOLO species on 100 unused images per class."""

import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

import torch.nn as nn

IMG_SIZE = 224


class DrynessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x).flatten(1)
        return self.fc(x).squeeze(1)

random.seed(42)
torch.manual_seed(42)

BALANCED = Path("datasets/balanced")
CLS_DIR = Path("datasets/yolo_flora_cls/train")
N_PER_CLASS = 100
DEVICE = "cpu"


def get_eval_samples():
    samples = []
    for cls_dir in sorted(CLS_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        all_imgs = sorted(cls_dir.glob("*.jpg"))
        used = set()
        balanced_dir = BALANCED / cls_name
        if balanced_dir.exists():
            used = {p.name for p in balanced_dir.iterdir()}
        unused = [p for p in all_imgs if p.name not in used]
        take = min(len(unused), N_PER_CLASS)
        if take == 0:
            continue
        chosen = random.sample(unused, take)
        for p in chosen:
            samples.append((str(p), cls_name))
    return samples


def extract_species(cls_name):
    return cls_name.rsplit("_", 2)[0]


def extract_dryness(cls_name):
    return 1.0 if cls_name.endswith("_dry") else 0.0


def main():
    samples = get_eval_samples()
    print(f"Eval samples: {len(samples)}")
    for cls_name in sorted(set(s for _, s in samples)):
        cnt = sum(1 for _, s in samples if s == cls_name)
        print(f"  {cls_name}: {cnt}")

    cnn = DrynessCNN()
    cnn.load_state_dict(torch.load("models/dryness/best_dryness_cnn.pt", map_location="cpu"))
    cnn.eval()
    print(f"\nCNN params: {sum(p.numel() for p in cnn.parameters()):,}")

    yolo = YOLO("models/yolo/best(6).pt")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    yolo_correct = 0
    cnn_correct = 0
    both_correct = 0
    total = len(samples)

    for path, cls_name in tqdm(samples, desc="Evaluating"):
        species_true = extract_species(cls_name)
        dryness_true = extract_dryness(cls_name)
        CLASS_FIX = {
            "baobab": "adansonia",
            "senegalia": "acacia",
            "vachellia": "acacia",
        }
        species_true = CLASS_FIX.get(species_true, species_true)

        img = Image.open(path).convert("RGB")

        cnn_input = transform(img).unsqueeze(0)
        with torch.no_grad():
            logit = cnn(cnn_input)
            pred_dry = (torch.sigmoid(logit) > 0.5).float().item()
        if pred_dry == dryness_true:
            cnn_correct += 1

        yolo_results = yolo(img)
        pred_species = None
        if yolo_results and yolo_results[0].probs is not None:
            pred_species = yolo_results[0].names[yolo_results[0].probs.top1]
            pred_species = CLASS_FIX.get(pred_species, pred_species)
            if pred_species == species_true:
                yolo_correct += 1

        if pred_dry == dryness_true and pred_species == species_true:
            both_correct += 1

    print(f"\n{'='*50}")
    print(f"Total: {total} images")
    print(f"CNN dryness accuracy:       {cnn_correct}/{total} = {100*cnn_correct/total:.1f}%")
    print(f"YOLO species accuracy:      {yolo_correct}/{total} = {100*yolo_correct/total:.1f}%")
    print(f"Both correct (species+dryness): {both_correct}/{total} = {100*both_correct/total:.1f}%")


if __name__ == "__main__":
    main()
