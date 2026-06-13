#!/usr/bin/env python3
"""Evaluate trained dryness CNN on eval_images, comparing against Moondream labels."""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

LABELS_CSV = Path("eval_images/dryness_labels.csv")


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


def load_labels():
    labels = {}
    if LABELS_CSV.exists():
        with open(LABELS_CSV) as f:
            for row in csv.DictReader(f):
                labels[row["path"]] = row["label"]
    return labels


def main():
    labels = load_labels()
    if not labels:
        print("No labels found. Run label_dryness_moondream.py first.")
        return

    model = DrynessCNN()
    model.load_state_dict(torch.load("best_dryness_cnn.pt", map_location="cpu"))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"{'Species':20s} {'Correct':>8} {'Total':>6} {'Acc':>6}")
    print("-" * 42)

    total_correct = total_all = 0
    for d in sorted(Path("eval_images").iterdir()):
        if not d.is_dir():
            continue
        correct = total = 0
        for f in d.glob("*.jpg"):
            gt = labels.get(str(f))
            if gt is None:
                continue
            img = transform(Image.open(f).convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                logit = model(img).item()
            pred = "dry" if torch.sigmoid(torch.tensor(logit)) > 0.5 else "alive"
            total += 1
            if pred == gt:
                correct += 1
        total_correct += correct
        total_all += total
        acc = 100 * correct / total if total else 0
        print(f"{d.name:20s} {correct:>8} {total:>6} {acc:>5.0f}%")

    print("-" * 42)
    print(f"{'Total':20s} {total_correct:>8} {total_all:>6} {100*total_correct/total_all:>5.0f}%")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
