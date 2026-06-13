#!/usr/bin/env python3
"""
Per-pixel green vs brown energy — any RGB image, no training needed.
"""

import sys
import random
from pathlib import Path

import numpy as np
from PIL import Image


def center_weight(h, w, sigma=0.4):
    """Gaussian weight map: center pixels matter more."""
    ys, xs = np.mgrid[:h, :w]
    cy, cx = h / 2, w / 2
    d = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    d_max = np.sqrt(cy ** 2 + cx ** 2)
    return np.exp(-0.5 * (d / (sigma * d_max)) ** 2)


def classify(img_array):
    """Per-pixel green vs brown energy comparison."""
    img = img_array.astype(np.float32) / 255.0
    h, w = img.shape[:2]
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Green energy per pixel: ExG = 2G - R - B
    green_energy = 2 * G - R - B
    green_energy = np.maximum(green_energy, 0)  # only positive green signal

    # Brown energy per pixel: R dominates over G and B
    brown_energy = np.maximum(R - G, 0) + np.maximum(R - B, 0)
    brown_energy = brown_energy / 2  # normalize to similar scale

    # Edge-weighted center mask (plant is usually centered)
    w_map = center_weight(h, w)
    green_total = (green_energy * w_map).sum()
    brown_total = (brown_energy * w_map).sum() + 1e-8

    ratio = green_total / brown_total
    is_dry = ratio < 0.6
    conf = min(abs(ratio - 0.6) / 0.6, 1.0)

    return is_dry, float(ratio), green_energy, brown_energy, w_map


def main():
    import os
    paths = []
    for a in sys.argv[1:]:
        p = Path(a)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend(sorted(p.glob("*.jpg")) + sorted(p.glob("*.jpeg")) + sorted(p.glob("*.png")))

    if not paths:
        # Test mode: run on eval_images
        paths = sorted(Path("eval_images").rglob("*.jpg"))[:50]

    print(f"{'Image':50s} {'Result':>8} {'Green/Brown':>12}")
    print("-" * 72)
    for f in paths:
        img = np.array(Image.open(f).convert("RGB"))
        is_dry, ratio, ge, be, wm = classify(img)
        label = "DRY  " if is_dry else "ALIVE"
        print(f"{str(f):50s} {label:>8} {ratio:>11.3f}")

    # Batch mode: per-species breakdown
    if len(paths) > 50:
        pass


if __name__ == "__main__":
    main()
