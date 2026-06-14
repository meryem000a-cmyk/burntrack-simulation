#!/usr/bin/env python3
"""
Dryness classifier using RGB-based vegetation indices.
Works on any RGB image — no NIR needed.

Usage:
    python3 dryness_classifier.py path/to/image.jpg
    python3 dryness_classifier.py path/to/directory/  (batch mode)
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image


def compute_indices(img_array):
    """Compute RGB-based vegetation/ drynes indices."""
    img = img_array.astype(np.float32) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    eps = 1e-8

    # Excess Green — high for green vegetation
    ExG = 2 * G - R - B

    # Green Leaf Index
    GLI = (2 * G - R - B) / (2 * G + R + B + eps)

    # Red/Green ratio — high for brown/dry, low for green
    R_over_G = R / (G + eps)

    # Green/Red ratio — high for green, low for dry/brown
    G_over_R = G / (R + eps)

    # Excess Red — high for soil/dry
    ExR = 1.4 * R - G

    # Excess Green minus Excess Red — standard vegetation index
    ExGR = ExG - ExR

    # HSV saturation — green plants have higher saturation in green channel
    hsv_s = 1 - 3 * np.min(img, axis=2) / (R + G + B + eps)

    # Proportion of "green" pixels (G significantly > R and G > B)
    green_mask = (G > R + 0.05) & (G > B + 0.05)
    green_fraction = green_mask.mean()

    # Proportion of "brown" pixels (R > G and low saturation)
    brown_mask = (R > G + 0.05) & (hsv_s < 0.3)
    brown_fraction = brown_mask.mean()

    return {
        "mean_ExG": float(ExG.mean()),
        "mean_GLI": float(GLI.mean()),
        "mean_R_over_G": float(R_over_G.mean()),
        "mean_G_over_R": float(G_over_R.mean()),
        "mean_ExGR": float(ExGR.mean()),
        "mean_saturation": float(hsv_s.mean()),
        "green_fraction": float(green_fraction),
        "brown_fraction": float(brown_fraction),
    }


def classify_dryness(indices):
    """Classify as dry or not-dry based on computed indices.

    Heuristic rules tuned for African savanna plants:
    - High R/G ratio + low ExG + low green fraction → dry
    - Low R/G ratio + high ExG + high green fraction → alive
    """
    score = 0.0

    # Excess Green: negative → dry, positive → alive
    if indices["mean_ExG"] < -0.05:
        score += 1
    elif indices["mean_ExG"] > 0.05:
        score -= 1

    # R/G ratio: > 1.0 → more red (dry), < 0.9 → more green (alive)
    if indices["mean_R_over_G"] > 1.05:
        score += 1
    elif indices["mean_R_over_G"] < 0.90:
        score -= 1

    # Green fraction: low → dry, high → alive
    if indices["green_fraction"] < 0.05:
        score += 1.5
    elif indices["green_fraction"] > 0.20:
        score -= 1.5

    # Brown fraction: high → dry
    if indices["brown_fraction"] > 0.15:
        score += 1
    elif indices["brown_fraction"] < 0.05:
        score -= 0.5

    # ExGR: negative → dry/bare, positive → vegetation
    if indices["mean_ExGR"] < -0.05:
        score += 0.5
    elif indices["mean_ExGR"] > 0.05:
        score -= 0.5

    # Saturation: low → dry/wilted, high → healthy green
    if indices["mean_saturation"] < 0.25:
        score += 0.5
    elif indices["mean_saturation"] > 0.35:
        score -= 0.5

    is_dry = score >= 0
    confidence = min(abs(score) / 4, 1.0)

    return is_dry, confidence, score


def generate_heatmap(img_array, indices):
    """Create a visualization highlighting green vs brown regions."""
    img = img_array.astype(np.float32) / 255.0
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    vis = np.zeros((*img.shape[:2], 3), dtype=np.uint8)

    green_mask = (G > R + 0.05) & (G > B + 0.05)
    brown_mask = (R > G + 0.05) & (1 - 3 * np.min(img, axis=2) / (R + G + B + 1e-8) < 0.3)

    vis[green_mask] = [0, 200, 0]
    vis[brown_mask] = [150, 80, 0]
    return vis


def analyze_image(image_path):
    """Analyze a single image and print dryness classification."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    indices = compute_indices(img_array)
    is_dry, confidence, score = classify_dryness(indices)

    label = "DRY  " if is_dry else "ALIVE"
    print(f"  {image_path.name:40s} → {label}  (score={score:+.2f}, confidence={confidence:.0%})")

    heatmap = generate_heatmap(img_array, indices)
    heat_path = image_path.parent / f"{image_path.stem}_drymap.jpg"
    Image.fromarray(heatmap).save(heat_path)

    return is_dry, indices


def main():
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print("Usage: python3 dryness_classifier.py <image_path> [image_path2 ...]")
        return

    print(f"{'Image':40s} {'Result':>12} {'Score':>8} {'Conf':>6}")
    print("-" * 68)

    files = []
    for p in paths:
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("*.jpg")) + sorted(p.glob("*.jpeg")) + sorted(p.glob("*.png")))

    dry_count = 0
    for f in files:
        try:
            is_dry, _ = analyze_image(f)
            if is_dry:
                dry_count += 1
        except Exception as e:
            print(f"  {f.name:40s} → ERROR: {e}")

    if len(files) > 1:
        print("-" * 68)
        alive = len(files) - dry_count
        print(f"  Total: {len(files)} images — {dry_count} dry ({100*dry_count/len(files):.0f}%), {alive} alive ({100*alive/len(files):.0f}%)")


if __name__ == "__main__":
    main()
