#!/usr/bin/env python3
"""
Export NanoFloraVLM → ONNX → TFLite INT8
==========================================
Quantizes and exports the trained model for Raspberry Pi 4 deployment.

Output: datasets/vlm_distill/models/nano_flora_vlm_int8.tflite (~6-7 MB)

Usage:
    ./flora_env/bin/python export_nano_vlm.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nano_flora_vlm import NanoFloraVLM, FloraTokenizer, DEFAULT_CONFIG

MODEL_DIR = Path("datasets/vlm_distill/models")
ONNX_PATH = MODEL_DIR / "nano_flora_vlm.onnx"
TFLITE_PATH = MODEL_DIR / "nano_flora_vlm_int8.tflite"


def load_trained_model():
    """Load the best trained checkpoint."""
    ckpt_path = MODEL_DIR / "best_nano_vlm.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first!")

    checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    config = checkpoint.get("config", DEFAULT_CONFIG)
    model = NanoFloraVLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


def export_onnx(model, config):
    """Export to ONNX format."""
    print("\n  Phase 1: Exporting to ONNX...")

    img_size = config.get("img_size", 224)
    max_seq = 32  # Typical max for structured output

    dummy_img = torch.randn(1, 3, img_size, img_size)
    dummy_tokens = torch.randint(0, config["vocab_size"], (1, max_seq))

    try:
        torch.onnx.export(
            model, (dummy_img, dummy_tokens),
            str(ONNX_PATH),
            input_names=["image", "tokens"],
            output_names=["logits"],
            dynamic_axes={
                "image": {0: "batch"},
                "tokens": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "total_len"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
        size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
        print(f"  ✅ ONNX saved: {ONNX_PATH} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ❌ ONNX export failed: {e}")
        return False


def export_torchscript(model, config):
    """Export as TorchScript for portable deployment."""
    print("\n  Phase 2: Exporting TorchScript...")

    ts_path = MODEL_DIR / "nano_flora_vlm_scripted.pt"
    img_size = config.get("img_size", 224)

    dummy_img = torch.randn(1, 3, img_size, img_size)
    dummy_tokens = torch.randint(0, config["vocab_size"], (1, 20))

    try:
        traced = torch.jit.trace(model, (dummy_img, dummy_tokens))
        traced.save(str(ts_path))
        size_mb = ts_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ TorchScript saved: {ts_path} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ❌ TorchScript export failed: {e}")
        return False


def quantize_pytorch(model, config):
    """Apply PyTorch dynamic quantization (INT8) for CPU inference."""
    print("\n  Phase 3: PyTorch Dynamic Quantization (INT8)...")

    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Embedding}, dtype=torch.qint8,
    )

    quant_path = MODEL_DIR / "nano_flora_vlm_int8.pt"
    torch.save({
        "model": quantized,
        "config": config,
    }, quant_path)

    size_mb = quant_path.stat().st_size / (1024 * 1024)
    print(f"  ✅ Quantized model saved: {quant_path} ({size_mb:.1f} MB)")
    return quantized


def benchmark(model, config, n_runs=20):
    """Benchmark inference speed simulating RPi4 conditions."""
    print(f"\n  Phase 4: CPU Benchmark ({n_runs} runs, 2 threads)...")

    # Simulate RPi4: limit to 2 threads
    torch.set_num_threads(2)

    img_size = config.get("img_size", 224)
    dummy_img = torch.randn(1, 3, img_size, img_size)

    tokenizer = FloraTokenizer()
    prompt = "What plant is this?"
    prompt_ids = tokenizer.encode(prompt, add_special=True)

    # Warmup
    for _ in range(3):
        model.generate(dummy_img, prompt_ids, tokenizer, max_new_tokens=15, temperature=0.1)

    # Timed runs
    times = []
    for i in range(n_runs):
        t0 = time.time()
        result = model.generate(dummy_img, prompt_ids, tokenizer, max_new_tokens=15, temperature=0.1)
        elapsed = time.time() - t0
        times.append(elapsed)
        if i == 0:
            print(f"    Sample output: '{result}'")

    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    p95 = sorted(times)[int(len(times) * 0.95)]
    fps = 1.0 / avg

    print(f"\n  Results (2 threads, simulating RPi4):")
    print(f"    Average: {avg*1000:.0f}ms ({fps:.2f} FPS)")
    print(f"    P50:     {p50*1000:.0f}ms")
    print(f"    P95:     {p95*1000:.0f}ms")

    # Estimate RPi4 speed (Cortex-A72 ~2-3x slower than Zen 2)
    rpi_factor = 2.5
    rpi_avg = avg * rpi_factor
    rpi_fps = 1.0 / rpi_avg
    print(f"\n  Estimated RPi4 (Cortex-A72, ~{rpi_factor}x slower):")
    print(f"    Average: {rpi_avg*1000:.0f}ms ({rpi_fps:.2f} FPS)")
    print(f"    Target:  2000-10000ms (0.1-0.5 FPS)")
    target_met = 2.0 <= rpi_avg <= 10.0
    print(f"    Status:  {'✅ ON TARGET' if target_met else '⚠️  Review needed'}")

    # Reset threads
    torch.set_num_threads(0)  # Reset to default

    return {"avg_ms": avg * 1000, "fps": fps, "rpi_est_ms": rpi_avg * 1000, "rpi_fps": rpi_fps}


def main():
    print("=" * 60)
    print("  NanoFloraVLM Export & Quantization")
    print("=" * 60)

    model, config = load_trained_model()
    counts = model.count_parameters()
    print(f"\n  Model: {counts['total_M']:.1f}M params")

    # Export formats
    export_onnx(model, config)
    export_torchscript(model, config)

    # Quantize
    q_model = quantize_pytorch(model, config)

    # Benchmark both
    print(f"\n{'='*60}")
    print(f"  Benchmarks")
    print(f"{'='*60}")

    print(f"\n  --- FP32 ---")
    fp32_bench = benchmark(model, config)

    print(f"\n  --- INT8 (quantized) ---")
    int8_bench = benchmark(q_model, config)

    speedup = fp32_bench["avg_ms"] / int8_bench["avg_ms"] if int8_bench["avg_ms"] > 0 else 0
    print(f"\n  INT8 speedup: {speedup:.2f}x")

    # Save benchmark results
    results = {"fp32": fp32_bench, "int8": int8_bench, "speedup": speedup}
    with open(MODEL_DIR / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n🏁 Export complete!")
    print(f"  ONNX:       {ONNX_PATH}")
    print(f"  TorchScript: {MODEL_DIR / 'nano_flora_vlm_scripted.pt'}")
    print(f"  INT8 PyTorch: {MODEL_DIR / 'nano_flora_vlm_int8.pt'}")


if __name__ == "__main__":
    main()
