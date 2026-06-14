#!/usr/bin/env python3
"""
NanoFloraVLM — Micro Vision-Language Model (~25M params)
=========================================================
Domain-specific VLM for African flora identification + dryness detection.
Distilled from Moondream/Gemini, designed for Raspberry Pi 4 deployment.

Architecture:
    MobileNetV3-Small (2.5M) → Projector (0.2M) → TinyGPT Decoder (22M)

Usage:
    from nano_flora_vlm import NanoFloraVLM, FloraTokenizer
    model = NanoFloraVLM()
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ── Config ──
DEFAULT_CONFIG = {
    "img_size": 224,
    "vision_dim": 576,
    "d_model": 384,
    "n_heads": 6,
    "n_layers": 4,
    "ff_dim": 1024,
    "vocab_size": 512,
    "max_seq_len": 64,
    "dropout": 0.1,
    "num_visual_tokens": 4,
}


# ── Flora Tokenizer Wrapper ──

class FloraTokenizer:
    """Wraps the HuggingFace BPE tokenizer for NanoFloraVLM."""

    def __init__(self, tokenizer_dir: str = "datasets/vlm_distill"):
        self.tokenizer_dir = Path(tokenizer_dir)
        meta_path = self.tokenizer_dir / "tokenizer_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Run build_flora_tokenizer.py first. Missing: {meta_path}")

        with open(meta_path) as f:
            self.meta = json.load(f)
        self.type = self.meta["type"]
        self.vocab_size = self.meta["vocab_size"]

        if self.type == "bpe":
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_dir / self.meta["tokenizer_file"]))
            self.pad_id = self.tokenizer.token_to_id("<pad>")
            self.bos_id = self.tokenizer.token_to_id("<bos>")
            self.eos_id = self.tokenizer.token_to_id("<eos>")
            self.img_id = self.tokenizer.token_to_id("<img>")
            self.unk_id = self.tokenizer.token_to_id("<unk>")
        else:
            with open(self.tokenizer_dir / self.meta["vocab_file"]) as f:
                self.vocab = json.load(f)
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
            self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
            self.img_id, self.unk_id = 3, 4

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        if self.type == "bpe":
            encoded = self.tokenizer.encode(text)
            ids = encoded.ids
            if not add_special:
                if ids and ids[0] == self.bos_id: ids = ids[1:]
                if ids and ids[-1] == self.eos_id: ids = ids[:-1]
            return ids
        else:
            ids = [self.bos_id] if add_special else []
            for word in text.split():
                clean = word.strip(".,!?:;")
                ids.append(self.vocab.get(clean, self.unk_id))
            if add_special: ids.append(self.eos_id)
            return ids

    def decode(self, ids: list[int]) -> str:
        if self.type == "bpe":
            return self.tokenizer.decode(ids)
        skip = {self.pad_id, self.bos_id, self.eos_id, self.img_id}
        return " ".join(self.inv_vocab.get(t, "") for t in ids if t not in skip)

    def pad_sequence(self, ids: list[int], max_len: int) -> list[int]:
        return ids[:max_len] if len(ids) >= max_len else ids + [self.pad_id] * (max_len - len(ids))


# ── TinyGPT Decoder ──

class TinyGPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attn_mask,
                                 key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPTDecoder(nn.Module):
    """Tiny GPT-style autoregressive decoder with visual token prefix."""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, ff_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TinyGPTBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight  # Weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, visual_tokens, text_tokens, padding_mask=None):
        B, n_vis = visual_tokens.shape[:2]
        seq_len = text_tokens.shape[1]
        total_len = n_vis + seq_len
        text_embeds = self.token_embed(text_tokens)
        combined = torch.cat([visual_tokens, text_embeds], dim=1)
        positions = torch.arange(total_len, device=combined.device)
        combined = self.embed_drop(combined + self.pos_embed(positions))
        causal_mask = torch.triu(torch.ones(total_len, total_len, device=combined.device, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            combined = block(combined, attn_mask=causal_mask, key_padding_mask=padding_mask)
        return self.lm_head(self.ln_f(combined))


# ── Full Model ──

class NanoFloraVLM(nn.Module):
    """~25M param micro-VLM for flora identification on RPi4."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.vision_encoder = nn.Sequential(backbone.features, backbone.avgpool, nn.Flatten())
        n_vis = self.config["num_visual_tokens"]
        self.projector = nn.Sequential(
            nn.Linear(self.config["vision_dim"], self.config["d_model"] * n_vis),
            nn.LayerNorm(self.config["d_model"] * n_vis),
            nn.GELU(),
        )
        self.n_visual_tokens = n_vis
        self.decoder = TinyGPTDecoder(
            self.config["vocab_size"], self.config["d_model"], self.config["n_heads"],
            self.config["n_layers"], self.config["ff_dim"], self.config["max_seq_len"],
            self.config["dropout"],
        )

    def encode_image(self, image):
        features = self.vision_encoder(image)
        projected = self.projector(features)
        return projected.view(-1, self.n_visual_tokens, self.config["d_model"])

    def forward(self, image, text_tokens, padding_mask=None):
        visual_tokens = self.encode_image(image)
        return self.decoder(visual_tokens, text_tokens, padding_mask)

    @torch.no_grad()
    def generate(self, image, prompt_ids, tokenizer, max_new_tokens=32, temperature=0.7):
        self.eval()
        device = next(self.parameters()).device
        visual_tokens = self.encode_image(image.to(device))
        generated = list(prompt_ids)
        for _ in range(max_new_tokens):
            text_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = self.decoder(visual_tokens, text_tensor)
            next_logits = logits[0, -1, :] / max(temperature, 1e-8)
            next_token = next_logits.argmax().item() if temperature < 0.1 else torch.multinomial(F.softmax(next_logits, dim=-1), 1).item()
            if next_token == tokenizer.eos_id: break
            generated.append(next_token)
        return tokenizer.decode(generated[len(prompt_ids):])

    def count_parameters(self):
        v = sum(p.numel() for p in self.vision_encoder.parameters())
        p = sum(p.numel() for p in self.projector.parameters())
        d = sum(p.numel() for p in self.decoder.parameters())
        t = v + p + d
        return {"vision_encoder": v, "projector": p, "decoder": d, "total": t, "total_M": t / 1e6}


def _test():
    print("=" * 60)
    print("  NanoFloraVLM — Architecture Test")
    print("=" * 60)
    model = NanoFloraVLM()
    counts = model.count_parameters()
    for k, v in counts.items():
        print(f"  {k:.<30} {v:.1f}M" if k == "total_M" else f"  {k:.<30} {v:,}")
    B = 2
    img = torch.randn(B, 3, 224, 224)
    tok = torch.randint(0, 512, (B, 20))
    out = model(img, tok)
    expect = model.n_visual_tokens + 20
    print(f"\n  Input: img={img.shape}, tok={tok.shape}")
    print(f"  Output: {out.shape} (expected [{B}, {expect}, 512])")
    assert out.shape == (B, expect, 512)
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    print(f"  Size: {mb:.1f}MB FP32, ~{mb/4:.1f}MB INT8")
    print("  ✅ Test passed!")


if __name__ == "__main__":
    _test()
