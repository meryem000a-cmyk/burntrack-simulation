#!/usr/bin/env python3
"""
Build Flora Tokenizer — Tiny BPE for NanoFloraVLM
===================================================
Creates a ~512-token BPE tokenizer covering only the flora domain.
This extreme vocabulary reduction (vs 32K+ in general VLMs) is what
makes the 22M-param decoder feasible for structured text generation.

Usage:
    ./flora_env/bin/python build_flora_tokenizer.py
"""

import json
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("datasets/vlm_distill")
VOCAB_SIZE = 512

# The 17 species in our dataset
SPECIES = [
    "adansonia", "acacia", "vachellia", "senegalia", "combretum",
    "brachystegia", "colophospermum", "ficus", "khaya", "macaranga",
    "euphorbia", "aloe", "protea", "erica", "themeda", "andropogon",
    "tamarix",
]

# Special tokens
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<img>", "<unk>"]

# ──────────────────────────────────────────────────────────────────────
# Training corpus — all possible outputs the model could generate
# ──────────────────────────────────────────────────────────────────────

def build_training_corpus() -> list[str]:
    """Generate all structured outputs plus natural variants for BPE training."""
    
    corpus = []
    
    conditions = ["yes", "no"]
    dryness_words = ["dry", "alive", "green", "brown", "cured", "healthy", "dead", "wilted"]
    
    # Primary structured format (what the model will actually output)
    for sp in SPECIES:
        for dry in conditions:
            corpus.append(f"species: {sp}, dry: {dry}")
            corpus.append(f"species: {sp}, condition: {'dry' if dry == 'yes' else 'alive'}")
    
    # Question templates (input side)
    questions = [
        "What species of plant is this?",
        "What plant is this?",
        "Identify this plant.",
        "What type of plant do you see?",
        "Is this plant dry or alive?",
        "What is the condition of this plant?",
        "Describe this plant.",
        "Identify this plant and its condition.",
        "What species is this and is it dry?",
        "Name this plant species.",
    ]
    corpus.extend(questions)
    
    # Natural language variants (for robustness)
    for sp in SPECIES:
        corpus.append(f"This is a {sp} plant.")
        corpus.append(f"This is a dry {sp}.")
        corpus.append(f"This is a healthy {sp}.")
        corpus.append(f"I see a {sp} tree.")
        corpus.append(f"The plant is {sp}.")
        corpus.append(f"This appears to be {sp}.")
        for dw in dryness_words:
            corpus.append(f"The {sp} looks {dw}.")
    
    # Add common botanical words
    botanical = [
        "tree", "plant", "shrub", "bush", "grass", "succulent", "flower",
        "leaf", "leaves", "bark", "branch", "stem", "root", "canopy",
        "foliage", "vegetation", "flora", "botanical", "african",
        "savanna", "woodland", "fynbos", "miombo", "bushveld",
        "baobab", "mahogany", "mopane", "heather",
        "species", "condition", "type", "identify", "describe",
        "this", "is", "a", "the", "of", "and", "or", "not",
        "what", "how", "dry", "alive", "yes", "no",
    ]
    for word in botanical:
        corpus.append(word)
    
    # Repeat key patterns to weight them in BPE
    for _ in range(10):
        for sp in SPECIES:
            corpus.append(f"species: {sp}, dry: yes")
            corpus.append(f"species: {sp}, dry: no")
    
    return corpus


def build_tokenizer_from_corpus(corpus: list[str]):
    """Build a BPE tokenizer using the tokenizers library."""
    
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    except ImportError:
        print("❌ 'tokenizers' library not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "tokenizers"])
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Train on our corpus
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
        show_progress=True,
    )
    
    # Write corpus to temp file for training
    corpus_path = OUTPUT_DIR / "tokenizer_corpus.txt"
    with open(corpus_path, "w") as f:
        for line in corpus:
            f.write(line + "\n")
    
    tokenizer.train([str(corpus_path)], trainer)
    
    # Add post-processing for BOS/EOS
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )
    
    return tokenizer


def build_simple_tokenizer(corpus: list[str]) -> dict:
    """
    Fallback: Build a simple character/word-level tokenizer if 
    the 'tokenizers' library is unavailable.
    """
    
    # Collect all unique characters and common subwords
    all_text = " ".join(corpus)
    chars = sorted(set(all_text))
    
    # Start with special tokens
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    idx = len(SPECIAL_TOKENS)
    
    # Add individual characters
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1
    
    # Add common words as full tokens (greedy)
    word_freq = {}
    for line in corpus:
        for word in line.split():
            word = word.strip(".,!?")
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency, add top words until vocab is full
    sorted_words = sorted(word_freq.items(), key=lambda x: -x[1])
    for word, freq in sorted_words:
        if idx >= VOCAB_SIZE:
            break
        if word not in vocab and len(word) > 1:
            vocab[word] = idx
            idx += 1
    
    return vocab


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  Building Flora Tokenizer")
    print("=" * 60)
    
    corpus = build_training_corpus()
    print(f"\n  Training corpus: {len(corpus)} lines")
    print(f"  Target vocab size: {VOCAB_SIZE}")
    
    # Try HuggingFace tokenizers first, fall back to simple
    try:
        tokenizer = build_tokenizer_from_corpus(corpus)
        
        # Save tokenizer
        tokenizer_path = OUTPUT_DIR / "flora_tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        
        actual_vocab = tokenizer.get_vocab_size()
        print(f"\n  ✅ BPE tokenizer built!")
        print(f"  Vocab size: {actual_vocab}")
        print(f"  Saved to: {tokenizer_path}")
        
        # Test encoding
        print(f"\n  Test encodings:")
        test_sentences = [
            "species: acacia, dry: yes",
            "species: colophospermum, dry: no",
            "What plant is this?",
            "This is a dry baobab tree.",
        ]
        for sent in test_sentences:
            encoded = tokenizer.encode(sent)
            decoded = tokenizer.decode(encoded.ids)
            print(f"    '{sent}'")
            print(f"    → tokens: {encoded.ids[:20]}{'...' if len(encoded.ids) > 20 else ''} ({len(encoded.ids)} tokens)")
            print(f"    → decoded: '{decoded}'")
            print()
        
        # Save vocab mapping for reference
        vocab = tokenizer.get_vocab()
        vocab_path = OUTPUT_DIR / "flora_vocab.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        print(f"  Vocab mapping saved to: {vocab_path}")
        
        # Save metadata
        meta = {
            "type": "bpe",
            "vocab_size": actual_vocab,
            "special_tokens": SPECIAL_TOKENS,
            "species": SPECIES,
            "tokenizer_file": "flora_tokenizer.json",
        }
        
    except Exception as e:
        print(f"\n  ⚠️  BPE tokenizer failed ({e}), building simple tokenizer...")
        vocab = build_simple_tokenizer(corpus)
        
        vocab_path = OUTPUT_DIR / "flora_vocab.json"
        with open(vocab_path, "w") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
        
        print(f"\n  ✅ Simple tokenizer built!")
        print(f"  Vocab size: {len(vocab)}")
        print(f"  Saved to: {vocab_path}")
        
        meta = {
            "type": "simple",
            "vocab_size": len(vocab),
            "special_tokens": SPECIAL_TOKENS,
            "species": SPECIES,
            "vocab_file": "flora_vocab.json",
        }
    
    meta_path = OUTPUT_DIR / "tokenizer_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to: {meta_path}")
    
    print(f"\n🏁 Tokenizer ready!")


if __name__ == "__main__":
    main()
