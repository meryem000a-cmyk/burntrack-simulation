"""
Synthetic dataset generator CLI.

Usage:
    python scripts/generate_synthetic.py --n-samples 50000 --output data/processed/synthetic_dataset.csv
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fire spread dataset")
    parser.add_argument("--n-samples", type=int, default=50000, help="Number of synthetic samples")
    parser.add_argument("--output", type=str, default="data/processed/synthetic_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-level", type=float, default=0.12)
    parser.add_argument("--split", action="store_true", help="Also create train/val/test splits")
    args = parser.parse_args()

    try:
        from burntrack.data.synthetic import generate_synthetic_dataset
    except ImportError:
        print("ERROR: burntrack.data.synthetic module not found.")
        sys.exit(1)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    df = generate_synthetic_dataset(
        n_samples=args.n_samples,
        seed=args.seed,
        noise_level=args.noise_level,
    )

    df.to_csv(args.output, index=False)
    print(f"Generated {len(df):,} samples -> {args.output}")

    if args.split:
        from sklearn.model_selection import train_test_split

        base, ext = os.path.splitext(args.output)
        train, temp = train_test_split(df, test_size=0.3, random_state=args.seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=args.seed)

        train_path = f"{base}_train{ext}"
        val_path = f"{base}_val{ext}"
        test_path = f"{base}_test{ext}"

        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)

        print(f"Train: {len(train):,} -> {train_path}")
        print(f"Val:   {len(val):,} -> {val_path}")
        print(f"Test:  {len(test):,} -> {test_path}")


if __name__ == "__main__":
    main()
