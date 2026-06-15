"""
Real fire dataset builder CLI.

Usage:
    python scripts/build_real_dataset.py --api-key YOUR_KEY --output data/processed/real_african_dataset.csv
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Build a real fire dataset from NASA FIRMS")
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="NASA FIRMS API key (or set NASA_FIRMS_API_KEY env var)"
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/real_african_dataset.csv",
        help="Output CSV path"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NASA_FIRMS_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set NASA_FIRMS_API_KEY.")
        sys.exit(1)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    try:
        from burntrack.data.real_dataset import build_real_dataset
    except ImportError:
        print("ERROR: burntrack.data.real_dataset module not found.")
        sys.exit(1)

    df = build_real_dataset(api_key=api_key)
    df.to_csv(args.output, index=False)
    print(f"Built dataset with {len(df):,} samples -> {args.output}")


if __name__ == "__main__":
    main()
