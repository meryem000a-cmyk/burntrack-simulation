"""
Real fire dataset builder CLI.

Usage:
    python scripts/build_real_dataset.py --api-key YOUR_KEY --days 30
    python scripts/build_real_dataset.py --api-key YOUR_KEY --start-date 2026-05-01 --days 60
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

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
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days of FIRMS data to collect (default: 30)"
    )
    parser.add_argument(
        "--start-date", type=str, default=None,
        help="Start date YYYY-MM-DD (default: --days before today)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("NASA_FIRMS_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set NASA_FIRMS_API_KEY.")
        sys.exit(1)

    if args.start_date:
        date_start = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        date_start = datetime.now() - timedelta(days=args.days)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    try:
        from burntrack.data.real_dataset import build_real_dataset
    except ImportError:
        print("ERROR: burntrack.data.real_dataset module not found.")
        sys.exit(1)

    df = build_real_dataset(
        api_key=api_key,
        date_start=date_start,
        days_range=args.days,
    )
    df.to_csv(args.output, index=False)
    print(f"Built dataset with {len(df):,} samples -> {args.output}")


if __name__ == "__main__":
    main()
