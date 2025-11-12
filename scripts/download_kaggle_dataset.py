#!/usr/bin/env python
"""Simple helper to download a Kaggle dataset via the kaggle CLI.

This script shells out to the `kaggle` command. It requires the user to
have placed their `kaggle.json` token in the appropriate location (see README).

Example:
  python scripts/download_kaggle_dataset.py --dataset owner/dataset-name --out data/
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Kaggle dataset using the kaggle CLI")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset identifier, e.g. owner/dataset-name or competitions -c name")
    parser.add_argument("--out", default="data/", help="Output directory to place dataset")
    parser.add_argument("--unzip", action="store_true", help="Unzip the downloaded file(s)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer datasets download command, but allow the competitions form if user passed -c style
    # If they passed a string starting with "-c" or contains "competitions", we try competitions download
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download", "-d", args.dataset, "-p", str(out_dir)]
    # If user explicitly wants unzip, add flag
    if args.unzip:
        cmd.extend(["--unzip"])

    print("Running:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        print("Failed to download with `kaggle datasets download`. Trying competitions download...", file=sys.stderr)
        # Try competitions style
        cmd2 = [sys.executable, "-m", "kaggle", "competitions", "download", "-c", args.dataset, "-p", str(out_dir)]
        if args.unzip:
            cmd2.extend(["--unzip"])
        try:
            subprocess.check_call(cmd2)
        except subprocess.CalledProcessError as e:
            print("Both dataset and competition downloads failed. Ensure the dataset identifier is correct and your kaggle credentials are set.", file=sys.stderr)
            return 2

    print("Download finished. See files in:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
