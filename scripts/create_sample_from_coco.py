#!/usr/bin/env python
"""Create a small sample dataset from COCO annotations.

This script copies a random subset of images and writes a CSV with image_path,caption
so it can be used with the project's preprocessing and training scripts for quick smoke tests.

Example:
  python scripts/create_sample_from_coco.py --annotations data/annotations/captions_train2017.json --images-dir data/train2017 --sample-size 100 --out-dir data/sample
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List


def load_coco_captions(ann_file: Path) -> Dict[int, List[str]]:
    with ann_file.open("r", encoding="utf8") as f:
        j = json.load(f)
    captions_by_img: Dict[int, List[str]] = {}
    for ann in j.get("annotations", []):
        img_id = ann["image_id"]
        captions_by_img.setdefault(img_id, []).append(ann["caption"])
    images = {img["id"]: img for img in j.get("images", [])}
    return captions_by_img, images


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a small sample from COCO-style annotations")
    parser.add_argument("--annotations", required=True, help="Path to COCO captions JSON")
    parser.add_argument("--images-dir", required=True, help="Directory containing COCO images")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of images to sample")
    parser.add_argument("--out-dir", default="data/sample", help="Output directory for sampled images and CSV")
    args = parser.parse_args()

    ann_file = Path(args.annotations)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_images = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    captions_by_img, images = load_coco_captions(ann_file)

    available_ids = [img_id for img_id in images.keys() if img_id in captions_by_img]
    if not available_ids:
        print("No images with captions found in annotations.")
        return 1

    sample_ids = random.sample(available_ids, min(args.sample_size, len(available_ids)))

    csv_path = out_dir / "sample_captions.csv"
    with csv_path.open("w", newline="", encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "caption"])
        for img_id in sample_ids:
            img_info = images[img_id]
            file_name = img_info["file_name"]
            src = images_dir / file_name
            if not src.exists():
                print(f"Warning: image not found: {src}")
                continue
            dst = out_images / file_name
            shutil.copy2(src, dst)
            # pick a single caption for the CSV (the first)
            caps = captions_by_img.get(img_id, [])
            caption = caps[0] if caps else ""
            writer.writerow([str(dst), caption])

    print(f"Sample created: images -> {out_images}, csv -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
