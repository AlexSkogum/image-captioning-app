"""
Prepare Flickr8k dataset for training.
Converts the downloaded Flickr8k data into CSV format expected by the training pipeline.
"""

import os
import csv
from pathlib import Path


def prepare_flickr8k():
    """Convert Flickr8k dataset to CSV format."""
    
    # Paths
    flickr_root = Path('data/Flickr_Data/Flickr_Data')
    images_dir = flickr_root / 'Images'
    captions_dir = flickr_root / 'flickr8ktextfiles'
    
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    images_output_dir = output_dir / 'images'
    images_output_dir.mkdir(exist_ok=True)
    
    csv_output = output_dir / 'captions.csv'
    
    print(f"Looking for images in: {images_dir}")
    print(f"Looking for captions in: {captions_dir}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not captions_dir.exists():
        raise FileNotFoundError(f"Captions directory not found: {captions_dir}")
    
    # Find all caption files (e.g., flickr_8k_val_dataset.txt)
    caption_files = list(captions_dir.glob('flickr_8k_*.txt'))
    print(f"Found {len(caption_files)} caption files")
    
    if len(caption_files) == 0:
        raise ValueError("No caption files found in captions directory!")
    
    # Process captions
    rows = []
    image_count = 0
    caption_count = 0
    
    for cap_file in caption_files:
        print(f"\nProcessing {cap_file.name}...")
        
        # Read captions from file
        # Format: image_id\tcaption (tab-separated)
        with open(cap_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header
        if lines and lines[0].startswith('image_id'):
            lines = lines[1:]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            
            img_name = parts[0].strip()
            caption = parts[1].strip()
            
            # Remove <start> and <end> tags
            caption = caption.replace('<start>', '').replace('<end>', '').strip()
            
            # Check image exists
            img_path = images_dir / img_name
            if not img_path.exists():
                # print(f"Warning: Image not found for {img_name}")
                continue
            
            if not caption:
                continue
            
            # Store just the image name (DataLoader will combine with images_dir)
            rows.append({
                'image_path': img_name,
                'caption': caption.lower()  # Lowercase for consistency
            })
            caption_count += 1
            
            if caption_count % 1000 == 0:
                print(f"  Processed {caption_count} captions...")
        
        image_count += 1
    
    print(f"\n✅ Total: {caption_count} captions from multiple files")
    
    if not rows:
        raise ValueError("No data to write!")
    
    # Write CSV
    fieldnames = ['image_path', 'caption']
    with open(csv_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV written to: {csv_output}")
    print(f"Total rows: {len(rows)}")
    
    # Link images to data/images/ for easier access
    print(f"\nLinking images to {images_output_dir}...")
    linked_count = 0
    skipped = 0
    for img_file in images_dir.glob('*.jpg'):
        target = images_output_dir / img_file.name
        if target.exists():
            skipped += 1
            continue
        
        try:
            # Use mklink for Windows (hard link or junction)
            os.system(f'mklink /H "{target}" "{img_file}" >nul 2>&1')
            linked_count += 1
            if linked_count % 100 == 0:
                print(f"  Linked {linked_count} images...")
        except Exception as e:
            print(f"Warning: Could not link {img_file.name}: {e}")
    
    print(f"Linked {linked_count} images ({skipped} already existed)")
    print("\n✅ Dataset preparation complete!")
    print(f"Dataset CSV: {csv_output}")
    print(f"Images directory: {images_output_dir}")


if __name__ == '__main__':
    prepare_flickr8k()
