import os
import json
import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import urllib.request
import time
import re

def prepare_dataset(base_dir="datasets"):
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, "annotations.csv")

    records = []

    print("Loading 'zenitsu09/indian-number-plate'...")
    try:
        ds = load_dataset("zenitsu09/indian-number-plate", split="train")
        print(f"Loaded dataset with {len(ds)} records.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print("Processing images and deduplicating...")
    
    total_records = len(ds)
    valid_images = 0
    with_plate_text = 0
    with_bboxes = 0
    skipped_records = 0
    
    seen_bases = set()

    for i, item in enumerate(tqdm(ds)):
        img = item.get("image")
        if img is None:
            skipped_records += 1
            continue
        valid_images += 1
            
        plate_text = item.get("plate_text")
        if not plate_text or str(plate_text).strip() == "":
            skipped_records += 1
            continue
        with_plate_text += 1
            
        # Deduplication based on orig_filename
        orig_filename = item.get("orig_filename", "")
        # Remove Roboflow augmentation hash e.g. .rf.0e78...
        base_name = re.sub(r'\.rf\.[a-f0-9]+', '', orig_filename)
        if base_name in seen_bases:
            # Skip duplicated augmentation
            skipped_records += 1
            continue
        seen_bases.add(base_name)

        # Get bbox
        try:
            xmin = float(item.get("xmin", 0))
            ymin = float(item.get("ymin", 0))
            xmax = float(item.get("xmax", 0))
            ymax = float(item.get("ymax", 0))
            if xmax > xmin and ymax > ymin:
                bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]
                with_bboxes += 1
            else:
                bbox = []
        except (ValueError, TypeError):
            bbox = []
            
        if not bbox:
            skipped_records += 1
            continue

        img_name = f"plate_{len(records):04d}.jpg"
        img_path = os.path.join(images_dir, img_name)
        
        # Save image
        try:
            if hasattr(img, "save"):
                img.convert("RGB").save(img_path)
            else:
                print(f"Unknown image format at index {i}")
                skipped_records += 1
                continue
        except Exception as e:
            skipped_records += 1
            continue
            
        records.append({
            "image_path": img_name,
            "plate_text": str(plate_text).strip(),
            "xmin": bbox[0],
            "ymin": bbox[1],
            "xmax": bbox[2],
            "ymax": bbox[3],
            "dataset_source": "zenitsu09/indian-number-plate"
        })
        
    print("\n--- DATASET PREPARATION SUMMARY ---")
    print(f"Total records in dataset: {total_records}")
    print(f"Records with valid image: {valid_images}")
    print(f"Records with plate_text: {with_plate_text}")
    print(f"Records with valid bboxes: {with_bboxes}")
    print(f"Invalid/skipped records (including augment/duplicates): {skipped_records}")
    print(f"Final usable deduplicated records: {len(records)}")

    if len(records) == 0:
        raise RuntimeError("FATAL ERROR: Zero annotations produced. Dataset extraction failed.")
        
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} annotations to {csv_path}")

if __name__ == "__main__":
    prepare_dataset("C:/Users/Krishi/.gemini/antigravity/scratch/anpr/datasets")
