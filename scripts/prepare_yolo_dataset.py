import os
import shutil
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml

def prepare_yolo_dataset(base_dir="datasets", output_dir="datasets/yolo", seed=42):
    csv_path = os.path.join(base_dir, "annotations.csv")
    images_dir = os.path.join(base_dir, "images")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} annotations.")
    
    # Shuffle and split 70/20/10
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_total = len(df)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.2)
    # the rest is test
    
    splits = {
        "train": df.iloc[:n_train],
        "val": df.iloc[n_train:n_train+n_val],
        "test": df.iloc[n_train+n_val:]
    }
    
    # Create directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)
        
    for split, split_df in splits.items():
        print(f"Preparing {split} set ({len(split_df)} images)...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            img_name = row['image_path']
            src_img_path = os.path.join(images_dir, img_name)
            
            if not os.path.exists(src_img_path):
                continue
                
            # Get dimensions for normalization
            try:
                with Image.open(src_img_path) as im:
                    w, h = im.size
            except Exception:
                continue
                
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            
            # YOLO format: class x_center y_center width height (normalized)
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h
            
            # Clamp to [0, 1] just in case
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            box_w = max(0.0, min(1.0, box_w))
            box_h = max(0.0, min(1.0, box_h))
            
            # Copy image
            dst_img_path = os.path.join(output_dir, "images", split, img_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Write label
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(output_dir, "labels", split, label_name)
            with open(label_path, "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                
    # Create data.yaml
    data_yaml = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "license_plate"
        }
    }
    
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    print(f"YOLO dataset prepared at {output_dir}")
    print(f"data.yaml path: {yaml_path}")

if __name__ == "__main__":
    prepare_yolo_dataset()
