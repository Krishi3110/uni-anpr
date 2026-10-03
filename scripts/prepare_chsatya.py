import os
import shutil
import pandas as pd
import numpy as np
import cv2
import yaml

def convert_to_yolo():
    base_dir = "datasets/chsatya"
    out_dir = "datasets/chsatya_yolo"
    
    # Read the dataset summary
    df = pd.read_csv(os.path.join(base_dir, "boundaries.csv"))
    
    # Filter out corrupted or missing values
    valid_rows = []
    for idx, row in df.iterrows():
        img_name = str(row['Img_name']).replace('.xml', '.jpg')
        img_path = os.path.join(base_dir, "Images", img_name)
        text = str(row['license_number']).strip()
        
        if os.path.exists(img_path) and pd.notna(text) and text != "":
            row['img_path'] = img_name
            valid_rows.append(row)
            
    df_valid = pd.DataFrame(valid_rows)
    print(f"Total valid images: {len(df_valid)}")
    
    df_valid = df_valid.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df_valid)
    train_end = int(n * 0.7)
    val_end = int(n * 0.9)
    
    train_df = df_valid.iloc[:train_end]
    val_df = df_valid.iloc[train_end:val_end]
    test_df = df_valid.iloc[val_end:]
    
    print(f"Split counts -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    splits = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    gt_records = []
    
    for split_name, split_df in splits.items():
        img_out_dir = os.path.join(out_dir, 'images', split_name)
        lbl_out_dir = os.path.join(out_dir, 'labels', split_name)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        for idx, row in split_df.iterrows():
            img_name = row['img_path']
            src_img = os.path.join(base_dir, "Images", img_name)
            dst_img = os.path.join(img_out_dir, img_name)
            
            # Read image to verify actual dimensions
            im = cv2.imread(src_img)
            if im is None:
                continue
            h, w = im.shape[:2]
            
            # Copy image
            shutil.copy(src_img, dst_img)
            
            # YOLO conversion
            xmin = float(row['xmin'])
            xmax = float(row['xmax'])
            ymin = float(row['ymin'])
            ymax = float(row['ymax'])
            
            # Clamp to image boundaries just in case
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)
            
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h
            
            lbl_name = img_name.replace('.jpg', '.txt')
            with open(os.path.join(lbl_out_dir, lbl_name), 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                
            gt_records.append({
                'split': split_name,
                'image_path': img_name,
                'plate_text': row['license_number'],
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })
            
    # Save ground truth for E2E
    pd.DataFrame(gt_records).to_csv(os.path.join(out_dir, "e2e_ground_truth.csv"), index=False)
    
    # Create data.yaml using absolute paths to avoid YOLO path issues
    abs_out_dir = os.path.abspath(out_dir).replace('\\', '/')
    yaml_data = {
        'path': abs_out_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 1,
        'names': ['license_plate']
    }
    with open(os.path.join(out_dir, "data.yaml"), 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
        
if __name__ == "__main__":
    convert_to_yolo()
