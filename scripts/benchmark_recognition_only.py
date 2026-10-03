import os
import sys
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paddleocr import TextRecognition
from src.normalizer import TextNormalizer

def evaluate_recognition_only(base_dir, out_dir):
    print("Loading models...")
    init_start = time.time()
    
    det_model = YOLO('runs/detect/models/plate_detector/weights/best.pt')
    
    rec_model = TextRecognition(device='cpu', engine='paddle_static')
    norm = TextNormalizer()
    
    init_time = time.time() - init_start
    print(f"Models initialized in {init_time:.2f}s")
    
    gt_df = pd.read_csv(os.path.join(base_dir, 'e2e_ground_truth.csv'))
    test_df = gt_df[gt_df['split'] == 'test'].copy()
    images_dir = os.path.join(base_dir, 'images', 'test')
    
    print(f"Running Recognition-Only E2E Benchmark on {len(test_df)} test images.")
    
    # Warmup
    if len(test_df) > 0:
        first_img = cv2.imread(os.path.join(images_dir, test_df.iloc[0]['image_path']))
        if first_img is not None:
            det_model.predict(first_img, verbose=False)
            h, w = first_img.shape[:2]
            crop = first_img[max(0, h//2-50):min(h, h//2+50), max(0, w//2-100):min(w, w//2+100)]
            _ = list(rec_model.predict(input=crop, batch_size=1))
            
    latencies = {
        'total': [],
        'yolo': [],
        'ocr': []
    }
    
    detector_hits = 0
    detector_misses = 0
    e2e_exact_matches = 0
    total_chars = 0
    matched_chars = 0
    no_reads = 0
    wrong_reads = 0
    
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_name = row['image_path']
        img_path = os.path.join(images_dir, img_name)
        gt_text = norm.normalize(str(row['plate_text']))
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        total_start = time.time()
        
        # 1. Detection
        det_start = time.time()
        det_res = det_model.predict(img, verbose=False)
        latencies['yolo'].append(time.time() - det_start)
        
        pred_box = None
        if len(det_res) > 0 and len(det_res[0].boxes) > 0:
            boxes = det_res[0].boxes
            best_idx = boxes.conf.argmax().item()
            pred_box = boxes.xyxy[best_idx].tolist()
            
        if pred_box is None:
            detector_misses += 1
            no_reads += 1
            latencies['ocr'].append(0)
            latencies['total'].append(time.time() - total_start)
            continue
            
        detector_hits += 1
        
        # 2. Optimized Crop (4px padding)
        x1, y1, x2, y2 = map(int, pred_box)
        pad = 4
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w, x2 + pad)
        py2 = min(h, y2 + pad)
        crop = img[py1:py2, px1:px2]
        
        # 3. Recognition Only
        ocr_start = time.time()
        rec_res = list(rec_model.predict(input=crop, batch_size=1))
        
        raw_text = ""
        if len(rec_res) > 0:
            raw_text = rec_res[0]['rec_text']
            
        latencies['ocr'].append(time.time() - ocr_start)
        
        # 4. Normalization
        pred_text = norm.normalize(raw_text)
        latencies['total'].append(time.time() - total_start)
        
        # Evaluation
        import difflib
        sm = difflib.SequenceMatcher(None, gt_text, pred_text)
        matched_chars += sum(n for i, j, n in sm.get_matching_blocks())
        total_chars += len(gt_text)
        
        match = (pred_text == gt_text)
        if match:
            e2e_exact_matches += 1
        else:
            if pred_text == "": no_reads += 1
            else: wrong_reads += 1
            
        results.append({
            'img': img_name,
            'gt': gt_text,
            'pred': pred_text,
            'match': match
        })

    total_imgs = len(results) + detector_misses
    e2e_accuracy = e2e_exact_matches / total_imgs if total_imgs else 0
    char_acc = matched_chars / total_chars if total_chars else 0
    no_read_rate = no_reads / total_imgs if total_imgs else 0
    wrong_read_rate = wrong_reads / total_imgs if total_imgs else 0
    
    summary = {
        "dataset_size": total_imgs,
        "exact_match": e2e_accuracy,
        "character_accuracy": char_acc,
        "no_read_rate": no_read_rate,
        "wrong_read_rate": wrong_read_rate,
        "latencies": {
            "init_time": init_time,
            "mean_total": np.mean(latencies['total']),
            "median_total": np.median(latencies['total']),
            "p95_total": np.percentile(latencies['total'], 95),
            "min_total": np.min(latencies['total']),
            "max_total": np.max(latencies['total']),
            "mean_yolo": np.mean(latencies['yolo']),
            "mean_ocr": np.mean(latencies['ocr'])
        }
    }
    
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    with open(os.path.join(out_dir, "reports", "recognition_only_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    print(json.dumps(summary, indent=2))
    
if __name__ == "__main__":
    evaluate_recognition_only("datasets/chsatya_yolo", "outputs")
