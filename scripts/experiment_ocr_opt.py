import os
import sys
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import re
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ocr import PlateOCR
from src.normalizer import TextNormalizer

# Strict Indian plate format validator
def is_valid_plate(text):
    # e.g., MH01AB1234, DL8CNA1234, KA19EJ1111, PY02S9326
    # Starts with 2 letters, then 1-2 digits, then 1-3 letters, then 4 digits
    pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$')
    return bool(pattern.match(text))

def run_experiment(base_dir, out_dir):
    print("Loading models...")
    det_model = YOLO('runs/detect/models/plate_detector/weights/best.pt')
    ocr_model = PlateOCR()
    ocr_model.load_model()
    norm = TextNormalizer()
    
    gt_df = pd.read_csv(os.path.join(base_dir, 'e2e_ground_truth.csv'))
    test_df = gt_df[gt_df['split'] == 'test'].copy()
    images_dir = os.path.join(base_dir, 'images', 'test')
    
    # Warmup
    if len(test_df) > 0:
        first_img = cv2.imread(os.path.join(images_dir, test_df.iloc[0]['image_path']))
        if first_img is not None:
            det_model.predict(first_img, verbose=False)
            ocr_model.recognize(first_img)
            
    results = {'BASELINE': [], 'OPTIMIZED': [], 'ADVANCED': []}
    latencies = {'BASELINE': [], 'OPTIMIZED': [], 'ADVANCED': []}
    
    os.makedirs(os.path.join(out_dir, "exp_failures"), exist_ok=True)
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_name = row['image_path']
        img_path = os.path.join(images_dir, img_name)
        gt_text = norm.normalize(str(row['plate_text']))
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # Detector is shared
        det_res = det_model.predict(img, verbose=False)
        pred_box = None
        if len(det_res) > 0 and len(det_res[0].boxes) > 0:
            boxes = det_res[0].boxes
            best_idx = boxes.conf.argmax().item()
            pred_box = boxes.xyxy[best_idx].tolist()
            
        if not pred_box:
            for cfg in ['BASELINE', 'OPTIMIZED', 'ADVANCED']:
                results[cfg].append({'img': img_name, 'gt': gt_text, 'pred': '', 'match': False, 'error': 'detector_miss'})
                latencies[cfg].append(0)
            continue
            
        x1, y1, x2, y2 = map(int, pred_box)
        
        # === BASELINE ===
        start_b = time.time()
        crop_b = img[max(0, y1):y2, max(0, x1):x2]
        raw_b, conf_b, _ = ocr_model.recognize(crop_b)
        pred_b = norm.normalize(raw_b)
        latencies['BASELINE'].append(time.time() - start_b)
        
        # === OPTIMIZED ===
        # Padding config: pad by 4 pixels to give DB detector room
        start_o = time.time()
        pad = 4
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w, x2 + pad)
        py2 = min(h, y2 + pad)
        crop_o = img[py1:py2, px1:px2]
        raw_o, conf_o, _ = ocr_model.recognize(crop_o)
        pred_o = norm.normalize(raw_o)
        latencies['OPTIMIZED'].append(time.time() - start_o)
        
        # === ADVANCED ===
        # Preprocessing Fallback: If not valid format or low conf, try alternatives
        start_a = time.time()
        
        candidates = [(pred_o, conf_o)] # start with optimized crop
        
        if not is_valid_plate(pred_o) or conf_o < 0.90:
            # Fallback 1: Grayscale + CLAHE
            gray = cv2.cvtColor(crop_o, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(gray)
            cl1_bgr = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
            raw_f1, conf_f1, _ = ocr_model.recognize(cl1_bgr)
            candidates.append((norm.normalize(raw_f1), conf_f1))
            
            # Fallback 2: Resize (x2)
            resized = cv2.resize(crop_o, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            raw_f2, conf_f2, _ = ocr_model.recognize(resized)
            candidates.append((norm.normalize(raw_f2), conf_f2))
            
        # Select best candidate: priority to valid format, then highest confidence
        best_pred = pred_o
        best_conf = conf_o
        valid_candidates = [c for c in candidates if is_valid_plate(c[0])]
        
        if valid_candidates:
            best_pred, best_conf = max(valid_candidates, key=lambda x: x[1])
        else:
            best_pred, best_conf = max(candidates, key=lambda x: x[1])
            
        latencies['ADVANCED'].append(time.time() - start_a)
        
        # Record results
        def categorize(pred, gt):
            if pred == gt: return None
            if pred == "": return "no_read"
            if gt in pred and len(pred) > len(gt): return "ocr_extra_text"
            if len(gt) == len(pred): return "ocr_char_confusion"
            return "other"
            
        results['BASELINE'].append({'img': img_name, 'gt': gt_text, 'pred': pred_b, 'match': pred_b == gt_text, 'error': categorize(pred_b, gt_text)})
        results['OPTIMIZED'].append({'img': img_name, 'gt': gt_text, 'pred': pred_o, 'match': pred_o == gt_text, 'error': categorize(pred_o, gt_text)})
        results['ADVANCED'].append({'img': img_name, 'gt': gt_text, 'pred': best_pred, 'match': best_pred == gt_text, 'error': categorize(best_pred, gt_text)})
        
        # Visualize if all failed or to compare
        if pred_b != gt_text or pred_o != gt_text or best_pred != gt_text:
            out_img = img.copy()
            cv2.rectangle(out_img, (px1, py1), (px2, py2), (0, 255, 0), 2)
            text = f"GT:{gt_text} | B:{pred_b} | O:{pred_o} | A:{best_pred}"
            cv2.putText(out_img, text, (px1, max(20, py1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            cv2.imwrite(os.path.join(out_dir, "exp_failures", f"cmp_{img_name}"), out_img)
            
    # Calculate metrics
    summary = {}
    for cfg in ['BASELINE', 'OPTIMIZED', 'ADVANCED']:
        matches = sum(1 for r in results[cfg] if r['match'])
        no_reads = sum(1 for r in results[cfg] if r['error'] == 'no_read' or r['pred'] == '')
        wrong_reads = len(results[cfg]) - matches - no_reads
        
        import difflib
        total_chars = 0
        matched_chars = 0
        for r in results[cfg]:
            sm = difflib.SequenceMatcher(None, r['gt'], r['pred'])
            matched_chars += sum(n for i, j, n in sm.get_matching_blocks())
            total_chars += len(r['gt'])
            
        summary[cfg] = {
            'exact_match_acc': matches / len(test_df),
            'char_acc': matched_chars / total_chars,
            'no_read_rate': no_reads / len(test_df),
            'wrong_read_rate': wrong_reads / len(test_df),
            'mean_latency': np.mean(latencies[cfg]),
            'median_latency': np.median(latencies[cfg]),
            'p95_latency': np.percentile(latencies[cfg], 95),
            'images_per_sec': 1.0 / np.mean(latencies[cfg]) if np.mean(latencies[cfg]) > 0 else 0
        }
        
    print(json.dumps(summary, indent=2))
    
    # Save comparison dataframe
    cmp_records = []
    for i in range(len(test_df)):
        cmp_records.append({
            'image': results['BASELINE'][i]['img'],
            'ground_truth': results['BASELINE'][i]['gt'],
            'baseline_pred': results['BASELINE'][i]['pred'],
            'optimized_pred': results['OPTIMIZED'][i]['pred'],
            'advanced_pred': results['ADVANCED'][i]['pred'],
            'baseline_error': results['BASELINE'][i]['error'],
            'optimized_error': results['OPTIMIZED'][i]['error'],
            'advanced_error': results['ADVANCED'][i]['error'],
            'latency_diff_o_b': latencies['OPTIMIZED'][i] - latencies['BASELINE'][i],
            'latency_diff_a_b': latencies['ADVANCED'][i] - latencies['BASELINE'][i],
        })
    pd.DataFrame(cmp_records).to_csv(os.path.join(out_dir, "reports", "ocr_opt_experiment.csv"), index=False)

if __name__ == "__main__":
    run_experiment("datasets/chsatya_yolo", "outputs")
