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
from paddleocr import PaddleOCR
from src.normalizer import TextNormalizer

def is_valid_plate(text):
    pattern = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$')
    return bool(pattern.match(text))

def run_benchmark(base_dir, out_dir):
    print("Initializing YOLO...")
    det_model = YOLO('runs/detect/models/plate_detector/weights/best.pt')
    norm = TextNormalizer()
    
    print("Initializing PaddleOCR CPU Static...")
    ocr_static = PaddleOCR(lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device='cpu', engine='paddle_static')
    
    print("Initializing PaddleOCR ONNX Runtime...")
    ocr_onnx = PaddleOCR(lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device='cpu', engine='onnxruntime')
    
    gt_df = pd.read_csv(os.path.join(base_dir, 'e2e_ground_truth.csv'))
    test_df = gt_df[gt_df['split'] == 'test'].copy()
    images_dir = os.path.join(base_dir, 'images', 'test')
    
    # Warmup
    if len(test_df) > 0:
        first_img = cv2.imread(os.path.join(images_dir, test_df.iloc[0]['image_path']))
        if first_img is not None:
            det_model.predict(first_img, verbose=False)
            h, w = first_img.shape[:2]
            crop = first_img[max(0, h//2-50):min(h, h//2+50), max(0, w//2-100):min(w, w//2+100)]
            _ = list(ocr_static.predict(crop))
            _ = list(ocr_onnx.predict(crop))
            
    latencies = {'A_static': [], 'B_onnx': [], 'C_advanced': []}
    yolo_lats = []
    crop_lats = []
    norm_lats = []
    
    results = {'A_static': [], 'B_onnx': [], 'C_advanced': []}
    plate_type = {}
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_name = row['image_path']
        img_path = os.path.join(images_dir, img_name)
        gt_text = norm.normalize(str(row['plate_text']))
        
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        # 1. Detection
        det_start = time.time()
        det_res = det_model.predict(img, verbose=False)
        yolo_lats.append(time.time() - det_start)
        
        pred_box = None
        if len(det_res) > 0 and len(det_res[0].boxes) > 0:
            boxes = det_res[0].boxes
            best_idx = boxes.conf.argmax().item()
            pred_box = boxes.xyxy[best_idx].tolist()
            
        if not pred_box:
            for cfg in ['A_static', 'B_onnx', 'C_advanced']:
                results[cfg].append({'gt': gt_text, 'pred': '', 'match': False})
                latencies[cfg].append(0)
            plate_type[idx] = 1
            crop_lats.append(0)
            norm_lats.append(0)
            continue
            
        # 2. Crop
        crop_start = time.time()
        x1, y1, x2, y2 = map(int, pred_box)
        pad = 4
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w, x2 + pad)
        py2 = min(h, y2 + pad)
        crop = img[py1:py2, px1:px2]
        crop_lats.append(time.time() - crop_start)
        
        # --- Config A: Static ---
        start_a = time.time()
        res_a = list(ocr_static.predict(crop))
        raw_a = "".join(res_a[0]['rec_texts']) if res_a and 'rec_texts' in res_a[0] else ""
        latencies['A_static'].append(time.time() - start_a)
        
        lines = len(res_a[0]['rec_texts']) if res_a and 'rec_texts' in res_a[0] else 1
        plate_type[idx] = lines
        
        # --- Config B: ONNX ---
        start_b = time.time()
        res_b = list(ocr_onnx.predict(crop))
        raw_b = "".join(res_b[0]['rec_texts']) if res_b and 'rec_texts' in res_b[0] else ""
        latencies['B_onnx'].append(time.time() - start_b)
        
        # --- Config C: Advanced Fallback (Static) ---
        start_c = time.time()
        pred_c_raw = raw_a
        conf_a = res_a[0]['rec_scores'][0] if res_a and 'rec_scores' in res_a[0] and len(res_a[0]['rec_scores'])>0 else 0
        norm_a = norm.normalize(raw_a)
        if not is_valid_plate(norm_a) or conf_a < 0.90:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
            res_c1 = list(ocr_static.predict(cl1))
            txt_c1 = "".join(res_c1[0]['rec_texts']) if res_c1 and 'rec_texts' in res_c1[0] else ""
            
            if is_valid_plate(norm.normalize(txt_c1)):
                pred_c_raw = txt_c1
            else:
                resized = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                res_c2 = list(ocr_static.predict(resized))
                txt_c2 = "".join(res_c2[0]['rec_texts']) if res_c2 and 'rec_texts' in res_c2[0] else ""
                pred_c_raw = txt_c2 if is_valid_plate(norm.normalize(txt_c2)) else pred_c_raw
        latencies['C_advanced'].append(time.time() - start_c)
        
        norm_start = time.time()
        pred_a = norm.normalize(raw_a)
        pred_b = norm.normalize(raw_b)
        pred_c = norm.normalize(pred_c_raw)
        norm_lats.append(time.time() - norm_start)
        
        results['A_static'].append({'gt': gt_text, 'pred': pred_a, 'match': pred_a == gt_text, 'lines': lines})
        results['B_onnx'].append({'gt': gt_text, 'pred': pred_b, 'match': pred_b == gt_text, 'lines': lines})
        results['C_advanced'].append({'gt': gt_text, 'pred': pred_c, 'match': pred_c == gt_text, 'lines': lines})
        
    summary = {}
    for cfg in ['A_static', 'B_onnx', 'C_advanced']:
        matches = sum(1 for r in results[cfg] if r['match'])
        no_reads = sum(1 for r in results[cfg] if r['pred'] == '')
        wrong_reads = len(results[cfg]) - matches - no_reads
        
        import difflib
        matched_chars = 0
        total_chars = 0
        for r in results[cfg]:
            sm = difflib.SequenceMatcher(None, r['gt'], r['pred'])
            matched_chars += sum(n for i, j, n in sm.get_matching_blocks())
            total_chars += len(r['gt'])
            
        single_line_total = sum(1 for r in results[cfg] if r['lines'] == 1)
        single_line_matches = sum(1 for r in results[cfg] if r['lines'] == 1 and r['match'])
        two_line_total = sum(1 for r in results[cfg] if r['lines'] > 1)
        two_line_matches = sum(1 for r in results[cfg] if r['lines'] > 1 and r['match'])
            
        summary[cfg] = {
            'exact_match': matches / len(test_df),
            'char_acc': matched_chars / total_chars,
            'no_read_rate': no_reads / len(test_df),
            'wrong_read_rate': wrong_reads / len(test_df),
            'mean_ocr_latency': np.mean(latencies[cfg]),
            'median_ocr_latency': np.median(latencies[cfg]),
            'p95_ocr_latency': np.percentile(latencies[cfg], 95),
            'min_ocr_latency': np.min(latencies[cfg]),
            'max_ocr_latency': np.max(latencies[cfg]),
            'total_e2e_latency_mean': np.mean(yolo_lats) + np.mean(crop_lats) + np.mean(latencies[cfg]) + np.mean(norm_lats),
            'single_line_count': single_line_total,
            'two_line_count': two_line_total,
            'single_line_acc': single_line_matches / single_line_total if single_line_total else 0,
            'two_line_acc': two_line_matches / two_line_total if two_line_total else 0
        }
        
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    with open(os.path.join(out_dir, "reports", "onnx_full_pipeline_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_benchmark("datasets/chsatya_yolo", "outputs")
