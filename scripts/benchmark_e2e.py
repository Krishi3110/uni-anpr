import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ocr import PlateOCR
from src.normalizer import TextNormalizer

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def evaluate_e2e(base_dir, out_dir):
    print("Loading models...")
    init_start = time.time()
    
    det_model = YOLO('runs/detect/models/plate_detector/weights/best.pt')
    ocr_model = PlateOCR()
    ocr_model.load_model()
    norm = TextNormalizer()
    
    init_time = time.time() - init_start
    print(f"Models initialized in {init_time:.2f}s")
    
    gt_df = pd.read_csv(os.path.join(base_dir, 'e2e_ground_truth.csv'))
    test_df = gt_df[gt_df['split'] == 'test'].copy()
    
    images_dir = os.path.join(base_dir, 'images', 'test')
    
    print(f"Running E2E Benchmark on {len(test_df)} test images.")
    
    # Warmup
    if len(test_df) > 0:
        first_img = cv2.imread(os.path.join(images_dir, test_df.iloc[0]['image_path']))
        if first_img is not None:
            det_model.predict(first_img, verbose=False)
            ocr_model.recognize(first_img)
            
    latencies = {
        'total': [],
        'yolo': [],
        'preproc': [],
        'ocr': [],
        'norm': []
    }
    
    results = []
    
    detector_hits = 0
    detector_misses = 0
    ocr_exact_matches = 0
    e2e_exact_matches = 0
    total_chars = 0
    matched_chars = 0
    no_reads = 0
    wrong_reads = 0
    
    ious = []
    
    os.makedirs(os.path.join(out_dir, "e2e_failures"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    
    start_time = time.time()
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_name = row['image_path']
        img_path = os.path.join(images_dir, img_name)
        gt_text = norm.normalize(str(row['plate_text']))
        gt_box = [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
        
        img = cv2.imread(img_path)
        if img is None: continue
        
        total_start = time.time()
        
        # 1. Detection
        det_start = time.time()
        det_res = det_model.predict(img, verbose=False)
        latencies['yolo'].append(time.time() - det_start)
        
        pred_box = None
        det_conf = 0.0
        if len(det_res) > 0 and len(det_res[0].boxes) > 0:
            boxes = det_res[0].boxes
            best_idx = boxes.conf.argmax().item()
            pred_box = boxes.xyxy[best_idx].tolist()
            det_conf = boxes.conf[best_idx].item()
            
        iou = bb_intersection_over_union(pred_box, gt_box) if pred_box else 0.0
        ious.append(iou)
        
        error_category = None
        
        if pred_box is None:
            detector_misses += 1
            error_category = 'detector_miss'
            no_reads += 1
            latencies['preproc'].append(0)
            latencies['ocr'].append(0)
            latencies['norm'].append(0)
            total_time = time.time() - total_start
            latencies['total'].append(total_time)
            
            results.append({
                'image_path': img_name,
                'gt_text': gt_text,
                'pred_text': '',
                'match': False,
                'iou': 0.0,
                'error_category': error_category
            })
            continue
            
        detector_hits += 1
        
        # 2. Preprocessing (Crop)
        prep_start = time.time()
        x1, y1, x2, y2 = map(int, pred_box)
        crop = img[max(0, y1):y2, max(0, x1):x2]
        latencies['preproc'].append(time.time() - prep_start)
        
        # 3. OCR
        ocr_start = time.time()
        raw_text, ocr_conf, _ = ocr_model.recognize(crop)
        latencies['ocr'].append(time.time() - ocr_start)
        
        # 4. Normalization
        norm_start = time.time()
        pred_text = norm.normalize(raw_text)
        latencies['norm'].append(time.time() - norm_start)
        
        latencies['total'].append(time.time() - total_start)
        
        # Evaluation
        import difflib
        sm = difflib.SequenceMatcher(None, gt_text, pred_text)
        matches = sum(n for i, j, n in sm.get_matching_blocks())
        matched_chars += matches
        total_chars += len(gt_text)
        
        match = (pred_text == gt_text)
        
        if match:
            e2e_exact_matches += 1
            ocr_exact_matches += 1
        else:
            if pred_text == "":
                no_reads += 1
            else:
                wrong_reads += 1
                
            if iou < 0.5:
                error_category = 'bad_bbox'
            elif gt_text in pred_text and len(pred_text) > len(gt_text):
                error_category = 'ocr_extra_text'
            elif len(gt_text) == len(pred_text):
                error_category = 'ocr_char_confusion'
            else:
                error_category = 'other'
                
        results.append({
            'image_path': img_name,
            'gt_text': gt_text,
            'pred_text': pred_text,
            'match': match,
            'iou': iou,
            'det_conf': det_conf,
            'ocr_conf': ocr_conf,
            'error_category': error_category
        })
        
        if not match:
            # Visualize
            vis = img.copy()
            # Draw GT box
            gx1, gy1, gx2, gy2 = map(int, gt_box)
            cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
            cv2.putText(vis, "GT: "+gt_text, (gx1, gy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw Pred box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, "PRED: "+pred_text, (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(os.path.join(out_dir, "e2e_failures", f"fail_{img_name}"), vis)

    duration = time.time() - start_time
    total_imgs = len(results)
    
    # YOLO Detector Test Metrics (mAP)
    # We will use ultralytics built-in val for precision, recall, mAP
    print("Running strict YOLO val on test set...")
    val_res = det_model.val(data=os.path.abspath(os.path.join(base_dir, 'data.yaml')), split='test', plots=False)
    
    det_precision = val_res.results_dict.get('metrics/precision(B)', 0.0)
    det_recall = val_res.results_dict.get('metrics/recall(B)', 0.0)
    det_map50 = val_res.results_dict.get('metrics/mAP50(B)', 0.0)
    det_map95 = val_res.results_dict.get('metrics/mAP50-95(B)', 0.0)
    
    mean_iou = np.mean(ious)
    det_success_rate = detector_hits / total_imgs if total_imgs else 0
    
    ocr_accuracy = ocr_exact_matches / detector_hits if detector_hits else 0
    e2e_accuracy = e2e_exact_matches / total_imgs if total_imgs else 0
    char_acc = matched_chars / total_chars if total_chars else 0
    no_read_rate = no_reads / total_imgs if total_imgs else 0
    wrong_read_rate = wrong_reads / total_imgs if total_imgs else 0
    
    # Calculate 95% CI for E2E Accuracy (Wald interval)
    z = 1.96
    p = e2e_accuracy
    ci_margin = z * np.sqrt((p * (1 - p)) / total_imgs) if total_imgs else 0
    
    summary = {
        "dataset_size": total_imgs,
        "detector_metrics": {
            "precision": det_precision,
            "recall": det_recall,
            "mAP50": det_map50,
            "mAP50_95": det_map95,
            "mean_iou": mean_iou,
            "success_rate": det_success_rate
        },
        "ocr_metrics_on_crops": {
            "exact_match": ocr_accuracy
        },
        "e2e_metrics": {
            "exact_match": e2e_accuracy,
            "exact_match_95ci": [max(0, p - ci_margin), min(1, p + ci_margin)],
            "character_accuracy": char_acc,
            "no_read_rate": no_read_rate,
            "wrong_read_rate": wrong_read_rate
        },
        "latency_metrics": {
            "initialization_sec": init_time,
            "total_mean_sec": np.mean(latencies['total']),
            "total_median_sec": np.median(latencies['total']),
            "total_p95_sec": np.percentile(latencies['total'], 95),
            "total_min_sec": np.min(latencies['total']),
            "total_max_sec": np.max(latencies['total']),
            "yolo_mean_sec": np.mean(latencies['yolo']),
            "preproc_mean_sec": np.mean(latencies['preproc']),
            "ocr_mean_sec": np.mean(latencies['ocr']),
            "norm_mean_sec": np.mean(latencies['norm']),
            "images_per_second": total_imgs / duration if duration else 0
        }
    }
    
    with open(os.path.join(out_dir, "reports", "e2e_benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    pd.DataFrame(results).to_csv(os.path.join(out_dir, "reports", "e2e_benchmark_results.csv"), index=False)
    print(json.dumps(summary, indent=2))
    
if __name__ == "__main__":
    evaluate_e2e("datasets/chsatya_yolo", "outputs")
