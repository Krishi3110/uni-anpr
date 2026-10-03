import os
import sys
import json
import pandas as pd
import time
import argparse
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import ANPRPipeline
from src.normalizer import TextNormalizer

def run_benchmark(dataset_dir, output_dir, mode="smoke_test"):
    csv_path = os.path.join(dataset_dir, "yolo", "labels", "test")
    # For full benchmark, we only use the HELD-OUT test set from the split
    images_dir = os.path.join(dataset_dir, "yolo", "images", "test")
    
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} not found. Run prepare_yolo_dataset.py first.")
        return
        
    # Re-read the main annotations.csv to get the ground truth texts
    main_csv_path = os.path.join(dataset_dir, "annotations.csv")
    df_all = pd.read_csv(main_csv_path)
    
    # Filter to only the images in the test set
    test_images = set(os.listdir(images_dir))
    df = df_all[df_all['image_path'].isin(test_images)].copy()
    
    if mode == "smoke_test":
        df = df.head(15)  # 15 timed images for smoke test
        print(f"Running SMOKE TEST on {len(df)} images.")
    else:
        print(f"Running FULL BENCHMARK on {len(df)} images.")
        
    pipeline = ANPRPipeline()
    
    print("Loading models...")
    pipeline.load_models()
    
    # WARMUP
    print("Running warmup inference...")
    warmup_start = time.time()
    if len(df) > 0:
        first_img = os.path.join(images_dir, df.iloc[0]['image_path'])
        pipeline.process_image(first_img)
    warmup_time = time.time() - warmup_start
    print(f"Warmup complete in {warmup_time:.2f} seconds.")
    
    results = []
    
    # Latency tracking
    total_latencies = []
    det_latencies = []
    prep_latencies = []
    ocr_latencies = []
    norm_latencies = []
    
    plates_detected = 0
    exact_matches = 0
    char_matches = 0
    total_chars = 0
    no_reads = 0
    wrong_reads = 0
    
    norm = TextNormalizer()
    
    benchmark_start_time = time.time()
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_name = row['image_path']
        img_path = os.path.join(images_dir, img_name)
        gt_text = norm.normalize(str(row['plate_text']))
        
        if not os.path.exists(img_path):
            continue
            
        res = pipeline.process_image(img_path)
        if res is None:
            continue
            
        latency = res.get('latency_sec', 0)
        total_latencies.append(latency)
        
        if res.get('status') != "NO_PLATE_DETECTED":
            det_latencies.append(res.get('det_latency', 0))
            prep_latencies.append(res.get('prep_latency', 0))
            ocr_latencies.append(res.get('ocr_latency', 0))
            norm_latencies.append(res.get('norm_latency', 0))
            plates_detected += 1
            
            pred_text = res['plate']
            
            # Character accuracy for OCR
            import difflib
            sm = difflib.SequenceMatcher(None, gt_text, pred_text)
            matches = sum(n for i, j, n in sm.get_matching_blocks())
            char_matches += matches
            total_chars += len(gt_text)
            
            if pred_text == "":
                no_reads += 1
            elif pred_text != gt_text:
                wrong_reads += 1
                
        pred_text = res['plate']
        det_conf = res['detector_confidence']
        status = res['status']
        
        # Calculate accuracy
        match = (pred_text == gt_text) and (len(gt_text) > 0)
        
        if match:
            exact_matches += 1
            
        results.append({
            "image_path": img_name,
            "gt_text": gt_text,
            "pred_text": pred_text,
            "match": match,
            "detector_conf": det_conf,
            "ocr_conf": res['ocr_confidence'],
            "final_conf": res['final_confidence'],
            "status": status,
            "latency": latency,
            "bbox": res.get("bbox", [])
        })
        
    benchmark_duration = time.time() - benchmark_start_time
    total_images = len(results)
    
    if total_images == 0:
        print("No images processed.")
        return
        
    recall = plates_detected / total_images
    accuracy = exact_matches / total_images
    
    # Calculate Latency Stats
    mean_lat = np.mean(total_latencies) if total_latencies else 0
    median_lat = np.median(total_latencies) if total_latencies else 0
    p95_lat = np.percentile(total_latencies, 95) if total_latencies else 0
    min_lat = np.min(total_latencies) if total_latencies else 0
    max_lat = np.max(total_latencies) if total_latencies else 0
    
    mean_det = np.mean(det_latencies) if det_latencies else 0
    mean_prep = np.mean(prep_latencies) if prep_latencies else 0
    mean_ocr = np.mean(ocr_latencies) if ocr_latencies else 0
    mean_norm = np.mean(norm_latencies) if norm_latencies else 0
    
    images_per_second = total_images / benchmark_duration if benchmark_duration > 0 else 0
    
    ocr_exact_acc = exact_matches / plates_detected if plates_detected > 0 else 0
    ocr_char_acc = char_matches / total_chars if total_chars > 0 else 0
    no_read_rate = no_reads / plates_detected if plates_detected > 0 else 0
    wrong_read_rate = wrong_reads / plates_detected if plates_detected > 0 else 0
    
    summary = {
        "dataset_size": total_images,
        "mode": mode,
        "detector_success_rate": recall,
        "ocr_exact_match_accuracy": ocr_exact_acc,
        "ocr_character_accuracy": ocr_char_acc,
        "ocr_no_read_rate": no_read_rate,
        "ocr_wrong_read_rate": wrong_read_rate,
        "end_to_end_exact_match_accuracy": accuracy,
        
        "latency_metrics": {
            "warmup_time_sec": warmup_time,
            "steady_state_mean_latency_sec": mean_lat,
            "median_latency_sec": median_lat,
            "p95_latency_sec": p95_lat,
            "min_latency_sec": min_lat,
            "max_latency_sec": max_lat,
            "mean_yolo_det_latency_sec": mean_det,
            "mean_preprocessing_latency_sec": mean_prep,
            "mean_ocr_latency_sec": mean_ocr,
            "mean_normalization_latency_sec": mean_norm,
            "images_per_second": images_per_second
        }
    }
    
    print("\n=== BENCHMARK SUMMARY ===")
    print(json.dumps(summary, indent=2))
    
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "failures"), exist_ok=True)
    
    # Save outputs
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "reports", f"{mode}_results.csv"), index=False)
    with open(os.path.join(output_dir, "reports", f"{mode}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    failed = [r for r in results if not r['match']]
    pd.DataFrame(failed).to_csv(os.path.join(output_dir, "failures", f"{mode}_failures.csv"), index=False)
    
    # Generate visualization for a few failures
    import cv2
    import random
    vis_count = 0
    
    # Shuffle or select up to 50 failures
    sample_fails = failed[:50]
    
    for fail in sample_fails:
        img_p = os.path.join(images_dir, fail['image_path'])
        if not os.path.exists(img_p):
            continue
        im = cv2.imread(img_p)
        if im is not None:
            bbox = fail.get('bbox', [])
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            text = f"GT: {fail['gt_text']} | PRED: {fail['pred_text']} | CONF: {fail['ocr_conf']:.2f}"
            cv2.putText(im, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out_p = os.path.join(output_dir, "failures", f"fail_{fail['image_path']}")
            cv2.imwrite(out_p, im)
            vis_count += 1
            
    print(f"Saved {vis_count} failure visualizations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run on a small subset")
    args = parser.parse_args()
    
    base_out = "C:/Users/Krishi/.gemini/antigravity/scratch/anpr/outputs"
    dataset_dir = "C:/Users/Krishi/.gemini/antigravity/scratch/anpr/datasets"
    
    mode = "smoke_test" if args.smoke_test else "full_benchmark"
    run_benchmark(dataset_dir, base_out, mode=mode)
