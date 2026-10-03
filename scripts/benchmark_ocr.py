import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ocr import PlateOCR
from src.normalizer import TextNormalizer

def run_ocr_benchmark(dataset_dir, output_dir, mode="smoke_test"):
    main_csv_path = os.path.join(dataset_dir, "annotations.csv")
    df = pd.read_csv(main_csv_path)
    images_dir = os.path.join(dataset_dir, "images")
    
    # We will use the 'test' split logic if we had one, but since we are only doing OCR now, 
    # let's just evaluate on a fixed reproducible random subset (test set) of the 1684 valid records.
    # To be consistent with previous split (70/20/10), we sample 170 images with seed 42.
    test_df = df.sample(n=170, random_state=42).copy()
    
    if mode == "smoke_test":
        test_df = test_df.head(15)
        print(f"Running OCR SMOKE TEST on {len(test_df)} images.")
    else:
        print(f"Running FULL OCR BENCHMARK on {len(test_df)} images.")
        
    ocr = PlateOCR()
    norm = TextNormalizer()
    
    print("Loading OCR Model...")
    ocr.load_model()
    
    print("Running warmup inference...")
    warmup_start = time.time()
    if len(test_df) > 0:
        first_img_path = os.path.join(images_dir, test_df.iloc[0]['image_path'])
        first_img = cv2.imread(first_img_path)
        if first_img is not None:
            ocr.recognize(first_img)
    warmup_time = time.time() - warmup_start
    print(f"Warmup complete in {warmup_time:.2f} seconds.")
    
    results = []
    total_latencies = []
    
    exact_matches = 0
    char_matches = 0
    total_chars = 0
    no_reads = 0
    wrong_reads = 0
    
    benchmark_start_time = time.time()
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        img_name = row['image_path']
        img_path = os.path.join(images_dir, img_name)
        gt_text = norm.normalize(str(row['plate_text']))
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        inf_start = time.time()
        # Pure OCR on the cropped plate (bypassing YOLO detector)
        raw_text, ocr_conf, _ = ocr.recognize(img)
        normalized_text = norm.normalize(raw_text)
        inf_latency = time.time() - inf_start
        
        total_latencies.append(inf_latency)
        
        import difflib
        sm = difflib.SequenceMatcher(None, gt_text, normalized_text)
        matches = sum(n for i, j, n in sm.get_matching_blocks())
        char_matches += matches
        total_chars += len(gt_text)
        
        if normalized_text == "":
            no_reads += 1
        elif normalized_text != gt_text:
            wrong_reads += 1
            
        match = (normalized_text == gt_text) and (len(gt_text) > 0)
        if match:
            exact_matches += 1
            
        results.append({
            "image_path": img_name,
            "gt_text": gt_text,
            "pred_text": normalized_text,
            "raw_text": raw_text,
            "match": match,
            "ocr_conf": ocr_conf,
            "latency": inf_latency
        })
        
    benchmark_duration = time.time() - benchmark_start_time
    total_images = len(results)
    
    if total_images == 0:
        return
        
    accuracy = exact_matches / total_images
    char_accuracy = char_matches / total_chars if total_chars > 0 else 0
    no_read_rate = no_reads / total_images
    wrong_read_rate = wrong_reads / total_images
    
    mean_lat = np.mean(total_latencies)
    median_lat = np.median(total_latencies)
    p95_lat = np.percentile(total_latencies, 95)
    min_lat = np.min(total_latencies)
    max_lat = np.max(total_latencies)
    
    images_per_second = total_images / benchmark_duration if benchmark_duration > 0 else 0
    
    summary = {
        "dataset_size": total_images,
        "mode": mode,
        "ocr_exact_match_accuracy": accuracy,
        "ocr_character_accuracy": char_accuracy,
        "ocr_no_read_rate": no_read_rate,
        "ocr_wrong_read_rate": wrong_read_rate,
        "latency_metrics": {
            "warmup_time_sec": warmup_time,
            "steady_state_mean_latency_sec": mean_lat,
            "median_latency_sec": median_lat,
            "p95_latency_sec": p95_lat,
            "min_latency_sec": min_lat,
            "max_latency_sec": max_lat,
            "images_per_second": images_per_second
        }
    }
    
    print("\n=== OCR BENCHMARK SUMMARY ===")
    print(json.dumps(summary, indent=2))
    
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "failures"), exist_ok=True)
    
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "reports", f"ocr_{mode}_results.csv"), index=False)
    with open(os.path.join(output_dir, "reports", f"ocr_{mode}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    failed = [r for r in results if not r['match']]
    pd.DataFrame(failed).to_csv(os.path.join(output_dir, "failures", f"ocr_{mode}_failures.csv"), index=False)
    
    vis_count = 0
    sample_fails = failed[:50]
    for fail in sample_fails:
        img_p = os.path.join(images_dir, fail['image_path'])
        im = cv2.imread(img_p)
        if im is not None:
            text = f"GT: {fail['gt_text']} | PRED: {fail['pred_text']} | CONF: {fail['ocr_conf']:.2f}"
            cv2.putText(im, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            out_p = os.path.join(output_dir, "failures", f"ocr_fail_{fail['image_path']}")
            cv2.imwrite(out_p, im)
            vis_count += 1
            
    print(f"Saved {vis_count} failure visualizations.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    base_out = "C:/Users/Krishi/.gemini/antigravity/scratch/anpr/outputs"
    dataset_dir = "C:/Users/Krishi/.gemini/antigravity/scratch/anpr/datasets"
    
    mode = "smoke_test" if args.smoke_test else "full_benchmark"
    run_ocr_benchmark(dataset_dir, base_out, mode=mode)
