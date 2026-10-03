import os
import sys
import pandas as pd
import cv2
import json

def visualize_results(output_dir, dataset_dir):
    results_path = os.path.join(output_dir, "reports", "benchmark_results.csv")
    if not os.path.exists(results_path):
        print("Results not found. Run benchmark.py first.")
        return
        
    df = pd.read_csv(results_path)
    images_dir = os.path.join(dataset_dir, "images")
    vis_dir = os.path.join(output_dir, "visualizations")
    
    # Take 5 matches and 5 failures
    matches = df[df['match'] == True].head(5)
    failures = df[df['match'] == False].head(5)
    
    samples = pd.concat([matches, failures])
    
    # We don't have bounding boxes saved in results, wait! 
    # Let me edit pipeline to return bbox, which is already done. But benchmark doesn't save bbox!
    # That's fine, we can just overlay the predicted text on top left for visualization.
    
    for _, row in samples.iterrows():
        img_name = row['filename']
        img_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        text = f"Pred: {row['pred_text']} | GT: {row['gt_text']}"
        color = (0, 255, 0) if row['match'] else (0, 0, 255)
        
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"DetConf:{row['detector_conf']:.2f} OCRConf:{row['ocr_conf']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        out_path = os.path.join(vis_dir, f"vis_{img_name}")
        cv2.imwrite(out_path, img)
        
    print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    visualize_results("C:/Users/Krishi/.gemini/antigravity/scratch/anpr/outputs", "C:/Users/Krishi/.gemini/antigravity/scratch/anpr/datasets")
