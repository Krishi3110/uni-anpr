import argparse
import os
import time
from ultralytics import YOLO

def validate_detector(model_path, data_yaml):
    print(f"Loading trained model: {model_path}")
    model = YOLO(model_path)
    
    # We want to evaluate only on the test set.
    # Ultralytics validates on the `val` split by default. 
    # To run on test set, we must pass split='test'.
    
    print("Running validation on the test split...")
    
    # Run warmup
    for _ in range(5):
        # random image for warmup
        dummy = os.path.join(os.path.dirname(data_yaml), "images", "test")
        import glob
        img = glob.glob(os.path.join(dummy, "*.jpg"))[0]
        model.predict(img, verbose=False)
        
    start_time = time.time()
    
    results = model.val(
        data=data_yaml,
        split="test",
        device=0,
        plots=False,
        verbose=True
    )
    
    duration = time.time() - start_time
    
    metrics = results.results_dict
    
    p = metrics.get('metrics/precision(B)', 0.0)
    r = metrics.get('metrics/recall(B)', 0.0)
    map50 = metrics.get('metrics/mAP50(B)', 0.0)
    map5095 = metrics.get('metrics/mAP50-95(B)', 0.0)
    
    # speed metrics in ms per image
    speed = results.speed
    preprocess_ms = speed.get('preprocess', 0.0)
    inference_ms = speed.get('inference', 0.0)
    postprocess_ms = speed.get('postprocess', 0.0)
    
    total_ms_per_image = preprocess_ms + inference_ms + postprocess_ms
    images_per_sec = 1000.0 / total_ms_per_image if total_ms_per_image > 0 else 0
    
    print("\n=== DETECTOR TEST SET EVALUATION ===")
    print(f"Precision:   {p:.4f}")
    print(f"Recall:      {r:.4f}")
    print(f"mAP@50:      {map50:.4f}")
    print(f"mAP@50-95:   {map5095:.4f}")
    print(f"Preprocess:  {preprocess_ms:.2f} ms")
    print(f"Inference:   {inference_ms:.2f} ms")
    print(f"Postprocess: {postprocess_ms:.2f} ms")
    print(f"Total/img:   {total_ms_per_image:.2f} ms")
    print(f"Images/sec:  {images_per_sec:.2f} FPS")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="runs/detect/plate_detector/weights/best.pt")
    parser.add_argument("--data", type=str, default="datasets/yolo/data.yaml")
    args = parser.parse_args()
    
    validate_detector(args.model, args.data)
