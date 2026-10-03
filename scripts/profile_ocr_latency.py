import time
import cv2
import numpy as np
from paddleocr import PaddleOCR

def profile_ocr():
    # Attempt to load just the recognizer if possible, but we'll test the pipeline first
    print("Initializing OCR...")
    ocr = PaddleOCR(lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, device='cpu')
    
    # Create fake plate crops of different sizes
    sizes = [(640, 480), (320, 100), (150, 40), (100, 30)]
    
    for w, h in sizes:
        print(f"\n--- Profiling Crop Size: {w}x{h} ---")
        img = np.ones((h, w, 3), dtype=np.uint8) * 255
        cv2.putText(img, "MH01AB1234", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Warmup
        ocr.predict(img)
        
        latencies = []
        for _ in range(10):
            start = time.time()
            ocr.predict(img)
            latencies.append(time.time() - start)
            
        print(f"Mean Latency: {np.mean(latencies):.4f}s")
        print(f"Min Latency: {np.min(latencies):.4f}s")
        print(f"Max Latency: {np.max(latencies):.4f}s")

if __name__ == "__main__":
    profile_ocr()
