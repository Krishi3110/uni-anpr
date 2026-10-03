import cv2
from .detector import PlateDetector
from .ocr import PlateOCR
from .preprocess import Preprocessor
from .normalizer import TextNormalizer
import time

class ANPRPipeline:
    def __init__(self):
        self.detector = PlateDetector()
        self.ocr = PlateOCR()
        self.preprocessor = Preprocessor()
        self.normalizer = TextNormalizer()
        
    def load_models(self):
        print("Loading Detector...")
        self.detector.load_model()
        print("Loading OCR...")
        self.ocr.load_model()
        
    def process_image(self, img_path):
        start = time.time()
        
        # 1. Detection
        boxes, det_confs, det_latency = self.detector.detect(img_path)
        
        if not boxes:
            return {
                "plate": "",
                "detector_confidence": 0.0,
                "ocr_confidence": 0.0,
                "final_confidence": 0.0,
                "status": "NO_PLATE_DETECTED",
                "latency_sec": time.time() - start,
                "bbox": []
            }
            
        # Take highest confidence box
        best_idx = det_confs.index(max(det_confs))
        best_box = boxes[best_idx]
        best_det_conf = det_confs[best_idx]
        
        # 2. Read image & 3. Crop & 4. Preprocess
        prep_start = time.time()
        img = cv2.imread(img_path)
        if img is None:
            return None
        cropped = self.preprocessor.crop(img, best_box)
        processed = self.preprocessor.process(cropped)
        prep_latency = time.time() - prep_start
        
        # 5. OCR
        raw_text, ocr_conf, ocr_latency = self.ocr.recognize(processed)
        
        # 6. Normalize & 7. Confidence & Status
        norm_start = time.time()
        normalized_text = self.normalizer.normalize(raw_text)
        final_conf, status = self.normalizer.calculate_confidence(best_det_conf, ocr_conf, normalized_text)
        norm_latency = time.time() - norm_start
        
        total_latency = time.time() - start
        
        return {
            "plate": normalized_text,
            "raw_text": raw_text,
            "detector_confidence": best_det_conf,
            "ocr_confidence": ocr_conf,
            "final_confidence": final_conf,
            "status": status,
            "latency_sec": total_latency,
            "det_latency": det_latency,
            "prep_latency": prep_latency,
            "ocr_latency": ocr_latency,
            "norm_latency": norm_latency,
            "bbox": best_box
        }
