import time
import logging

# Suppress PaddleOCR debug logs
logging.getLogger("ppocr").setLevel(logging.ERROR)

class PlateOCR:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.ocr = None
        
    def load_model(self):
        try:
            from paddleocr import PaddleOCR
            # Use current 3.x API
            self.ocr = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device="gpu" if self.use_gpu else "cpu"
            )
        except ImportError as e:
            print(f"PaddleOCR not installed: {e}")
            self.ocr = None

    def recognize(self, img):
        # img is a cropped numpy array from cv2
        if self.ocr is None:
            return "", 0.0, 0.0
            
        start = time.time()
        
        try:
            import types
            results = self.ocr.predict(img)
            
            if isinstance(results, types.GeneratorType):
                results = list(results)
                
            if not results or len(results) == 0:
                return "", 0.0, time.time() - start
                
            first_res = results[0]
            
            # The new dict structure has 'rec_texts' and 'rec_scores'
            rec_texts = first_res.get('rec_texts', [])
            rec_scores = first_res.get('rec_scores', [])
            
            if not rec_texts:
                return "", 0.0, time.time() - start
                
            # For a license plate, join all text chunks (if multiple detected)
            full_text = "".join(rec_texts)
            # Average confidence
            avg_conf = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
            
            return full_text, float(avg_conf), time.time() - start
            
        except Exception as e:
            print(f"OCR Exception: {e}")
            return "", 0.0, time.time() - start
