import cv2
import numpy as np
import pprint
import paddleocr

print(f"PaddleOCR version: {paddleocr.__version__}")

from paddleocr import PaddleOCR

def run_test():
    ocr = PaddleOCR(
        lang='en',
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device='cpu'
    )
    
    print("Model initialized successfully.")
    
    # Create a dummy image mimicking a cropped plate
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, "MH01DE2780", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Predict
    import types
    res = ocr.predict(img)
    if isinstance(res, types.GeneratorType):
        res = list(res)
        
    print("\n--- Raw Result Structure ---")
    pprint.pprint(res)
    
    if res and len(res) > 0:
        first_res = res[0]
        rec_texts = first_res.get('rec_texts', [])
        rec_scores = first_res.get('rec_scores', [])
        
        full_text = "".join(rec_texts)
        avg_conf = sum(rec_scores) / len(rec_scores) if rec_scores else 0.0
        
        print("\n--- Extracted ---")
        print(f"Text: {full_text}")
        print(f"Confidence: {avg_conf:.4f}")
    else:
        print("No result found.")

if __name__ == "__main__":
    run_test()
