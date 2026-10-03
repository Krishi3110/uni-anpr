import sys
import pprint

try:
    from paddleocr import PaddleOCR
    print(f"PaddleOCR version: {paddleocr.__version__}")
    
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu"
    )
    
    import cv2
    import numpy as np
    
    # Create a dummy image with some text
    img = np.zeros((100, 300, 3), dtype=np.uint8)
    cv2.putText(img, "MH01DE2780", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if hasattr(ocr, 'predict'):
        res = ocr.predict(img)
    elif hasattr(ocr, 'ocr'):
        res = ocr.ocr(img)
    else:
        res = "No predict or ocr method found!"
        
    print("\n--- Raw Result ---")
    pprint.pprint(res)
    
except Exception as e:
    import traceback
    traceback.print_exc()
