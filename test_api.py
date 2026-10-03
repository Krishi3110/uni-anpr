import requests
import time
import json
import cv2
import numpy as np

def post_image(img_path, description):
    print(f"\n--- Testing: {description} ---")
    with open(img_path, 'rb') as f:
        start_time = time.time()
        resp = requests.post("http://127.0.0.1:8000/api/anpr/scan", files={"image": f})
        end_time = time.time()
        print(f"Request latency: {round(end_time - start_time, 3)}s")
        if resp.status_code == 200:
            res = resp.json()
            print(json.dumps(res, indent=2))
        else:
            print("Error:", resp.status_code, resp.text)

# 1. STUDENT (MH)
post_image(r"datasets/chsatya_yolo/images/train/MH6.jpg", "Known STUDENT plate (MH)")

# 2. STAFF (DL)
post_image(r"datasets/chsatya_yolo/images/train/DL13.jpg", "Known STAFF plate (DL)")

# 3. VISITOR (KA)
post_image(r"datasets/chsatya_yolo/images/train/KA25.jpg", "Known VISITOR plate (KA)")

# 4. UNKNOWN (AP)
post_image(r"datasets/chsatya_yolo/images/train/AP21.jpg", "Unknown plate (AP)")

# Create corrupted images for edge cases
img = cv2.imread(r"datasets/chsatya_yolo/images/train/MH6.jpg")
h, w = img.shape[:2]

# 5. Invalid / Garbled OCR (Draw over the plate)
# Plate box for MH6 is around the bottom center. Let's just draw random text over the bottom half.
img_garbled = img.copy()
cv2.putText(img_garbled, "@@##XYZ", (w//2-50, h-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)
cv2.rectangle(img_garbled, (w//2-100, h-100), (w//2+100, h), (0,0,0), -1) # block plate
# draw something that looks like a plate to YOLO but has garbled text
cv2.rectangle(img_garbled, (w//2-80, h-80), (w//2+80, h-20), (255,255,255), -1)
cv2.putText(img_garbled, "XX99QQQ", (w//2-70, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.imwrite("garbled.jpg", img_garbled)
post_image("garbled.jpg", "Invalid/Garbled OCR (Fails regex)")

# 6. Low Confidence Result (Heavy blur on plate)
img_blur = img.copy()
# Blur the entire bottom half heavily
img_blur[h//2:h, :] = cv2.GaussianBlur(img_blur[h//2:h, :], (51, 51), 30)
cv2.imwrite("blur.jpg", img_blur)
post_image("blur.jpg", "Low Confidence Result (Blurry Plate)")

# 7. No detectable plate
img_no_plate = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.imwrite("no_plate.jpg", img_no_plate)
post_image("no_plate.jpg", "No Detectable Plate (Blank Image)")
