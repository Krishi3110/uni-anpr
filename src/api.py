import os
import yaml
import re
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO
from paddleocr import PaddleOCR
from src.normalizer import TextNormalizer
import psycopg2

app = FastAPI(title="ANPR API")

# Load Configuration
with open("config/prod_v1.yaml", "r") as f:
    config = yaml.safe_load(f)

print("Loading models...")
det_model = YOLO(config["pipeline"]["detector"]["model_path"])
ocr_cfg = config["pipeline"]["ocr"]
ocr_model = PaddleOCR(
    lang=ocr_cfg["lang"],
    use_doc_orientation_classify=ocr_cfg["use_doc_orientation_classify"],
    use_doc_unwarping=ocr_cfg["use_doc_unwarping"],
    use_textline_orientation=ocr_cfg["use_textline_orientation"],
    engine=ocr_cfg["engine"],
    device=ocr_cfg["device"],
    show_log=False
)
norm = TextNormalizer()
plate_regex = re.compile(ocr_cfg["regex_validation"])

def query_vehicle_status(plate_text: str) -> str:
    """Mock PostgreSQL lookup for the vehicle status."""
    try:
        # Placeholder for actual DB connection
        # conn = psycopg2.connect("dbname=anpr user=guard")
        # cur = conn.cursor()
        # cur.execute("SELECT status FROM vehicles WHERE plate = %s", (plate_text,))
        # res = cur.fetchone()
        # return res[0] if res else "UNKNOWN"
        
        # Hardcoded mocks for demonstration
        if plate_text.startswith("MH"): return "STUDENT"
        if plate_text.startswith("DL"): return "STAFF"
        if plate_text.startswith("KA"): return "VISITOR"
        return "UNKNOWN"
    except Exception as e:
        return "UNKNOWN"

@app.post("/api/anpr/scan")
async def scan_plate(image: UploadFile = File(...)):
    img_bytes = await image.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Invalid image"}
        
    h, w = img.shape[:2]
    
    # 1. Detect Plate
    det_res = det_model.predict(img, verbose=False)
    if not det_res or len(det_res[0].boxes) == 0:
        return {
            "plate": "",
            "detector_confidence": 0.0,
            "ocr_confidence": 0.0,
            "final_confidence": 0.0,
            "status": "MANUAL_VERIFICATION",
            "message": "No plate detected"
        }
        
    boxes = det_res[0].boxes
    best_idx = boxes.conf.argmax().item()
    det_conf = float(boxes.conf[best_idx].item())
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
    
    # 2. Crop with padding
    pad = config["pipeline"]["crop"]["padding_px"]
    px1, py1 = max(0, x1 - pad), max(0, y1 - pad)
    px2, py2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img[py1:py2, px1:px2]
    
    # 3. OCR
    res = list(ocr_model.predict(crop))
    raw_text = ""
    ocr_conf = 0.0
    if res and res[0] and 'rec_texts' in res[0]:
        raw_text = "".join(res[0]['rec_texts'])
        # Average confidence if multi-line
        scores = res[0]['rec_scores']
        ocr_conf = sum(scores) / len(scores) if scores else 0.0
        
    # 4. Normalize
    plate_text = norm.normalize(raw_text)
    
    # 5. Validation and Confidence
    final_conf = det_conf * ocr_conf
    is_valid = bool(plate_regex.match(plate_text))
    
    # 6. Status Determination
    status = "MANUAL_VERIFICATION"
    if is_valid and final_conf >= config["pipeline"]["ocr"]["confidence_threshold"]:
        status = query_vehicle_status(plate_text)
        
    return {
        "plate": plate_text,
        "detector_confidence": round(det_conf, 4),
        "ocr_confidence": round(ocr_conf, 4),
        "final_confidence": round(final_conf, 4),
        "status": status,
        "is_format_valid": is_valid
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
