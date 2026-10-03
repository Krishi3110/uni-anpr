import os
import yaml
import re
import cv2
import numpy as np
import time
import logging
import json
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from ultralytics import YOLO
from paddleocr import PaddleOCR
from src.normalizer import TextNormalizer
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# Structured Logging Setup
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage()
        }
        if hasattr(record, "extra_info"):
            log_record.update(record.extra_info)
        return json.dumps(log_record)

logger = logging.getLogger("anpr_api")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(JSONFormatter())
logger.addHandler(ch)

app = FastAPI(title="ANPR API")

# Load Configuration
with open("config/prod_v1.yaml", "r") as f:
    config = yaml.safe_load(f)

logger.info("Initializing models...", extra={"extra_info": {"engine": config["pipeline"]["ocr"]["engine"]}})
det_model = YOLO(config["pipeline"]["detector"]["model_path"])
ocr_cfg = config["pipeline"]["ocr"]
ocr_model = PaddleOCR(
    lang=ocr_cfg["lang"],
    use_doc_orientation_classify=ocr_cfg["use_doc_orientation_classify"],
    use_doc_unwarping=ocr_cfg["use_doc_unwarping"],
    use_textline_orientation=ocr_cfg["use_textline_orientation"],
    engine=ocr_cfg["engine"],
    device=ocr_cfg["device"]
)
norm = TextNormalizer()
plate_regex = re.compile(ocr_cfg["regex_validation"])

def query_vehicle_status(plate_text: str) -> str:
    """PostgreSQL lookup for the vehicle status."""
    use_mock = os.getenv("USE_MOCK_DB", "true").lower() == "true"
    
    if use_mock:
        if plate_text.startswith("MH"): return "STUDENT"
        if plate_text.startswith("DL"): return "STAFF"
        if plate_text.startswith("KA"): return "VISITOR"
        return "UNKNOWN"
        
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            dbname=os.getenv("DB_NAME", "anpr_db")
        )
        cur = conn.cursor()
        cur.execute("SELECT status FROM vehicles WHERE plate = %s", (plate_text,))
        res = cur.fetchone()
        conn.close()
        return res[0] if res else "UNKNOWN"
    except Exception as e:
        logger.error(f"DB Error: {e}")
        return "UNKNOWN"

@app.post("/api/anpr/scan")
async def scan_plate(image: UploadFile = File(...)):
    start_time = time.time()
    
    img_bytes = await image.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        logger.error("Invalid image received.")
        return {"error": "Invalid image"}
        
    h, w = img.shape[:2]
    
    # 1. Detect Plate
    det_res = det_model.predict(img, verbose=False)
    if not det_res or len(det_res[0].boxes) == 0:
        latency = round(time.time() - start_time, 3)
        res = {
            "plate": "",
            "detector_confidence": 0.0,
            "ocr_confidence": 0.0,
            "final_confidence": 0.0,
            "status": "MANUAL_VERIFICATION",
            "message": "No plate detected",
            "latency_s": latency
        }
        logger.info("Scan completed (No Plate)", extra={"extra_info": res})
        return res
        
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
    res_ocr = list(ocr_model.predict(crop))
    raw_text = ""
    ocr_conf = 0.0
    if res_ocr and res_ocr[0] and 'rec_texts' in res_ocr[0] and len(res_ocr[0]['rec_texts']) > 0:
        raw_text = "".join(res_ocr[0]['rec_texts'])
        scores = res_ocr[0]['rec_scores']
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
        
    latency = round(time.time() - start_time, 3)
    response = {
        "plate": plate_text,
        "detector_confidence": round(det_conf, 4),
        "ocr_confidence": round(ocr_conf, 4),
        "final_confidence": round(final_conf, 4),
        "status": status,
        "is_format_valid": is_valid,
        "latency_s": latency
    }
    
    logger.info(f"Scan completed: {status}", extra={"extra_info": response})
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
