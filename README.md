# ANPR Production Pipeline

This repository contains the production-ready Automatic Number Plate Recognition (ANPR) pipeline, utilizing YOLOv8n for plate localization and PaddleOCR 3.7 with ONNX Runtime for sub-second, highly-accurate text recognition. 

It exposes a FastAPI backend integrated with PostgreSQL (or a mock fallback) to instantly categorize vehicles for gate security.

---

## Architecture Overview
- **Detector**: YOLOv8n (`models/plate_detector/best.pt`) targeting `license_plate` classes. **Uses NVIDIA GPU if available**.
- **OCR Engine**: PaddleOCR 3.7.0 running the complete Text Detection + Recognition pipeline. Accelerated using **ONNX Runtime on CPU**.
- **Normalization**: Enforces strict Indian license plate regex logic (`^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$`).
- **Safety Gate**: Rejects any plate where `YOLO_confidence * OCR_confidence < 0.90`, defaulting to `MANUAL_VERIFICATION`.

---

## 🛠️ Setup Instructions (Windows CMD)

**CRITICAL PREREQUISITE**: You **must** use **Python 3.12**. Python 3.14 breaks compatibility with the current PaddlePaddle 3.2.0 binaries. 

1. **Clone the repository:**
   ```cmd
   git clone <YOUR_REPOSITORY_URL>
   cd anpr
   ```

2. **Verify Python Version:**
   ```cmd
   python --version
   ```
   *Ensure the output is 3.12.x.*

3. **Create and activate a virtual environment:**
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. **Install all locked dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```
   *(Note: The environment handles CPU acceleration seamlessly via ONNX Runtime.)*

5. **Set up Environment Variables:**
   ```cmd
   copy .env.example .env
   ```
   *By default, `.env` enables `USE_MOCK_DB=true` so you can test immediately without providing real PostgreSQL credentials.*

6. **Start the API Server:**
   ```cmd
   python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

7. **Access Swagger UI:**
   Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.

8. **Run the Automated Integration Test:**
   Open a new CMD window (leave the API running) and execute:
   ```cmd
   .venv\Scripts\activate
   python test_api.py
   ```
   *This will run 7 edge-case scenarios and output a PASS/FAIL summary.*

9. **Manual Vehicle-Image Test (cURL):**
   ```cmd
   curl -X POST "http://127.0.0.1:8000/api/anpr/scan" ^
        -H "accept: application/json" ^
        -H "Content-Type: multipart/form-data" ^
        -F "image=@path_to_your_vehicle_image.jpg"
   ```

---

## 📋 Team Testing Checklist

Please execute this checklist locally to confirm your hardware setup perfectly replicates the pipeline:

- [ ] **Device Specs:** Note your CPU & GPU model. (YOLO will auto-select NVIDIA GPU if CUDA is available, otherwise CPU. OCR always utilizes CPU ONNX).
- [ ] **Python Version:** Verified Python 3.12.
- [ ] **Startup Success:** `uvicorn src.api:app` bound to port 8000 successfully without crash.
- [ ] **Clear Plate Test:** Test API passes (e.g., `STAFF` returned for known `DL` plates).
- [ ] **Unknown Plate Test:** Test API passes (e.g., `UNKNOWN` returned for an unrecognized plate prefix).
- [ ] **Low-Confidence / No-Plate Test:** Test API passes (returns `MANUAL_VERIFICATION`).
- [ ] **Observed Latency:** Ensure API JSON outputs `"latency_s"` well under `1.0s` (after the first warm-up request).
- [ ] **Issues:** Log any pipeline setup errors in the project issue tracker.

---

## Note on Datasets
The `datasets/` directory containing proprietary images is **excluded** from Git for security and bandwidth reasons. This repository strictly tracks the inference code, weights, configuration, and API deployment layer.
