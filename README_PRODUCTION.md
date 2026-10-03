# ANPR Production API Setup

## Dependencies
Ensure the environment uses Python 3.10-3.12. Install the exact production requirements via `uv pip` (or standard pip):

```bash
uv pip install fastapi uvicorn python-multipart ultralytics paddleocr onnxruntime psycopg2-binary pyyaml opencv-python
```

*Note: PaddleOCR internally requires `paddlepaddle`. If your environment defaults to CPU, standard `paddlepaddle` is installed. ONNX Runtime handles the CPU acceleration.*

## Running the API

Start the FastAPI application via uvicorn. To bind it to `0.0.0.0` port `8000`:

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

*Note: To run as a daemon or background process in a Linux environment, you can use `nohup python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 &` or deploy via a process manager like systemd, Docker, or pm2.*

## Endpoint Reference

### `POST /api/anpr/scan`

**Input:** Form-data with key `image` containing the vehicle image file.

**Output (JSON):**
```json
{
  "plate": "MH47N0712",
  "detector_confidence": 0.9650,
  "ocr_confidence": 0.9820,
  "final_confidence": 0.9476,
  "status": "STUDENT",
  "is_format_valid": true,
  "latency_s": 0.150
}
```

## Security & Logging
- **No PII:** The API response strictly returns vehicle category status (`STUDENT`, `STAFF`, `VISITOR`, `UNKNOWN`, `MANUAL_VERIFICATION`). It never returns owner names, contact info, or PII.
- **Image Retention:** Uploaded vehicle images are processed in RAM using `cv2.imdecode` and are immediately discarded. They are never written to disk.
- **Structured Logging:** All scans are logged via standard stdout in JSON format to seamlessly integrate with ELK, Datadog, or CloudWatch, allowing metrics tracking on latency and match rates without leaking binary data.
