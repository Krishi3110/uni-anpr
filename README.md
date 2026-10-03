# ANPR Prototype Pipeline

This project is a prototype Automatic Number Plate Recognition (ANPR) pipeline, using YOLO for plate detection and PaddleOCR for text recognition.

## Requirements
- Windows OS
- Python 3.8+
- PyTorch (CUDA recommended)
- Ultralytics (YOLO)
- PaddleOCR

## Setup Instructions (Windows CMD)

1. **Create and activate a virtual environment:**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```cmd
   pip install -r requirements.txt
   ```
   *(Note: For GPU acceleration, you may need to install the CUDA version of PyTorch and PaddlePaddle manually according to your CUDA version.)*

3. **Run Environment Diagnostic:**
   ```cmd
   python diagnostic.py
   ```

4. **Prepare Dataset:**
   This script attempts to download the Indian number plates dataset from Hugging Face.
   ```cmd
   python scripts\prepare_dataset.py
   ```

5. **Run Smoke Test (10 images):**
   ```cmd
   python scripts\benchmark.py --smoke-test
   ```

6. **Run Full Benchmark (~3000 images):**
   ```cmd
   python scripts\benchmark.py
   ```

## Output
Results are saved in `outputs/` folder, including a JSON summary, a CSV of all predictions, and a CSV of failures.
