import sys
import platform
import subprocess

def check_env():
    print("=== ANPR Environment Diagnostic ===")
    print(f"Python version: {sys.version}")
    print(f"OS: {platform.system()} {platform.release()}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed")
        
    try:
        import paddle
        print(f"Paddle version: {paddle.__version__}")
        print(f"Paddle CUDA compiled: {paddle.is_compiled_with_cuda()}")
        print(f"Paddle device: {paddle.get_device()}")
    except ImportError:
        print("PaddlePaddle not installed")
        
    try:
        from paddleocr import PaddleOCR
        print("PaddleOCR available")
    except ImportError:
        print("PaddleOCR not installed")

    try:
        import cv2
        print("OpenCV available")
    except ImportError:
        print("OpenCV not installed")

    try:
        import ultralytics
        print("Ultralytics available")
    except ImportError:
        print("Ultralytics not installed")

    # Check Ollama and qwen3:14b
    try:
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if res.returncode == 0:
            print("Ollama available")
            if "qwen3:14b" in res.stdout:
                print("Model qwen3:14b is available locally")
            else:
                print("Model qwen3:14b NOT found in ollama list")
        else:
            print("Ollama returned an error")
    except FileNotFoundError:
        print("Ollama is not installed or not in PATH")

if __name__ == "__main__":
    check_env()
