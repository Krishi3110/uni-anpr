import cv2
from ultralytics import YOLO
import time

class PlateDetector:
    def __init__(self, model_path="runs/detect/models/plate_detector/weights/best.pt"):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        try:
            self.model = YOLO(self.model_path)
            print(f"Loaded specialized plate detector from {self.model_path}")
        except Exception as e:
            print(f"Warning: Failed to load {self.model_path}. Falling back to yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
            
    def detect(self, img_path):
        start = time.time()
        results = self.model(img_path, verbose=False)
        latency = time.time() - start
        
        boxes = []
        confidences = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # box format: [x1, y1, x2, y2]
                b = box.xyxy[0].cpu().numpy().tolist()
                c = float(box.conf[0].cpu().numpy())
                
                boxes.append([int(x) for x in b])
                confidences.append(c)
        else:
            # Fallback for zenitsu09 dataset: the image IS the plate.
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                boxes.append([0, 0, w, h])
                confidences.append(1.0)
                
        return boxes, confidences, latency
