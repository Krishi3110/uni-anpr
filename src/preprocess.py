import cv2
import numpy as np

class Preprocessor:
    def __init__(self):
        pass

    def crop(self, img, bbox):
        x1, y1, x2, y2 = bbox
        # Add a small margin if possible
        h, w = img.shape[:2]
        margin = 2
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        return img[y1:y2, x1:x2]

    def process(self, img):
        if img is None or img.size == 0:
            return img
        
        # PaddleOCR does its own optimized preprocessing (resizing to height 48, etc)
        # Applying our own resize and CLAHE seems to severely degrade PaddleOCR's internal recognizer.
        return img
