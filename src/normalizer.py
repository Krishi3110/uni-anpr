import re

class TextNormalizer:
    def __init__(self):
        pass

    def normalize(self, text):
        # Uppercase
        text = text.upper()
        
        # Remove whitespace and punctuation
        text = re.sub(r'[^A-Z0-9]', '', text)
        
        # Simple rule-based correction for common OCR errors
        # Note: Be careful not to aggressively replace unless we are sure of the position.
        # For a more advanced system, we'd use regex for Indian plates e.g. ^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$
        # and correct characters based on expected type (letter vs digit).
        
        # We will do a basic pass for obvious confusions if we know it's a standard format.
        # But for prototype, we just return the alphanumeric cleaned text.
        
        return text

    def calculate_confidence(self, detector_conf, ocr_conf, text):
        # A simple fusion logic for final confidence
        # Length check
        if len(text) < 4 or len(text) > 10:
            length_penalty = 0.5
        else:
            length_penalty = 1.0
            
        final_conf = (detector_conf * 0.4 + ocr_conf * 0.6) * length_penalty
        
        # Thresholds
        if final_conf > 0.8:
            status = "ACCEPTED"
        elif final_conf > 0.5:
            status = "MANUAL_VERIFICATION_REQUIRED"
        else:
            status = "UNKNOWN"
            
        return final_conf, status
