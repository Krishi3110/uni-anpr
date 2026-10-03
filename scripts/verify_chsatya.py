import os
import cv2
import random

def verify_bboxes():
    img_dir = "datasets/chsatya_yolo/images/train"
    lbl_dir = "datasets/chsatya_yolo/labels/train"
    out_dir = "outputs/visualizations"
    os.makedirs(out_dir, exist_ok=True)
    
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    random.seed(42)
    sample_images = random.sample(images, 5)
    
    for img_name in sample_images:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.txt'))
        
        im = cv2.imread(img_path)
        if im is None: continue
        h, w = im.shape[:2]
        
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id, xc, yc, bw, bh = map(float, parts)
                xmin = int((xc - bw / 2) * w)
                ymin = int((yc - bh / 2) * h)
                xmax = int((xc + bw / 2) * w)
                ymax = int((yc + bh / 2) * h)
                
                cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(im, "plate", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
        out_path = os.path.join(out_dir, f"verified_{img_name}")
        cv2.imwrite(out_path, im)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    verify_bboxes()
