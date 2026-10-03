import argparse
import os
from ultralytics import YOLO

def train_yolo(data_yaml, model_name="yolov8n.pt", epochs=10, imgsz=640, batch=16, seed=42):
    print(f"Training {model_name} on {data_yaml}")
    print(f"Epochs: {epochs}, Imgsz: {imgsz}, Batch: {batch}, Seed: {seed}")
    
    # Load a model
    model = YOLO(model_name)  # load a pretrained model
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        seed=seed,
        device=0,  # GPU 0
        project="models",
        name="plate_detector",
        exist_ok=True,
        plots=True
    )
    print("Training complete. Best weights saved in models/plate_detector/weights/best.pt")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="datasets/yolo/data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    train_yolo(args.data, args.model, args.epochs, args.imgsz, args.batch, args.seed)
