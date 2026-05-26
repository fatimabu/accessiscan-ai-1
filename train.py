#!/usr/bin/env python3

"""
ACCESSISCAN TRAINER (The Laboratory)
------------------------------------
Purpose:
    Handles the training and fine-tuning of the YOLO-based vision models.
    This module encapsulates the training loop, hyperparameter management,
    and dataset integration for the DSAPT compliance project.

Functionality:
    - Loads training configurations from 'data.yaml'.
    - Configures model architecture and initialization.
    - Manages training cycles, validation, and weight exportation.

Integration:
    This script is intended to be run during the development and iteration phases 
    to produce new model weights for the AccesiScanDetector.
"""

import argparse
import os
import shutil
import wandb
from ultralytics import YOLO



def main():
    # Establish the Project Root
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)  # This forces everything to happen inside your project folder

    # Define the path to your config
    config_path = os.path.join(BASE_DIR, 'configs', 'data.yaml')

    # Initialize the parser ONCE
    parser = argparse.ArgumentParser(description="Train YOLOv8 on AccessiScan dataset")
    
    # Use the absolute path defined above as the default
    parser.add_argument('--data', default=config_path, help='path to dataset yaml')
    parser.add_argument('--model', default='yolov8n.pt', help='base model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--project', default='runs', help='save directory')
    parser.add_argument('--name', default='train', help='run name')
    
    args = parser.parse_args()

    # 4. Proceed with training
    os.makedirs(args.project, exist_ok=True)
    
    print(f"--- AccessiScan-AI: Training Started ---")
    print(f"Model: {args.model} | Data: {args.data} | Resolution: {args.imgsz}")

    model = YOLO(args.model)

    # Start training
    # workers=0 is maintained for Windows OS stability
    results=model.train( 
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        workers=0,
    )

    # This reduces model size and prunes metadata for lightweight mobile deployment
    print("Exporting model to ONNX format for mobile deployment...")
    model.export(format='onnx') 


    # Copy the 'best' brain to a stable path for the ai_engine.py
    src_best = os.path.join(args.project, args.name, 'weights', 'best.pt')
    dest_dir = os.path.join(args.project, 'weights')
    dest_best = os.path.join(dest_dir, 'best.pt')

    if os.path.exists(src_best):
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_best, dest_best)
        print(f"Success: Best weights archived to {dest_best}")
    else:
        print(f"Warning: Training complete but best.pt not found at {src_best}.")

# Returns results for programmatic analytical proof in master test scripts
    return results

if __name__ == '__main__':
    # Initialize Experiment Tracking 
    # This proves training metrics to government stakeholders
    wandb.init(
        project="AccesiScan-AI", 
        name="TRL3-Final-Audit-Build",
        config={
            "epochs": 100,
            "imgsz": 640,
            "architecture": "YOLOv8-Nano"
        }
    )
    training_results=main()
    wandb.finish()
