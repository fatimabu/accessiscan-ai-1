# YOLO AI Detection Project

## Project Overview

This project uses YOLO (Ultralytics) for accessibility sign detection as part of the AccessiScan system.

Current implementation:

* YOLO object detection
* Accessibility sign detection
* FastAPI-ready structure
* Train/validation dataset setup

---

# Project Structure

```text
ai-detection/
├── data/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── datasets/
│   └── accessibility.yaml
├── notebooks/
├── runs/
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── venv/
├── .gitignore
├── README.md
└── requirements.txt
```

---

# Create Virtual Environment

```bash
python3 -m venv venv
```

# Activate Virtual Environment

```bash
source venv/bin/activate
```

---

# Install Required Packages

```bash
pip install ultralytics opencv-python fastapi uvicorn python-multipart pillow numpy matplotlib
```

---

# Save Requirements

```bash
pip freeze > requirements.txt
```

---

# Dataset Structure

Images and labels must follow this structure:

```text
data/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each image must have a matching label file.

Example:

```text
image1.jpg
image1.txt
```

---

# YOLO Dataset Configuration

Create:

```text
datasets/accessibility.yaml
```

Example:

```yaml
path: ./data

train: images/train
val: images/val

names:
  0: accessible_sign
```

---

# Train YOLO Model

```bash
yolo detect train data=datasets/accessibility.yaml model=yolo11n.pt epochs=50 imgsz=640
```

---

# Run Prediction

```bash
yolo detect predict model=../runs/detect/train/weights/best.pt source=data/images/val conf=0.1
```

---

# Training Outputs

YOLO training results are saved in:

```text
runs/detect/train/
```

Important files:

```text
best.pt
results.png
confusion_matrix.png
```

---

# Git Commands

## Check Git Status

```bash
git status
```

## Add Files

```bash
git add .
```

## Commit Changes

```bash
git commit -m "Initial YOLO AI detection setup"
```

## Push Changes

```bash
git push
```

---

# Notes

* The `venv/` folder is ignored using `.gitignore`
* Training outputs inside `runs/` are ignored
* Model weights (`*.pt`) are ignored
* Dataset images and labels are ignored from Git tracking

---

# Future Improvements

* Add mor
