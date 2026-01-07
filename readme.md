# Retail Product Detection & Checkout System (YOLOv8)

## 📌 Project Overview
This project implements a retail product detection and checkout system based on **YOLOv8**.
The system detects multiple products in an image, counts each SKU, and exports structured checkout results.

## 🚀 Features
- COCO-format dataset conversion to YOLO format
- Training YOLOv8 on RPC (Retail Product Checkout) dataset
- GPU-accelerated training (NVIDIA CUDA)
- Batch inference and CSV export for checkout lists
- Robust handling for large-scale inference (streaming export)

## 🗂 Project Structure
apmcm/
├── ai/                  # training & inference scripts
├── rpc_yolo/            # YOLO-format dataset config
├── scripts/             # shell scripts (optional)
├── requirements.txt
└── README.md
## 📊 Dataset
- Dataset: **Retail Product Checkout (RPC)**
- Format: COCO → YOLO
- Note: Dataset files are **not included** in this repository.

## 🏋️ Training
```bash
python ai/trainrpc.py \
  --data ai/rpc_yolo/rpc.yaml \
  --model yolov8s.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device 0

python ai/export.py \
  --weights runs/detect/rpc_det/weights/best.pt \
  --source ai/rpc_yolo/images/val \
  --out_csv shopping_list_long.csv \
  --out_wide_csv shopping_list_wide.csv
```
⚙️ Environment
	•	OS: Arch Linux
	•	GPU: NVIDIA RTX 5060
	•	Python: 3.13
	•	Framework: PyTorch + Ultralytics YOLOv8

📌 Notes
	•	Training outputs (runs/) and datasets are excluded from GitHub.
	•	This repository focuses on reproducibility and engineering structure.

