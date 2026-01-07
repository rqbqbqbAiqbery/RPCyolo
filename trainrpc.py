from __future__ import annotations

import argparse
import os

from ultralytics import YOLO


def auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to rpc.yaml")
    ap.add_argument("--model", type=str, default="yolov8s.pt",
                    help="Pretrained weights, e.g. yolov8n.pt / yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", type=str, default="", help="mps / cpu / 0 (cuda)")
    ap.add_argument("--workers", type=int, default=0, help="macOS 建议 0，避免 dataloader 问题")
    ap.add_argument("--name", type=str, default="rpc_det")
    ap.add_argument("--project", type=str, default="runs/detect")
    args = ap.parse_args()

    # MPS 有时会遇到不支持的算子，允许回退 CPU（更稳）
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    device = args.device.strip() or auto_device()
    print(f"[INFO] device = {device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=20,
        close_mosaic=10,
    )

    print("[DONE] Training finished. Check:")
    print(f"  {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()