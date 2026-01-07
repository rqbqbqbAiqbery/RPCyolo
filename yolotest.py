from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return x1, y1, x2, y2


def read_labels(lbl_path: Path):
    if not lbl_path.exists():
        return []
    txt = lbl_path.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    boxes = []
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        boxes.append((cls, xc, yc, w, h))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", type=str, required=True, help="Path to rpc_yolo folder")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=20, help="How many images to visualize")
    ap.add_argument("--out_dir", type=str, default="vis_samples")
    args = ap.parse_args()

    root = Path(args.yolo_root).expanduser().resolve()
    img_dir = root / "images" / args.split
    lbl_dir = root / "labels" / args.split
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
    if not imgs:
        raise SystemExit(f"[ERR] No images found in {img_dir}")

    random.shuffle(imgs)
    imgs = imgs[: min(args.n, len(imgs))]

    for p in imgs:
        rel = p.relative_to(img_dir)
        lbl = lbl_dir / rel.with_suffix(".txt")

        im = cv2.imread(str(p))
        if im is None:
            continue
        H, W = im.shape[:2]

        boxes = read_labels(lbl)
        for cls, xc, yc, w, h in boxes:
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, str(cls), (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        save_path = out_dir / f"{args.split}_{rel.as_posix().replace('/', '_')}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), im)

    print(f"[DONE] Saved {len(imgs)} visualizations to: {out_dir}")


if __name__ == "__main__":
    main()