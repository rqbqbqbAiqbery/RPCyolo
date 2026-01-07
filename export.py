from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(src: Path) -> list[Path]:
    if src.is_file():
        return [src]
    imgs = []
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            imgs.append(p)
    return sorted(imgs)


def cls_name(names, cid: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cid, cid))
    if isinstance(names, (list, tuple)):
        return str(names[cid]) if 0 <= cid < len(names) else str(cid)
    return str(cid)


def load_price_map(price_csv: Path) -> dict[str, float]:
    """
    price_csv: columns can be [sku, price] OR [name, price]
    """
    df = pd.read_csv(price_csv)
    cols = {c.lower(): c for c in df.columns}
    name_col = cols.get("sku") or cols.get("name")
    price_col = cols.get("price")
    if not name_col or not price_col:
        raise ValueError("price_csv 必须包含列：sku(or name) 和 price")
    mp = {}
    for _, r in df.iterrows():
        mp[str(r[name_col])] = float(r[price_col])
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str, required=True, help="image file or folder")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--save_dir", type=str, default="runs/predict", help="where to save annotated images")
    ap.add_argument("--out_csv", type=str, default="shopping_list_long.csv")
    ap.add_argument("--out_wide_csv", type=str, default="shopping_list_wide.csv")
    ap.add_argument("--price_csv", type=str, default="", help="Optional: csv with columns sku(or name),price")
    args = ap.parse_args()

    model = YOLO(args.weights)
    names = model.names

    src = Path(args.source).expanduser().resolve()
    images = list_images(src)
    if not images:
        raise SystemExit(f"[ERR] No images found in: {src}")

    price_map = None
    if args.price_csv.strip():
        price_map = load_price_map(Path(args.price_csv).expanduser().resolve())

    # 推理（save=True 会保存画框结果）
    results = model.predict(
        source=[str(p) for p in images],
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=args.save_dir,
        name="exp",
        verbose=False,
        stream=True,  # 不占太多内存
    )

    long_rows = []
    wide_rows = []

    total_counter = Counter()

    for r in results:
        img_path = Path(r.path)
        img_name = img_path.name

        counter = Counter()
        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
            counter.update(cls_ids)

        total_counter.update(counter)

        # long format
        for cid, cnt in counter.items():
            sku = cls_name(names, cid)
            unit_price = price_map.get(sku) if price_map else None
            subtotal = (unit_price * cnt) if unit_price is not None else None
            long_rows.append({
                "image": img_name,
                "cls_id": cid,
                "sku": sku,
                "count": cnt,
                "unit_price": unit_price,
                "subtotal": subtotal,
            })

        # wide format row
        row = {"image": img_name}
        for cid, cnt in counter.items():
            row[cls_name(names, cid)] = cnt
        wide_rows.append(row)

    # add total summary (long)
    if long_rows:
        df_long = pd.DataFrame(long_rows)
    else:
        df_long = pd.DataFrame(columns=["image", "cls_id", "sku", "count", "unit_price", "subtotal"])

    # total summary row for long
    for cid, cnt in total_counter.items():
        sku = cls_name(names, cid)
        unit_price = price_map.get(sku) if price_map else None
        subtotal = (unit_price * cnt) if unit_price is not None else None
        df_long = pd.concat([df_long, pd.DataFrame([{
            "image": "__TOTAL__",
            "cls_id": cid,
            "sku": sku,
            "count": cnt,
            "unit_price": unit_price,
            "subtotal": subtotal,
        }])], ignore_index=True)

    # wide csv
    df_wide = pd.DataFrame(wide_rows).fillna(0)
    # total row
    total_row = {"image": "__TOTAL__"}
    for cid, cnt in total_counter.items():
        total_row[cls_name(names, cid)] = cnt
    df_wide = pd.concat([df_wide, pd.DataFrame([total_row])], ignore_index=True).fillna(0)

    # 整型化 wide 的数值列
    for c in df_wide.columns:
        if c != "image":
            df_wide[c] = df_wide[c].astype(int)

    out_csv = Path(args.out_csv).expanduser().resolve()
    out_wide = Path(args.out_wide_csv).expanduser().resolve()
    df_long.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_wide.to_csv(out_wide, index=False, encoding="utf-8-sig")

    print(f"[DONE] annotated images saved to: {Path(args.save_dir).resolve()}/exp")
    print(f"[DONE] long  csv: {out_csv}")
    print(f"[DONE] wide  csv: {out_wide}")


if __name__ == "__main__":
    main()