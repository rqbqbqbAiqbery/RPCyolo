from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from tqdm import tqdm


def is_coco_like_json(p: Path) -> bool:
    """Heuristically check whether a JSON file looks like COCO annotation."""
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return (
            isinstance(data, dict)
            and isinstance(data.get("images"), list)
            and isinstance(data.get("annotations"), list)
            and isinstance(data.get("categories"), list)
        )
    except Exception:
        return False


def find_coco_jsons(root: Path) -> List[Path]:
    """Find COCO-like json files under root (shallow+deep)."""
    cands = []
    for p in root.rglob("*.json"):
        # skip hidden/metadata-like
        if p.name.startswith("."):
            continue
        if is_coco_like_json(p):
            cands.append(p)
    return cands


def guess_split(p: Path) -> str:
    """Guess split name from filename or parent folder."""
    name = p.name.lower()
    parent = p.parent.name.lower()

    def _match(s: str) -> bool:
        return s in name or s in parent

    if _match("train"):
        return "train"
    if _match("val") or _match("valid") or _match("validation"):
        return "val"
    if _match("test"):
        return "test"
    return "unknown"


def safe_link_or_copy(src: Path, dst: Path, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if use_symlink:
        try:
            os.symlink(src, dst)
            return
        except OSError:
            # fallback to copy
            pass

    shutil.copy2(src, dst)


def build_cat_mapping(categories: List[dict]) -> Tuple[Dict[int, int], List[str]]:
    """
    Build mapping: category_id (COCO) -> class_index (0..nc-1)
    and a names list where names[i] corresponds to class_index i.
    """
    cats_sorted = sorted(categories, key=lambda x: int(x.get("id", 0)))
    cat_id_to_idx: Dict[int, int] = {}
    names: List[str] = []
    for idx, c in enumerate(cats_sorted):
        cid = int(c["id"])
        cat_id_to_idx[cid] = idx
        nm = c.get("name")
        names.append(str(nm if nm is not None else cid))
    return cat_id_to_idx, names


def resolve_image_path(
    root: Path,
    file_name: str,
    candidate_dirs: List[Path],
) -> Optional[Path]:
    """
    Try to resolve an image path from COCO's file_name.
    Handles both relative paths and plain basenames.
    """
    fn = Path(file_name)

    # 1) root / file_name
    p1 = (root / fn).resolve()
    if p1.exists():
        return p1

    # 2) try candidate dirs joined with file_name
    for d in candidate_dirs:
        p2 = (d / fn).resolve()
        if p2.exists():
            return p2

    # 3) try candidate dirs + basename
    base = fn.name
    for d in candidate_dirs:
        p3 = (d / base).resolve()
        if p3.exists():
            return p3

    return None


def coco_to_yolo_bbox(bbox_xywh: List[float], w: int, h: int) -> Tuple[float, float, float, float]:
    """
    COCO bbox: [x_min, y_min, width, height] in pixels
    YOLO bbox: x_center, y_center, width, height normalized to [0,1]
    """
    x, y, bw, bh = bbox_xywh
    xc = x + bw / 2.0
    yc = y + bh / 2.0
    return xc / w, yc / h, bw / w, bh / h


def dump_yaml(out_yaml: Path, out_root: Path, names: List[str]) -> None:
    import yaml

    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with out_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc_root", type=str, required=True, help="Path to the downloaded RPC dataset root folder")
    ap.add_argument("--out", type=str, default="rpc_yolo", help="Output folder for YOLO-format dataset")
    ap.add_argument("--copy", action="store_true", help="Copy images instead of symlink (default: symlink)")
    ap.add_argument("--max_images", type=int, default=0, help="Limit images per split (0 means no limit)")
    args = ap.parse_args()

    rpc_root = Path(args.rpc_root).expanduser().resolve()
    out_root = Path(args.out).expanduser().resolve()
    use_symlink = not args.copy

    coco_jsons = find_coco_jsons(rpc_root)
    if not coco_jsons:
        raise SystemExit(f"[ERR] No COCO-like JSON found under: {rpc_root}")

    # pick one json per split (if multiple, choose the largest by file size)
    split_to_json: Dict[str, Path] = {}
    for p in coco_jsons:
        sp = guess_split(p)
        if sp == "unknown":
            continue
        if sp not in split_to_json:
            split_to_json[sp] = p
        else:
            if p.stat().st_size > split_to_json[sp].stat().st_size:
                split_to_json[sp] = p

    # if still missing splits, try to keep unknown as fallback
    if "train" not in split_to_json and "val" not in split_to_json and "test" not in split_to_json:
        # take the largest json as train
        p = max(coco_jsons, key=lambda x: x.stat().st_size)
        split_to_json["train"] = p

    print("[INFO] Using COCO JSON files:")
    for k, v in split_to_json.items():
        print(f"  - {k}: {v}")

    # candidate image dirs to resolve file paths
    candidate_dirs = [rpc_root]
    for d in rpc_root.iterdir():
        if d.is_dir():
            candidate_dirs.append(d)
            # common nested
            for sub in ["images", "imgs", "JPEGImages"]:
                if (d / sub).is_dir():
                    candidate_dirs.append(d / sub)

    # build names mapping from first available split
    first_json = next(iter(split_to_json.values()))
    with first_json.open("r", encoding="utf-8") as f:
        first = json.load(f)
    cat_id_to_idx, names = build_cat_mapping(first["categories"])
    print(f"[INFO] Categories (nc) = {len(names)}")

    # convert each split
    for split, json_path in split_to_json.items():
        with json_path.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        # images: id -> (file_name, width, height)
        images_by_id: Dict[int, dict] = {}
        for im in coco["images"]:
            images_by_id[int(im["id"])] = im

        # annotations grouped by image_id
        anns_by_image: Dict[int, List[dict]] = defaultdict(list)
        for ann in coco["annotations"]:
            if int(ann.get("iscrowd", 0)) == 1:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            # skip invalid bbox
            if bbox[2] <= 1 or bbox[3] <= 1:
                continue
            anns_by_image[int(ann["image_id"])].append(ann)

        img_items = list(images_by_id.items())
        if args.max_images and args.max_images > 0 and len(img_items) > args.max_images:
            img_items = img_items[: args.max_images]

        for image_id, im in tqdm(img_items, desc=f"Converting {split}", unit="img"):
            file_name = im.get("file_name")
            if not file_name:
                continue
            w = int(im.get("width", 0))
            h = int(im.get("height", 0))

            src_img = resolve_image_path(rpc_root, str(file_name), candidate_dirs)
            if src_img is None or not src_img.exists():
                # skip if cannot find
                continue

            # keep relative path (avoid absolute)
            rel = Path(file_name)
            if rel.is_absolute():
                rel = Path(rel.name)

            dst_img = out_root / "images" / split / rel
            dst_lbl = out_root / "labels" / split / rel.with_suffix(".txt")

            safe_link_or_copy(src_img, dst_img, use_symlink=use_symlink)

            # write labels
            lines: List[str] = []
            for ann in anns_by_image.get(image_id, []):
                cid = int(ann["category_id"])
                if cid not in cat_id_to_idx:
                    continue
                cls = cat_id_to_idx[cid]
                bbox = ann["bbox"]
                if w <= 0 or h <= 0:
                    # if width/height missing, skip for safety
                    continue
                xc, yc, bw, bh = coco_to_yolo_bbox(bbox, w=w, h=h)
                # clamp to [0,1]
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            with dst_lbl.open("w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    # write yaml
    out_yaml = out_root / "rpc.yaml"
    dump_yaml(out_yaml, out_root, names)
    print(f"[DONE] YOLO dataset written to: {out_root}")
    print(f"[DONE] Dataset yaml: {out_yaml}")


if __name__ == "__main__":
    main()