#!/usr/bin/env python
"""Prepare BDD100K detection *val* (10k images) as a COCO-format set with COCO category ids.

Source: val-only HF mirror by Hirundo (no login; 575 MB zip, BSD-3 LICENSE.txt copied
from the bdd100k repo), streamed + extracted on the fly:
  https://huggingface.co/datasets/hirundo-io/bdd100k-validation-only
  -> "BDD100K Val from Hirundo.zip": bdd100k/images/100k/val/*.jpg (10000) +
     bdd100k/bdd100k.csv (object_id,image_path,class_name,xmin,ymin,xmax,ymax).
The CSV is a flat export of the BDD100K det-2020 val labels (13 classes, integer-rounded
coordinates, *no* occluded/truncated/crowd attributes).

Output: datasets/bdd100k/annotations/instances_val_cocoids.json, images in
datasets/bdd100k/images/val/, raw CSV kept in datasets/bdd100k/raw/.

Class mapping (BDD -> COCO): pedestrian, rider -> person; car; truck; bus; train;
motorcycle; bicycle; traffic light -> traffic light. Ignored (iscrowd=1, replicated per
category so detections of any class inside are not penalised; see prepare_visdrone.py):
traffic sign (no COCO equivalent; COCO 'stop sign' is a narrow subset), other vehicle,
other person, trailer.

Conventions: image id = 1..N in sorted file-name order, file_name = '<name>.jpg',
bbox = [xmin, ymin, xmax-xmin, ymax-ymin] clipped to the image, area = w*h.

Usage: CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/python scripts/prepare_bdd100k.py [--check-only]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_common as pc  # noqa: E402

URL = ("https://huggingface.co/datasets/hirundo-io/bdd100k-validation-only/resolve/main/"
       "BDD100K%20Val%20from%20Hirundo.zip")
OUT = pc.DATASETS / "bdd100k"
IMG = OUT / "images" / "val"
CSV = OUT / "raw" / "bdd100k_val_hirundo.csv"
EXPECTED_IMAGES = 10000
EXPECTED_ROWS = 186033

B2COCO = {"pedestrian": "person", "rider": "person", "car": "car", "truck": "truck",
          "bus": "bus", "train": "train", "motorcycle": "motorcycle", "bicycle": "bicycle",
          "traffic light": "traffic light"}
IGNORE = {"traffic sign", "other vehicle", "other person", "trailer"}


def download() -> None:
    n = len(list(IMG.glob("*.jpg"))) if IMG.is_dir() else 0
    if n == EXPECTED_IMAGES and CSV.exists():
        print(f"[download] {IMG} already complete ({n} images), skipping")
        return
    pc.ensure_space((pc.http_size(URL) or 0) + 50 * 10**6, "BDD100K val")

    def route(rel: str):
        if rel.startswith("__MACOSX/") or rel.endswith(".DS_Store"):
            return None
        if rel == "bdd100k/bdd100k.csv":
            return CSV
        if rel.startswith("bdd100k/images/100k/val/") and rel.endswith(".jpg"):
            return IMG / Path(rel).name
        return None

    print(f"[download] streaming {URL}")
    print(f"[download] extracted {pc.stream_unzip_url(URL, route)} files")


def convert() -> dict:
    jpgs = sorted(IMG.glob("*.jpg"))
    if len(jpgs) != EXPECTED_IMAGES:
        sys.exit(f"expected {EXPECTED_IMAGES} images, found {len(jpgs)}")
    with open(CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != EXPECTED_ROWS:
        print(f"[convert] WARNING: {len(rows)} CSV rows (expected {EXPECTED_ROWS})")
    by_img = defaultdict(list)
    for r in rows:
        by_img[Path(r["image_path"]).name].append(r)
    unknown = set(by_img) - {p.name for p in jpgs}
    if unknown:
        sys.exit(f"{len(unknown)} annotated images missing, e.g. {sorted(unknown)[:3]}")
    names = list(dict.fromkeys(B2COCO.values()))
    cats = pc.coco_categories(names)
    cid = {c["name"]: c["id"] for c in cats}
    images, anns, st = [], [], Counter()
    for img_id, p in enumerate(jpgs, 1):
        W, H = pc.image_size(p)
        images.append({"id": img_id, "file_name": p.name, "width": W, "height": H})
        for r in by_img.get(p.name, []):
            cls = r["class_name"]
            x0, y0, x1, y1 = (float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
            st["clipped"] += (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1)
            if cx1 <= cx0 or cy1 <= cy0:
                st["dropped_degenerate"] += 1
                continue
            bbox = [cx0, cy0, cx1 - cx0, cy1 - cy0]
            base = {"image_id": img_id, "bbox": bbox, "area": bbox[2] * bbox[3], "bdd_cat": cls}
            if cls in B2COCO:
                anns.append(dict(base, category_id=cid[B2COCO[cls]], iscrowd=0))
            elif cls in IGNORE:
                anns += [dict(base, category_id=c["id"], iscrowd=1) for c in cats]
            else:
                sys.exit(f"unknown BDD class {cls!r}")
            st[cls] += 1
    for i, a in enumerate(anns, 1):
        a["id"] = i
    print(f"[convert] images without boxes: {sum(p.name not in by_img for p in jpgs)}; "
          f"stats: {dict(st)}")
    return {"info": {"description": "BDD100K det val (Hirundo HF mirror of det-2020 labels) with COCO "
                                    "category ids; traffic sign/other vehicle/other person/trailer -> "
                                    "iscrowd=1 replicated per category", "version": "1.0"},
            "licenses": [], "images": images, "annotations": anns, "categories": cats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    ann = OUT / "annotations" / "instances_val_cocoids.json"
    if not args.check_only:
        download()
        pc.write_json(convert(), ann)
    pc.sanity(ann, IMG, EXPECTED_IMAGES)
    print(f"[disk] datasets/bdd100k uses {pc.du(OUT)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
