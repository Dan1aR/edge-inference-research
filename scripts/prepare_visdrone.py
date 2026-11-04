#!/usr/bin/env python
"""Prepare VisDrone2019-DET (val / train) as COCO-format detection sets.

Source: Ultralytics mirror of the official VisDrone2019-DET zips
  https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{val,train}.zip
streamed and extracted on the fly (no archive on disk) to
  datasets/visdrone/VisDrone2019-DET-<split>/{images,annotations}/

Outputs (datasets/visdrone/annotations/):
  instances_<split>_cocoids.json  COCO category ids, for zero-shot eval of COCO-trained models
      pedestrian, people -> person(1); bicycle -> bicycle(2); car, van -> car(3);
      motor -> motorcycle(4); bus -> bus(6); truck -> truck(8).
  instances_<split>_native.json   the 10 native VisDrone classes, ids 1..10, for fine-tuning.

Ignore handling (both files): ignored regions (class 0), 'others' (11), any box with
score==0 and -- in the cocoids file only -- the unmapped tricycle(7)/awning-tricycle(8)
become iscrowd=1 boxes. Because COCOeval matches crowd GT only within the same category,
every ignore box is replicated once per category of the file (same bbox, iscrowd=1), so a
detection of *any* class that falls inside it is neither a TP nor an FP (COCO crowd IoU =
intersection / detection area, i.e. the VisDrone "inside ignored region" semantics).
Each annotation keeps the source class in `visdrone_cat` plus `truncation`/`occlusion`.

Conventions (as scripts/prepare_voc07_coco.py): image id = 1..N in sorted file-name order,
file_name = original JPEG name (relative to the images/ root), bbox = [x,y,w,h] clipped to
the image (VisDrone boxes are 0-based pixel boxes), area = w*h; degenerate boxes dropped.

Usage:
  CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/python scripts/prepare_visdrone.py --split val
  ... --split train   |   --check-only
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_common as pc  # noqa: E402

URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{}.zip"
EXPECTED = {"val": 548, "train": 6471, "test-dev": 1610}
OUT = pc.DATASETS / "visdrone"

NATIVE = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
          "awning-tricycle", "bus", "motor"]  # VisDrone ids 1..10
V2COCO = {1: "person", 2: "person", 3: "bicycle", 4: "car", 5: "car", 6: "truck",
          9: "bus", 10: "motorcycle"}


def download(split: str) -> None:
    root = OUT / f"VisDrone2019-DET-{split}"
    n_exp = EXPECTED[split]
    n_img = len(list((root / "images").glob("*.jpg"))) if (root / "images").is_dir() else 0
    n_txt = len(list((root / "annotations").glob("*.txt"))) if (root / "annotations").is_dir() else 0
    if n_img == n_exp and n_txt == n_exp:
        print(f"[download] {root} already complete ({n_img} images), skipping")
        return
    url = URL.format(split)
    size = pc.http_size(url) or 0
    pc.ensure_space(size + 50 * 10**6, f"VisDrone {split}")
    prefix = f"VisDrone2019-DET-{split}/"

    def route(rel: str):
        if not rel.startswith(prefix):
            return None
        sub = rel[len(prefix):]
        if (sub.startswith("images/") and sub.endswith(".jpg")) or \
           (sub.startswith("annotations/") and sub.endswith(".txt")):
            return root / sub
        return None

    print(f"[download] streaming {url} ({size/1e6:.0f} MB)")
    n = pc.stream_unzip_url(url, route)
    print(f"[download] extracted {n} files -> {root}")


def parse(txt: Path) -> list[list[int]]:
    rows = []
    for ln in txt.read_text().splitlines():
        v = [x for x in ln.strip().split(",") if x != ""]
        if len(v) >= 6:
            rows.append([int(float(x)) for x in v[:8]] + [0] * (8 - len(v[:8])))
    return rows


def convert(split: str) -> tuple[dict, dict]:
    root = OUT / f"VisDrone2019-DET-{split}"
    jpgs = sorted((root / "images").glob("*.jpg"))
    if len(jpgs) != EXPECTED[split]:
        sys.exit(f"expected {EXPECTED[split]} images, found {len(jpgs)}")
    coco_cats = pc.coco_categories(sorted(set(V2COCO.values()), key=list(V2COCO.values()).index))
    coco_id = {c["name"]: c["id"] for c in coco_cats}
    native_cats = [{"id": i + 1, "name": n, "supercategory": "visdrone"} for i, n in enumerate(NATIVE)]
    out = {k: {"info": {"description": f"VisDrone2019-DET-{split} ({k} categories; ignore regions,"
                                      " 'others', score==0" + (", tricycles" if k == "cocoids" else "")
                                      + " -> iscrowd=1 replicated per category)", "version": "1.0"},
               "licenses": [], "images": [], "annotations": [],
               "categories": coco_cats if k == "cocoids" else native_cats}
           for k in ("cocoids", "native")}
    stats = Counter()
    for img_id, p in enumerate(jpgs, 1):
        w_img, h_img = pc.image_size(p)
        im = {"id": img_id, "file_name": p.name, "width": w_img, "height": h_img}
        for d in out.values():
            d["images"].append(im)
        txt = root / "annotations" / (p.stem + ".txt")
        if not txt.exists():
            sys.exit(f"missing annotation {txt}")
        for x, y, w, h, score, cat, trunc, occ in parse(txt):
            x0, y0 = max(x, 0), max(y, 0)
            x1, y1 = min(x + w, w_img), min(y + h, h_img)
            stats["raw"] += 1
            if (x0, y0, x1, y1) != (x, y, x + w, y + h):
                stats["clipped"] += 1
            if x1 <= x0 or y1 <= y0:
                stats["dropped_degenerate"] += 1
                continue
            stats[f"cat{cat}_score{int(score > 0)}"] += 1
            bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
            base = {"image_id": img_id, "bbox": bbox, "area": bbox[2] * bbox[3],
                    "visdrone_cat": cat, "truncation": trunc, "occlusion": occ}
            ignore = cat in (0, 11) or score == 0
            # native
            d = out["native"]
            if ignore:
                d["annotations"] += [dict(base, category_id=c["id"], iscrowd=1) for c in native_cats]
            else:
                d["annotations"].append(dict(base, category_id=cat, iscrowd=0))
            # cocoids
            d = out["cocoids"]
            if ignore or cat not in V2COCO:
                d["annotations"] += [dict(base, category_id=c["id"], iscrowd=1) for c in coco_cats]
            else:
                d["annotations"].append(dict(base, category_id=coco_id[V2COCO[cat]], iscrowd=0))
    for d in out.values():
        for i, a in enumerate(d["annotations"], 1):
            a["id"] = i
    print(f"[convert] raw-box stats: {dict(sorted(stats.items()))}")
    return out["cocoids"], out["native"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(EXPECTED), default="val")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    ann = {k: OUT / "annotations" / f"instances_{args.split}_{k}.json" for k in ("cocoids", "native")}
    if not args.check_only:
        download(args.split)
        cocoids, native = convert(args.split)
        pc.write_json(cocoids, ann["cocoids"])
        pc.write_json(native, ann["native"])
    img_root = OUT / f"VisDrone2019-DET-{args.split}" / "images"
    for k, p in ann.items():
        pc.sanity(p, img_root, EXPECTED[args.split], check_sizes=(k == "cocoids"))
    print(f"[disk] datasets/visdrone uses {pc.du(OUT)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
