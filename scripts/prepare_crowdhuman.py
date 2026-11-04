#!/usr/bin/env python
"""Prepare CrowdHuman *val* (4370 images) as a COCO-format person-detection set.

Source: HF mirror of the official release (same file sizes as the Google-Drive originals),
  https://huggingface.co/datasets/sshao0516/CrowdHuman  (also Funtion/CrowdHuman)
    annotation_val.odgt  (23 MB)   CrowdHuman_val.zip (2.49 GB, Images/<ID>.jpg)
The zip is stream-extracted (no archive on disk) to datasets/crowdhuman/images/val/.

Output: datasets/crowdhuman/annotations/instances_val_cocoids.json, one category
  person (COCO id 1). For every gtbox of the odgt:
    tag == 'person' and extra.ignore != 1  -> full-body box (fbox), iscrowd=0
    tag == 'mask' or extra.ignore == 1     -> fbox, iscrowd=1 (ignore region)
  fbox is the *full-body* box incl. occluded/out-of-frame parts; it is clipped to the
  image (raw box kept in 'fbox_raw', plus 'vbox'/'hbox'); area = clipped w*h.
  image id = 1..N in odgt order, file_name = '<ID>.jpg'.

Deviation from the official protocol: CrowdHuman reports AP/MR^-2/JI on fbox with its own
matcher (no 100-detection cap, ignore handling via IoA); here plain COCOeval is used.

Usage:
  CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/python scripts/prepare_crowdhuman.py
  ... --annotations-only   (download the odgt and print statistics; no images / no JSON)
  ... --check-only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_common as pc  # noqa: E402

HF = "https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main/"
OUT = pc.DATASETS / "crowdhuman"
IMG = OUT / "images" / "val"
ODGT = OUT / "raw" / "annotation_val.odgt"
EXPECTED_IMAGES = 4370


def get_odgt() -> list[dict]:
    if not ODGT.exists():
        import httpx
        ODGT.parent.mkdir(parents=True, exist_ok=True)
        tmp = ODGT.with_suffix(".part")
        with httpx.stream("GET", HF + "annotation_val.odgt", follow_redirects=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for c in r.iter_bytes(1 << 20):
                    f.write(c)
        tmp.rename(ODGT)
    with open(ODGT) as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    if len(recs) != EXPECTED_IMAGES:
        sys.exit(f"odgt has {len(recs)} records, expected {EXPECTED_IMAGES}")
    return recs


def odgt_stats(recs: list[dict]) -> None:
    st = Counter()
    for r in recs:
        for g in r["gtboxes"]:
            ign = g["tag"] != "person" or g.get("extra", {}).get("ignore", 0) == 1
            st[f"{g['tag']}_{'ignore' if ign else 'keep'}"] += 1
    print(f"[odgt] {len(recs)} images, gtbox stats {dict(st)}")


def download() -> None:
    n = len(list(IMG.glob("*.jpg"))) if IMG.is_dir() else 0
    if n == EXPECTED_IMAGES:
        print(f"[download] {IMG} already complete, skipping")
        return
    url = HF + "CrowdHuman_val.zip"
    pc.ensure_space((pc.http_size(url) or 0) + 50 * 10**6, "CrowdHuman val")

    def route(rel: str):
        parts = rel.split("/")
        if len(parts) == 2 and parts[0] == "Images" and parts[1].endswith(".jpg"):
            return IMG / parts[1]
        return None

    print(f"[download] streaming {url}")
    print(f"[download] extracted {pc.stream_unzip_url(url, route)} files")


def convert(recs: list[dict]) -> dict:
    cats = pc.coco_categories(["person"])
    images, anns, st = [], [], Counter()
    for img_id, r in enumerate(recs, 1):
        p = IMG / f"{r['ID']}.jpg"
        if not p.exists():
            sys.exit(f"missing image {p}")
        W, H = pc.image_size(p)
        images.append({"id": img_id, "file_name": p.name, "width": W, "height": H})
        for g in r["gtboxes"]:
            x, y, w, h = (float(v) for v in g["fbox"])
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
            st["clipped"] += (x0, y0, x1, y1) != (x, y, x + w, y + h)
            if x1 <= x0 or y1 <= y0:
                st["dropped_degenerate"] += 1
                continue
            crowd = int(g["tag"] != "person" or g.get("extra", {}).get("ignore", 0) == 1)
            bbox = [x0, y0, x1 - x0, y1 - y0]
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": 1, "bbox": bbox,
                         "area": bbox[2] * bbox[3], "iscrowd": crowd, "tag": g["tag"],
                         "fbox_raw": g["fbox"], "vbox": g.get("vbox"), "hbox": g.get("hbox")})
    print(f"[convert] box stats {dict(st)}")
    return {"info": {"description": "CrowdHuman val, full-body boxes (clipped) as COCO person; "
                                    "mask/ignore -> iscrowd=1", "version": "1.0"},
            "licenses": [], "images": images, "annotations": anns, "categories": cats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations-only", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    ann = OUT / "annotations" / "instances_val_cocoids.json"
    if not args.check_only:
        recs = get_odgt()
        odgt_stats(recs)
        if args.annotations_only:
            return
        download()
        pc.write_json(convert(recs), ann)
    pc.sanity(ann, IMG, EXPECTED_IMAGES)
    print(f"[disk] datasets/crowdhuman uses {pc.du(OUT)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
