#!/usr/bin/env python
"""Prepare COCO-O (Mao et al., ICCV 2023; github.com/alibaba/easyrobust/tree/main/benchmarks/coco_o).

Source: the authors' Google Drive file ood_coco.zip (1.1 GB, id 1aBfIJN0zo_i80Hv4p7Ch7M8pRzO37qbq),
fetched through Drive's "download anyway" form (no gdown needed) and stream-extracted to
  datasets/coco_o/<domain>/{val2017/*.jpg, annotations/instances_val2017.json}   (originals)
for the 6 domains cartoon, handmake, painting, sketch, tattoo, weather.

Outputs (datasets/coco_o/annotations/, image root = datasets/coco_o/):
  instances_<domain>.json   original annotations, file_name rewritten to
                            '<domain>/val2017/<file>' so all files share one image root;
                            image/annotation ids unchanged; categories = the 80 COCO ones.
  instances_all.json        the 6 domains merged; image id = (domain_index+1)*10**7 + id,
                            annotation id = (domain_index+1)*10**7 + id; images carry 'domain'.
Annotations are already COCO-format with COCO ids; nothing else is changed.

Usage: CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/python scripts/prepare_coco_o.py [--check-only]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_common as pc  # noqa: E402

DRIVE_ID = "1aBfIJN0zo_i80Hv4p7Ch7M8pRzO37qbq"
OUT = pc.DATASETS / "coco_o"
DOMAINS = ["cartoon", "handmake", "painting", "sketch", "tattoo", "weather"]
EXPECTED_TOTAL_IMAGES = 6782  # paper: 6,782 images, 26,624 boxes
EXPECTED_TOTAL_BOXES = 26624
SIZE = 1.15 * 10**9


def drive_url() -> str:
    import httpx
    base = "https://drive.usercontent.google.com/download"
    r = httpx.get(base, params={"id": DRIVE_ID, "export": "download"}, follow_redirects=True, timeout=60)
    r.raise_for_status()
    if "text/html" not in r.headers.get("content-type", ""):
        return str(r.url)
    fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', r.text))
    if "id" not in fields:
        sys.exit("could not parse Google Drive confirm form (quota exceeded?)")
    return str(httpx.URL(base, params=fields))


def complete() -> bool:
    for d in DOMAINS:
        ann = OUT / d / "annotations" / "instances_val2017.json"
        if not ann.exists():
            return False
        with open(ann) as f:
            n = len(json.load(f)["images"])
        if len(list((OUT / d / "val2017").glob("*"))) < n:
            return False
    return True


def download() -> None:
    if complete():
        print("[download] all 6 domains present, skipping")
        return
    pc.ensure_space(int(SIZE), "COCO-O")
    url = drive_url()

    def route(rel: str):
        parts = rel.split("/")
        if len(parts) != 4 or parts[0] != "ood_coco" or parts[1] not in DOMAINS:
            return None
        if parts[2] == "val2017" and not parts[3].startswith("."):
            return OUT / parts[1] / "val2017" / parts[3]
        if parts[2] == "annotations" and parts[3] == "instances_val2017.json":
            return OUT / parts[1] / "annotations" / parts[3]
        return None

    print(f"[download] streaming ood_coco.zip from Google Drive")
    print(f"[download] extracted {pc.stream_unzip_url(url, route)} files")


def convert() -> dict[str, Path]:
    outs, merged = {}, None
    for k, d in enumerate(DOMAINS):
        with open(OUT / d / "annotations" / "instances_val2017.json") as f:
            data = json.load(f)
        for im in data["images"]:
            im["file_name"] = f"{d}/val2017/{Path(im['file_name']).name}"
        on_disk = {p.name for p in (OUT / d / "val2017").iterdir()}
        extra = on_disk - {Path(im["file_name"]).name for im in data["images"]}
        print(f"[convert] {d}: {len(data['images'])} images, {len(data['annotations'])} anns, "
              f"{len(extra)} unreferenced files on disk {sorted(extra)[:3]}")
        data.setdefault("info", {})["description"] = f"COCO-O {d} (file_name relative to datasets/coco_o)"
        outs[d] = OUT / "annotations" / f"instances_{d}.json"
        pc.write_json(data, outs[d])
        off = (k + 1) * 10**7
        if merged is None:
            merged = {"info": {"description": "COCO-O, all 6 domains merged; id = (domain_idx+1)*1e7 + id",
                               "version": "1.0"}, "licenses": data.get("licenses", []),
                      "images": [], "annotations": [], "categories": data["categories"]}
        elif data["categories"] != merged["categories"]:
            sys.exit(f"category list of {d} differs")
        merged["images"] += [dict(im, id=im["id"] + off, domain=d) for im in data["images"]]
        merged["annotations"] += [dict(a, id=a["id"] + off, image_id=a["image_id"] + off)
                                  for a in data["annotations"]]
    outs["all"] = OUT / "annotations" / "instances_all.json"
    pc.write_json(merged, outs["all"])
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.check_only:
        download()
        outs = convert()
    else:
        outs = {d: OUT / "annotations" / f"instances_{d}.json" for d in DOMAINS + ["all"]}
    res = {k: pc.sanity(p, OUT, check_sizes=True) for k, p in outs.items()}
    tot_i, tot_b = res["all"]["images"], res["all"]["boxes"] + res["all"]["ignore"]
    print(f"\n[summary] total images {tot_i} (paper {EXPECTED_TOTAL_IMAGES}), "
          f"total boxes {tot_b} (paper {EXPECTED_TOTAL_BOXES})")
    for k, r in res.items():
        print(f"[summary] {k:9s} imgs={r['images']:5d} boxes={r['boxes']:6d} crowd={r['ignore']:4d} "
              f"S/M/L={r['small']:.3f}/{r['medium']:.3f}/{r['large']:.3f}")
    print(f"[disk] datasets/coco_o uses {pc.du(OUT)/1e9:.2f} GB")


if __name__ == "__main__":
    main()
