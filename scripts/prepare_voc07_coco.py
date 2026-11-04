#!/usr/bin/env python
"""Prepare Pascal VOC 2007 *test* as a COCO-format detection set with COCO category ids.

Purpose: zero-shot evaluation of a COCO-trained detector (e.g. hustvl/yolos-tiny) on
VOC07 test with pycocotools. Only the 20 VOC classes are listed as categories, but each
carries the COCO `id`, `name` and `supercategory` taken from COCO instances_val2017.json,
so detector outputs (COCO label ids) can be fed to COCOeval unchanged. Detections of
COCO classes that have no VOC counterpart are simply ignored by COCOeval (their
category_id is not in the GT categories), as long as `params.catIds` is left at default.

Conventions:
  * image id   = numeric VOC id (000001.jpg -> 1); file_name = original JPEG name.
  * bbox       = [xmin-1, ymin-1, xmax-xmin+1, ymax-ymin+1] (VOC is 1-based inclusive).
  * area       = w*h; iscrowd = 1 for VOC `difficult` objects (pycocotools then ignores
                 them: matching them is neither a TP nor an FP), else 0.

Steps (all CPU-only; no GPU, no torch):
  1. download VOCtest_06-Nov-2007.tar (mirrors tried in order, MD5 checked),
  2. stream the archive: JPEGs are written flat to <out>/images/, XMLs are parsed in
     memory (no extracted copy of the devkit is kept on disk),
  3. write <out>/annotations/instances_voc07test_cocoids.json,
  4. run sanity checks (counts, image sizes, GT-as-detections COCOeval AP == 1),
  5. delete the archive (unless --keep-archive).

Usage:
  CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/python scripts/prepare_voc07_coco.py
  CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/python scripts/prepare_voc07_coco.py --check-only
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[1]

ARCHIVE_TAR = "VOCtest_06-Nov-2007.tar"
TAR_MD5 = "b6e924de25625d8de591ea690078ad9f"  # official VOC2007 test tar (same as torchvision)
TAR_SIZE = 451_020_800
MIRRORS = [
    # (url, archive file name, expected md5 or None)
    ("http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar", ARCHIVE_TAR, TAR_MD5),
    ("https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar", ARCHIVE_TAR, TAR_MD5),
    ("https://github.com/ultralytics/assets/releases/download/v0.0.0/VOCtest_06-Nov-2007.zip",
     "VOCtest_06-Nov-2007.zip", None),
]

# VOC name -> COCO name
VOC2COCO = {
    "aeroplane": "airplane", "bicycle": "bicycle", "bird": "bird", "boat": "boat",
    "bottle": "bottle", "bus": "bus", "car": "car", "cat": "cat", "chair": "chair",
    "cow": "cow", "diningtable": "dining table", "dog": "dog", "horse": "horse",
    "motorbike": "motorcycle", "person": "person", "pottedplant": "potted plant",
    "sheep": "sheep", "sofa": "couch", "train": "train", "tvmonitor": "tv",
}

# Official VOC2007 test totals.
EXPECTED_IMAGES = 4952
EXPECTED_OBJECTS = 14976
EXPECTED_NONDIFFICULT = 12032


# --------------------------------------------------------------------------- download
def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dl_dir: Path, timeout: float) -> tuple[Path, str]:
    dl_dir.mkdir(parents=True, exist_ok=True)
    # Reuse an already downloaded archive if it is valid.
    for url, name, md5 in MIRRORS:
        p = dl_dir / name
        if p.exists() and (md5 is None or md5sum(p) == md5):
            print(f"[download] reusing {p}")
            return p, f"(cached) {url}"
    for url, name, md5 in MIRRORS:
        p = dl_dir / name
        tmp = p.with_suffix(p.suffix + ".part")
        print(f"[download] trying {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
        except Exception as e:  # noqa: BLE001
            print(f"[download]   failed: {e!r}")
            tmp.unlink(missing_ok=True)
            continue
        if md5 is not None:
            got = md5sum(tmp)
            if got != md5:
                print(f"[download]   md5 mismatch ({got} != {md5}), skipping")
                tmp.unlink(missing_ok=True)
                continue
        tmp.rename(p)
        print(f"[download]   ok: {p} ({p.stat().st_size} bytes)")
        return p, url
    sys.exit("[download] all mirrors failed")


# --------------------------------------------------------------------------- archive
def iter_archive(path: Path):
    """Yield (relative path inside VOC2007/, file-like reader factory) for regular files."""
    def rel(name: str) -> str | None:
        parts = name.replace("\\", "/").split("/")
        if "VOC2007" not in parts or any(x in ("..", "") for x in parts[:-1]) or name.startswith("/"):
            return None
        return "/".join(parts[parts.index("VOC2007") + 1:])

    if path.suffix == ".tar":
        with tarfile.open(path, "r:") as tf:
            for m in tf:
                if not m.isreg():
                    continue
                r = rel(m.name)
                if r:
                    yield r, (lambda m=m: tf.extractfile(m))
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                r = rel(info.filename)
                if r:
                    yield r, (lambda info=info: zf.open(info))
    else:
        raise ValueError(path)


def parse_xml(data: bytes) -> dict:
    root = ET.fromstring(data)
    size = root.find("size")
    objs = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        d = o.find("difficult")
        objs.append({
            "name": o.findtext("name").strip(),
            "difficult": int(d.text) if d is not None and d.text else 0,
            "box": [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")],
        })
    return {
        "filename": root.findtext("filename").strip(),
        "width": int(size.findtext("width")),
        "height": int(size.findtext("height")),
        "objects": objs,
    }


def extract(archive: Path, img_dir: Path) -> tuple[list[str], dict[str, dict]]:
    img_dir.mkdir(parents=True, exist_ok=True)
    test_ids: list[str] | None = None
    anns: dict[str, dict] = {}
    n_img = 0
    for r, opener in iter_archive(archive):
        if r == "ImageSets/Main/test.txt":
            with opener() as f:
                test_ids = [ln.strip() for ln in f.read().decode().splitlines() if ln.strip()]
        elif r.startswith("Annotations/") and r.endswith(".xml"):
            with opener() as f:
                anns[Path(r).stem] = parse_xml(f.read())
        elif r.startswith("JPEGImages/") and r.endswith(".jpg"):
            name = Path(r).name  # flat, original file name
            with opener() as f, open(img_dir / name, "wb") as out:
                shutil.copyfileobj(f, out, length=1 << 20)
            n_img += 1
    if test_ids is None:
        sys.exit("ImageSets/Main/test.txt not found in archive")
    print(f"[extract] {n_img} JPEGs -> {img_dir}, {len(anns)} XMLs, {len(test_ids)} ids in test.txt")
    return test_ids, anns


# --------------------------------------------------------------------------- convert
def coco_categories(coco_ann: Path) -> list[dict]:
    with open(coco_ann) as f:
        cats = {c["name"]: c for c in json.load(f)["categories"]}
    out = []
    for voc, coco_name in VOC2COCO.items():
        c = cats[coco_name]
        out.append({"id": c["id"], "name": c["name"], "supercategory": c["supercategory"],
                    "voc_name": voc})
    return sorted(out, key=lambda c: c["id"])


def convert(test_ids: list[str], anns: dict[str, dict], categories: list[dict]) -> dict:
    voc2id = {c["voc_name"]: c["id"] for c in categories}
    images, annotations = [], []
    ann_id = 1
    for sid in test_ids:
        a = anns[sid]
        img_id = int(sid)
        images.append({"id": img_id, "file_name": f"{sid}.jpg",
                       "width": a["width"], "height": a["height"]})
        for o in a["objects"]:
            xmin, ymin, xmax, ymax = o["box"]
            x, y, w, h = xmin - 1, ymin - 1, xmax - xmin + 1, ymax - ymin + 1
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": voc2id[o["name"]],
                "bbox": [x, y, w, h], "area": w * h,
                "iscrowd": 1 if o["difficult"] else 0,
            })
            ann_id += 1
    return {
        "info": {"description": "PASCAL VOC2007 test in COCO format with COCO category ids "
                                "(difficult -> iscrowd=1)", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# --------------------------------------------------------------------------- checks
def check(ann_path: Path, img_dir: Path) -> bool:
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ok = True
    with open(ann_path) as f:
        d = json.load(f)
    cats = {c["id"]: c for c in d["categories"]}
    n_img, n_obj = len(d["images"]), len(d["annotations"])
    n_nd = sum(a["iscrowd"] == 0 for a in d["annotations"])
    print(f"[check] images={n_img} objects={n_obj} non-difficult={n_nd} categories={len(cats)}")
    for name, got, exp in [("images", n_img, EXPECTED_IMAGES), ("objects", n_obj, EXPECTED_OBJECTS),
                           ("non-difficult", n_nd, EXPECTED_NONDIFFICULT), ("categories", len(cats), 20)]:
        if got != exp:
            print(f"[check]   FAIL {name}: {got} != {exp}")
            ok = False
    if len({a["id"] for a in d["annotations"]}) != n_obj or len({i["id"] for i in d["images"]}) != n_img:
        print("[check]   FAIL: duplicate ids")
        ok = False

    obj_all, obj_nd, imgs_per = Counter(), Counter(), {c: set() for c in cats}
    for a in d["annotations"]:
        obj_all[a["category_id"]] += 1
        obj_nd[a["category_id"]] += a["iscrowd"] == 0
        imgs_per[a["category_id"]].add(a["image_id"])
    print(f"[check] {'voc':12s} {'coco':14s} {'id':>3s} {'imgs':>5s} {'objs':>5s} {'non-diff':>8s}")
    for cid in sorted(cats):
        c = cats[cid]
        print(f"[check] {c['voc_name']:12s} {c['name']:14s} {cid:3d} {len(imgs_per[cid]):5d} "
              f"{obj_all[cid]:5d} {obj_nd[cid]:8d}")

    # image files exist, open, and sizes match; boxes inside image
    bad = 0
    for im in d["images"]:
        p = img_dir / im["file_name"]
        try:
            with Image.open(p) as pil:
                pil.load()
                if pil.size != (im["width"], im["height"]):
                    print(f"[check]   size mismatch {p}: {pil.size} vs {(im['width'], im['height'])}")
                    bad += 1
        except Exception as e:  # noqa: BLE001
            print(f"[check]   cannot open {p}: {e!r}")
            bad += 1
    sizes = {im["id"]: (im["width"], im["height"]) for im in d["images"]}
    oob = sum(1 for a in d["annotations"]
              if a["bbox"][0] < 0 or a["bbox"][1] < 0 or a["bbox"][2] <= 0 or a["bbox"][3] <= 0
              or a["bbox"][0] + a["bbox"][2] > sizes[a["image_id"]][0]
              or a["bbox"][1] + a["bbox"][3] > sizes[a["image_id"]][1])
    print(f"[check] image files: {n_img - bad}/{n_img} open with matching size; "
          f"boxes outside image / degenerate: {oob}")
    ok &= bad == 0 and oob == 0
    n_files = len(list(img_dir.glob("*.jpg")))
    if n_files != n_img:
        print(f"[check]   note: {n_files} jpg files in {img_dir} vs {n_img} images in JSON")

    # GT boxes as detections (score 1.0) -> AP should be 1.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_path))
        dets = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
                 "score": 1.0} for a in d["annotations"]]
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
    ev.summarize()
    s = ev.stats
    print(f"[check] GT-as-dets: AP={s[0]:.4f} AP50={s[1]:.4f} AP75={s[2]:.4f}")
    # per-class AP@[.5:.95], area=all, maxDets=100
    prec = ev.eval["precision"][:, :, :, 0, -1]  # T,R,K
    per_cls = [float(np.mean(prec[:, :, k][prec[:, :, k] > -1])) for k in range(prec.shape[2])]
    print(f"[check] per-class AP min={min(per_cls):.4f} max={max(per_cls):.4f}")
    if not (abs(s[0] - 1.0) < 1e-6 and min(per_cls) > 1 - 1e-6):
        print("[check]   FAIL: GT-as-detections AP != 1")
        ok = False
    print("[check] ALL OK" if ok else "[check] SOME CHECKS FAILED")
    return ok


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "voc")
    ap.add_argument("--coco-ann", type=Path, default=REPO / "coco/annotations/instances_val2017.json")
    ap.add_argument("--archive", type=Path, default=None, help="use this local archive instead of downloading")
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    img_dir = args.out / "images"
    ann_path = args.out / "annotations" / "instances_voc07test_cocoids.json"
    if not args.check_only:
        dl_dir = args.out / "_download"
        if args.archive:
            archive, src = args.archive, str(args.archive)
        else:
            archive, src = download(dl_dir, args.timeout)
        print(f"[source] {src}")
        test_ids, anns = extract(archive, img_dir)
        missing = [i for i in test_ids if i not in anns]
        if missing:
            sys.exit(f"{len(missing)} test ids have no annotation, e.g. {missing[:5]}")
        data = convert(test_ids, anns, coco_categories(args.coco_ann))
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_path, "w") as f:
            json.dump(data, f)
        print(f"[convert] wrote {ann_path}")
        if not args.keep_archive and archive.parent == dl_dir:
            archive.unlink()
            with contextlib.suppress(OSError):
                dl_dir.rmdir()
            print(f"[cleanup] removed {archive}")
    sys.exit(0 if check(ann_path, img_dir) else 1)


if __name__ == "__main__":
    main()
