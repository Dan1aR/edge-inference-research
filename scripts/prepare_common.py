"""Shared helpers for scripts/prepare_<dataset>.py (CPU only, no torch).

* disk guard: refuse a download that would leave < MIN_FREE bytes on / ,
* streamed zip / tar.gz extraction straight from HTTP (no archive kept on disk),
* COCO category lookup (ids/names/supercategories from instances_val2017.json),
* sanity report: per-class counts, small/medium/large fractions, every image
  exists (and its size matches), faster_coco_eval load + GT-as-detections AP == 1.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tarfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO = Path(__file__).resolve().parents[1]
DATASETS = REPO / "datasets"
COCO_VAL_ANN = REPO / "coco/annotations/instances_val2017.json"
MIN_FREE = 8 * 10**9  # bytes that must stay free on / after a download

SMALL, MEDIUM = 32**2, 96**2  # COCO area thresholds


# --------------------------------------------------------------------------- disk
def free_bytes(path: str = "/") -> int:
    return shutil.disk_usage(path).free


def ensure_space(need_bytes: int, what: str, min_free: int = MIN_FREE) -> None:
    free = free_bytes()
    after = free - need_bytes
    print(f"[disk] free={free/1e9:.2f} GB, {what} needs ~{need_bytes/1e9:.2f} GB "
          f"-> {after/1e9:.2f} GB left (min {min_free/1e9:.0f} GB)")
    if after < min_free:
        sys.exit(f"[disk] NOT ENOUGH SPACE for {what}; aborting before download")


def http_size(url: str) -> int | None:
    import httpx
    with httpx.Client(follow_redirects=True, timeout=60) as c:
        r = c.head(url)
        r.raise_for_status()
        n = r.headers.get("content-length")
        return int(n) if n else None


# --------------------------------------------------------------------------- streaming
def _safe_rel(name: str) -> str | None:
    name = name.replace("\\", "/")
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if not parts or name.startswith("/") or ".." in parts:
        return None
    return "/".join(parts)


def _http_chunks(url: str, headers: dict | None = None):
    import httpx
    with httpx.stream("GET", url, timeout=300, follow_redirects=True, headers=headers or {}) as r:
        r.raise_for_status()
        yield from r.iter_bytes(chunk_size=1 << 20)


def stream_unzip_url(url: str, route, headers: dict | None = None) -> int:
    """Stream a zip from `url`. `route(relpath) -> Path | None` decides where each member goes
    (None = skip). Returns the number of files written."""
    from stream_unzip import stream_unzip
    n = 0
    for name, _size, chunks in stream_unzip(_http_chunks(url, headers)):
        rel = _safe_rel(name.decode("utf-8", "replace"))
        dest = route(rel) if rel and not name.endswith(b"/") else None
        if dest is None:
            for _ in chunks:
                pass
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as f:
            for c in chunks:
                f.write(c)
        tmp.rename(dest)
        n += 1
        if n % 2000 == 0:
            print(f"[extract] {n} files", flush=True)
    return n


class _ChunkReader(io.RawIOBase):
    def __init__(self, it):
        self.it, self.buf = it, b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            try:
                self.buf = next(self.it)
            except StopIteration:
                return 0
        n = min(len(b), len(self.buf))
        b[:n], self.buf = self.buf[:n], self.buf[n:]
        return n


def stream_untar_url(url: str, route, headers: dict | None = None) -> int:
    """Stream a .tar(.gz) from `url` (mode r|*), same `route` contract as stream_unzip_url."""
    n = 0
    raw = io.BufferedReader(_ChunkReader(_http_chunks(url, headers)), buffer_size=1 << 20)
    with tarfile.open(fileobj=raw, mode="r|*") as tf:
        for m in tf:
            if not m.isreg():
                continue
            rel = _safe_rel(m.name)
            dest = route(rel) if rel else None
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            with tf.extractfile(m) as src, open(tmp, "wb") as f:
                shutil.copyfileobj(src, f, length=1 << 20)
            tmp.rename(dest)
            n += 1
            if n % 2000 == 0:
                print(f"[extract] {n} files", flush=True)
    return n


# --------------------------------------------------------------------------- categories
def coco_categories(names: list[str], extra: dict[str, dict] | None = None) -> list[dict]:
    """COCO category dicts (true COCO id/name/supercategory) for the given COCO names."""
    with open(COCO_VAL_ANN) as f:
        cats = {c["name"]: c for c in json.load(f)["categories"]}
    out = []
    for n in names:
        c = dict(id=cats[n]["id"], name=n, supercategory=cats[n]["supercategory"])
        c.update((extra or {}).get(n, {}))
        out.append(c)
    return sorted(out, key=lambda c: c["id"])


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.part")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.rename(path)
    print(f"[convert] wrote {path} ({len(data['images'])} images, {len(data['annotations'])} anns)")


# --------------------------------------------------------------------------- checks
def sanity(ann_path: Path, img_root: Path, expected_images: int | None = None,
           check_sizes: bool = True, gt_ap: bool = True) -> dict:
    """Print + return a summary. Exits non-zero on hard failures."""
    import numpy as np
    from faster_coco_eval import COCO, COCOeval_faster

    with open(ann_path) as f:
        d = json.load(f)
    ok = True
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}
    anns = d["annotations"]
    print(f"\n[check] {ann_path.relative_to(REPO)}  (images under {img_root.relative_to(REPO)})")
    if len(imgs) != len(d["images"]) or len({a["id"] for a in anns}) != len(anns):
        print("[check]   FAIL duplicate image/annotation ids"); ok = False
    if expected_images is not None and len(imgs) != expected_images:
        print(f"[check]   FAIL images {len(imgs)} != expected {expected_images}"); ok = False
    bad_ref = sum(a["image_id"] not in imgs or a["category_id"] not in cats for a in anns)
    if bad_ref:
        print(f"[check]   FAIL {bad_ref} annotations reference unknown image/category"); ok = False

    # images exist (+ size)
    missing, mism = 0, 0
    for im in d["images"]:
        p = img_root / im["file_name"]
        if not p.is_file():
            missing += 1
            continue
        if check_sizes and image_size(p) != (im["width"], im["height"]):
            mism += 1
    print(f"[check] image files: {len(imgs) - missing}/{len(imgs)} present, size mismatches: {mism}")
    ok &= missing == 0 and mism == 0

    # boxes
    degen = sum(1 for a in anns if a["bbox"][2] <= 0 or a["bbox"][3] <= 0)
    over = [max(-a["bbox"][0], -a["bbox"][1],
                a["bbox"][0] + a["bbox"][2] - imgs[a["image_id"]]["width"],
                a["bbox"][1] + a["bbox"][3] - imgs[a["image_id"]]["height"]) for a in anns]
    oob = [o for o in over if o > 1e-3]
    if degen:
        print(f"[check]   FAIL {degen} degenerate boxes (w or h <= 0)"); ok = False
    if oob:
        print(f"[check]   WARN {len(oob)} boxes extend outside the image (max {max(oob):.1f} px); "
              "left unchanged")
    nc = [a for a in anns if not a.get("iscrowd", 0)]
    n_crowd = len(anns) - len(nc)
    per, per_img = Counter(a["category_id"] for a in nc), {c: set() for c in cats}
    per_crowd = Counter(a["category_id"] for a in anns if a.get("iscrowd", 0))
    for a in nc:
        per_img[a["category_id"]].add(a["image_id"])
    sizes = Counter("small" if a["area"] < SMALL else "medium" if a["area"] < MEDIUM else "large"
                    for a in nc)
    frac = {k: sizes[k] / max(len(nc), 1) for k in ("small", "medium", "large")}
    per_image_counts = Counter(a["image_id"] for a in nc)
    mx = max(per_image_counts.values(), default=0)
    over100 = sum(v > 100 for v in per_image_counts.values())
    print(f"[check] images={len(imgs)} boxes(non-crowd)={len(nc)} crowd/ignore={n_crowd} "
          f"categories={len(cats)}")
    print(f"[check] size fractions (non-crowd): small={frac['small']:.3f} "
          f"medium={frac['medium']:.3f} large={frac['large']:.3f}")
    print(f"[check] boxes/image: mean={len(nc)/max(len(imgs),1):.1f} max={mx} "
          f"images with >100 boxes: {over100}")
    print(f"[check] {'id':>4s} {'name':20s} {'imgs':>6s} {'boxes':>7s} {'ignore':>7s}")
    for cid in sorted(cats):
        print(f"[check] {cid:4d} {cats[cid]:20s} {len(per_img[cid]):6d} {per[cid]:7d} "
              f"{per_crowd[cid]:7d}")

    ap = None
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ann_path))
    print(f"[check] faster_coco_eval.COCO loaded: {len(gt.getImgIds())} imgs, "
          f"{len(gt.getAnnIds())} anns, {len(gt.getCatIds())} cats")
    if gt_ap and nc:
        dets = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"],
                 "score": 1.0} for a in nc]
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(dets)
            ev = COCOeval_faster(gt, dt, "bbox")
            ev.params.maxDets = [1, 10, max(100, mx)]  # GT-as-dets: no truncation
            ev.evaluate(); ev.accumulate(); ev.summarize()
        ap = float(ev.stats[0])
        print(f"[check] GT-as-detections AP (maxDets={max(100, mx)}) = {ap:.4f}")
        if abs(ap - 1) > 1e-3:
            print("[check]   FAIL GT-as-detections AP != 1"); ok = False
    print("[check] ALL OK" if ok else "[check] SOME CHECKS FAILED")
    if not ok:
        sys.exit(1)
    return dict(images=len(imgs), boxes=len(nc), ignore=n_crowd, out_of_image=len(oob), **frac, max_per_image=mx,
                imgs_over_100=over100, per_class={cats[c]: per[c] for c in sorted(cats)})


def du(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
