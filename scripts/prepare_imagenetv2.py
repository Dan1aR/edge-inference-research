#!/usr/bin/env python
"""Prepare ImageNetV2 matched-frequency (Recht et al. 2019) for ImageNet-1k classifier eval.

Source: https://huggingface.co/datasets/vaishaal/ImageNetV2 (imagenetv2-matched-frequency.tar.gz,
1.26 GB, public), stream-extracted (no archive on disk) to
  datasets/imagenetv2/matched-frequency/<class_idx>/<hash>.jpeg
In the archive (imagenetv2-matched-frequency-format-val/<k>/...) the folder name k is the
ImageNet-1k class index in the standard (sorted-wnid) order used by torchvision / timm / HF
(id2label of facebook/deit-*), so label = int(folder). 10 images per class, 10,000 total.

Output: datasets/imagenetv2/matched-frequency.csv  (header 'path,label'; path relative to
datasets/imagenetv2/) and the same list as matched-frequency.json ([{"path","label"}, ...]).

Usage: CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/python scripts/prepare_imagenetv2.py [--check-only]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_common as pc  # noqa: E402

URL = ("https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/"
       "imagenetv2-matched-frequency.tar.gz")
OUT = pc.DATASETS / "imagenetv2"
IMG = OUT / "matched-frequency"
PREFIX = "imagenetv2-matched-frequency-format-val/"
N_CLASSES, PER_CLASS = 1000, 10


def files() -> list[Path]:
    return sorted(p for p in IMG.glob("*/*") if p.is_file()) if IMG.is_dir() else []


def download() -> None:
    if len(files()) == N_CLASSES * PER_CLASS:
        print(f"[download] {IMG} already complete, skipping")
        return
    pc.ensure_space((pc.http_size(URL) or 0) + 50 * 10**6, "ImageNetV2 matched-frequency")

    def route(rel: str):
        if not rel.startswith(PREFIX):
            return None
        parts = rel[len(PREFIX):].split("/")
        if len(parts) != 2 or not parts[0].isdigit() or parts[1].startswith("."):
            return None
        return IMG / str(int(parts[0])) / parts[1]

    print(f"[download] streaming {URL}")
    print(f"[download] extracted {pc.stream_untar_url(URL, route)} files")


def build() -> list[dict]:
    rows = [{"path": str(p.relative_to(OUT)), "label": int(p.parent.name)} for p in files()]
    rows.sort(key=lambda r: (r["label"], r["path"]))
    with open(OUT / "matched-frequency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label"])
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "matched-frequency.json", "w") as f:
        json.dump(rows, f)
    print(f"[convert] wrote {OUT/'matched-frequency.csv'} and .json ({len(rows)} rows)")
    return rows


def check() -> None:
    from PIL import Image
    with open(OUT / "matched-frequency.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    cnt = Counter(int(r["label"]) for r in rows)
    ok = len(rows) == N_CLASSES * PER_CLASS and set(cnt) == set(range(N_CLASSES)) \
        and set(cnt.values()) == {PER_CLASS}
    bad, modes, sizes = 0, Counter(), []
    for r in rows:
        try:
            with Image.open(OUT / r["path"]) as im:
                im.load()
                modes[im.mode] += 1
                sizes.append(min(im.size))
        except Exception as e:  # noqa: BLE001
            print(f"[check]   cannot open {r['path']}: {e!r}")
            bad += 1
    sizes.sort()
    print(f"[check] rows={len(rows)} classes={len(cnt)} per-class={sorted(set(cnt.values()))} "
          f"unreadable={bad} modes={dict(modes)} "
          f"short-side min/median/max={sizes[0]}/{sizes[len(sizes)//2]}/{sizes[-1]}")
    ok &= bad == 0
    print("[check] ALL OK" if ok else "[check] SOME CHECKS FAILED")
    print(f"[disk] datasets/imagenetv2 uses {pc.du(OUT)/1e9:.2f} GB")
    if not ok:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    if not args.check_only:
        download()
        build()
    check()


if __name__ == "__main__":
    main()
