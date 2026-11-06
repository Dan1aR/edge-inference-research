"""Sanity checks of the VisDrone evaluation from saved detections (CPU only, no model run).

  python scripts/check_visdrone_eval.py results/x7v/B_s42_none.dets.npz [--vis_dir DIR]

Reports: GT statistics (classes, ignore boxes, objects per image), detections per image, AP at
maxDets 100 vs 500, oracle runs (GT boxes as detections; GT truncated to the 100 largest / 100 random
boxes per image = the ceiling for a 100-query detector), box-convention perturbations of the saved
detections (xyxy reading, swapped h/w scaling, half-pixel shift), and optionally draws GT (green),
ignore regions (grey) and detections with score >= 0.3 (red) on 3 val images.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import DATASETS  # noqa: E402

from faster_coco_eval import COCO, COCOeval_faster  # noqa: E402


def ap(gt, anns, img_ids, max_dets=100):
    if not anns:
        return {"AP": 0.0}
    dt = gt.loadRes(anns)
    ev = COCOeval_faster(gt, dt, iouType="bbox", print_function=lambda *a, **k: None)
    ev.params.imgIds = img_ids
    ev.params.maxDets = [1, 10, max_dets]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {"AP": round(float(s[0]), 4), "AP50": round(float(s[1]), 4), "AP75": round(float(s[2]), 4),
            f"AR@{max_dets}": round(float(s[8]), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dets")
    p.add_argument("--dataset", default="visdrone_val_native")
    p.add_argument("--vis_dir", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    img_root, ann_file = DATASETS[a.dataset]
    gt = COCO(str(ann_file))
    d = np.load(a.dets)
    img_ids = sorted(int(i) for i in d["image_ids_evaluated"])
    res = {}

    real = [x for x in gt.dataset["annotations"] if not x["iscrowd"]]
    crowd = [x for x in gt.dataset["annotations"] if x["iscrowd"]]
    per_img = Counter(x["image_id"] for x in real)
    n = np.array([per_img.get(i, 0) for i in img_ids])
    res["gt"] = {"images": len(img_ids), "objects": int(n.sum()), "per_image_mean": round(float(n.mean()), 1),
                 "per_image_max": int(n.max()), "frac_images_gt100": round(float((n > 100).mean()), 3),
                 "frac_objects_beyond_100": round(float(np.maximum(n - 100, 0).sum() / n.sum()), 3),
                 "classes": {gt.cats[c]["name"]: k for c, k in sorted(Counter(x["category_id"] for x in real).items())},
                 "source_classes_in_gt": sorted(set(x["visdrone_cat"] for x in real)),
                 "ignore_boxes_unique": len(crowd) // len(gt.cats), "ignore_source_classes":
                 sorted(set(x["visdrone_cat"] for x in crowd))}

    dets = [{"image_id": int(i), "category_id": int(c), "bbox": [float(v) for v in b], "score": float(s)}
            for i, c, b, s in zip(d["image_id"], d["category_id"], d["bbox"], d["score"])]
    dpi = Counter(x["image_id"] for x in dets)
    res["dets"] = {"total": len(dets), "per_image_max": max(dpi.values()), "per_image_min": min(dpi.values()),
                   "categories": sorted(set(x["category_id"] for x in dets))}
    res["ap_maxdets100"] = ap(gt, dets, img_ids, 100)
    res["ap_maxdets500"] = ap(gt, dets, img_ids, 500)

    # oracles: GT as detections (score 1); and GT truncated to 100 boxes per image
    rng = np.random.default_rng(0)
    orc = [{"image_id": x["image_id"], "category_id": x["category_id"], "bbox": x["bbox"], "score": 1.0} for x in real]
    res["oracle_all_gt"] = ap(gt, orc, img_ids, 500)
    by_img = {}
    for x in orc:
        by_img.setdefault(x["image_id"], []).append(x)
    top_area = [y for v in by_img.values() for y in sorted(v, key=lambda z: -z["bbox"][2] * z["bbox"][3])[:100]]
    rand = [y for v in by_img.values() for y in (list(rng.permutation(np.array(v, dtype=object)))[:100])]
    res["oracle_gt_100_largest"] = ap(gt, top_area, img_ids, 100)
    res["oracle_gt_100_random"] = ap(gt, rand, img_ids, 100)

    # box-convention perturbations of the real detections (a convention bug would make one of them better)
    size = {i: (gt.imgs[i]["width"], gt.imgs[i]["height"]) for i in img_ids}

    def pert(f):
        return [dict(x, bbox=f(x["bbox"], *size[x["image_id"]])) for x in dets]
    res["perturb_read_as_xyxy"] = ap(gt, pert(lambda b, w, h: [b[0], b[1], b[2] - b[0], b[3] - b[1]]), img_ids)
    res["perturb_swap_hw_scale"] = ap(gt, pert(lambda b, w, h: [b[0] * h / w, b[1] * w / h, b[2] * h / w,
                                                                 b[3] * w / h]), img_ids)
    res["perturb_shift_half_px"] = ap(gt, pert(lambda b, w, h: [b[0] + 0.5, b[1] + 0.5, b[2], b[3]]), img_ids)
    res["perturb_center_as_corner"] = ap(gt, pert(lambda b, w, h: [b[0] + b[2] / 2, b[1] + b[3] / 2, b[2], b[3]]),
                                         img_ids)

    if a.vis_dir:
        from PIL import Image, ImageDraw
        out = Path(a.vis_dir)
        out.mkdir(parents=True, exist_ok=True)
        pick = [img_ids[k] for k in (0, len(img_ids) // 2, len(img_ids) - 1)]
        for i in pick:
            im = Image.open(Path(img_root) / gt.imgs[i]["file_name"]).convert("RGB")
            dr = ImageDraw.Draw(im)
            seen = set()
            for x in gt.imgToAnns[i]:
                x0, y0, w, h = x["bbox"]
                if x["iscrowd"]:
                    if tuple(x["bbox"]) in seen:
                        continue
                    seen.add(tuple(x["bbox"]))
                    dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(160, 160, 160), width=2)
                else:
                    dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(0, 255, 0), width=2)
            k = 0
            for x in dets:
                if x["image_id"] == i and x["score"] >= 0.3:
                    x0, y0, w, h = x["bbox"]
                    dr.rectangle([x0, y0, x0 + w, y0 + h], outline=(255, 0, 0), width=2)
                    k += 1
            im.save(out / f"vis_{i}.jpg", quality=85)
            print(f"vis image {i} {gt.imgs[i]['file_name']} {im.size}: gt={len([x for x in gt.imgToAnns[i] if not x['iscrowd']])}"
                  f" dets>=0.3: {k}")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
