# Extra evaluation / fine-tuning datasets

Every dataset is produced by an idempotent CPU-only script `scripts/prepare_<name>.py` (shared helpers in
`scripts/prepare_common.py`). The scripts stream-extract archives straight from HTTP, so no archive is ever kept
on disk. Before each download they check that at least 8 GB stay free on `/`. A rerun skips the download if the
images are already complete, then rewrites the JSONs and repeats the checks. Checks: image count, every
referenced image exists and its size matches the JSON, no degenerate boxes, the file loads with
`faster_coco_eval.COCO`, and AP = 1 when the non-crowd GT is fed back as detections. Each check run also prints
per-class box counts and the S/M/L fractions (COCO area thresholds 32² and 96²).

Run: `CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/python scripts/prepare_<name>.py [--check-only]`

**Ignore convention.** Ignore regions become `iscrowd=1` boxes. COCOeval matches crowd GT only within the same
category, so each ignore box is **replicated once per category in the file**. A detection of any class that
falls mostly inside the box (COCO crowd IoU = intersection / detection area) then counts as neither TP nor FP.
That is roughly VisDrone's "inside ignored region" rule. Crowd boxes are excluded from all box counts below.
For training, drop `iscrowd=1` boxes. The HF `YolosImageProcessor` already does this
(`prepare_coco_detection_annotation`).

"cocoids" files list only the mapped classes, using their true COCO ids, names and supercategories. Detections
of other COCO classes are therefore ignored by COCOeval, as in `voc/` (`scripts/prepare_voc07_coco.py`).

S/M/L below = fraction of non-crowd boxes with area < 32², < 96², ≥ 96².

## VisDrone2019-DET (val, train): ready
* Source: Ultralytics mirror of the official zips,
  `https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{val,train}.zip` (82 MB / 1.55 GB).
  Official site: https://github.com/VisDrone/VisDrone-Dataset.
* License: research / non-commercial use only (VisDrone terms). Do not redistribute.
* Images: `datasets/visdrone/VisDrone2019-DET-{val,train}/images/`. Original txt annotations are next to them.
* JSONs: `datasets/visdrone/annotations/instances_{val,train}_{cocoids,native}.json`
* Conventions: image id = 1..N in sorted file-name order; `file_name` = original name. Each box keeps
  `visdrone_cat`, `truncation` and `occlusion`. Boxes are clipped to the image: none needed clipping, and 3 train
  boxes with zero width or height were dropped.
* cocoids mapping: pedestrian, people → person(1); bicycle → bicycle(2); car, van → car(3); motor → motorcycle(4);
  bus → bus(6); truck → truck(8). Ignored (iscrowd, ×6 categories): class 0 "ignored regions", 11 "others",
  tricycle(7), awning-tricycle(8), and any box with score = 0.
* native: the 10 classes with ids 1..10 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle,
  bus, motor). Ignored (×10): classes 0 and 11, and score = 0.

| split | images | boxes cocoids | boxes native | S / M / L (cocoids) | max boxes/img | imgs >100 boxes |
|---|---|---|---|---|---|---|
| val | 548 | 37,182 | 38,759 | 0.690 / 0.282 / 0.028 | 317 | 93 |
| train | 6,471 | 335,146 | 343,204 | 0.609 / 0.336 / 0.055 | 898 | 666 |

Per-class val counts (cocoids): person 13,969 · car 16,039 · motorcycle 4,886 · bicycle 1,287 · truck 750 · bus 251.

**Deviation from the official protocol.** The official VisDrone toolkit computes AP@[.5:.95] with maxDets = 500
(also reporting AR at 1/10/100/500), removes detections inside ignored regions, and has no S/M/L split. We use
the plain COCO protocol (maxDets = 100) with crowd boxes for the ignore regions. 17% of val images have more than
100 GT boxes, so with maxDets = 100 recall is capped on those images. Consider also reporting
`params.maxDets = [1, 10, 500]`. Since 69% of val boxes are small, the input resolution will matter a lot.

## ImageNetV2 matched-frequency: ready
* Source: https://huggingface.co/datasets/vaishaal/ImageNetV2, `imagenetv2-matched-frequency.tar.gz` (1.26 GB,
  public). Paper: Recht et al., "Do ImageNet Classifiers Generalize to ImageNet?" (ICML 2019).
* License: the images come from Flickr under their original terms (as with ImageNet). Research use only.
* Images: `datasets/imagenetv2/matched-frequency/<label>/<hash>.jpeg`. 10,000 images, 10 per class, all RGB,
  short side 59–2583 px (median 364).
* Lists: `datasets/imagenetv2/matched-frequency.csv` (header `path,label`) and `matched-frequency.json`. Paths are
  relative to `datasets/imagenetv2/`. Each label is the folder name, i.e. the standard ImageNet-1k index 0..999
  (sorted-wnid order, the same as the `id2label` in DeiT checkpoints).
* Protocol: the usual ImageNet val preprocessing (resize 256 / center crop 224 for DeiT-224); report top-1 and top-5.

## COCO-O: ready
* Source: the authors' Google Drive `ood_coco.zip` (1.1 GB, file id `1aBfIJN0zo_i80Hv4p7Ch7M8pRzO37qbq`), linked
  from https://github.com/alibaba/easyrobust/tree/main/benchmarks/coco_o. It is fetched through Drive's
  "download anyway" form, so gdown is not needed. Paper: Mao et al., ICCV 2023.
* License: easyrobust is Apache-2.0. The images are web-collected; research use only.
* Originals: `datasets/coco_o/<domain>/{val2017/, annotations/instances_val2017.json}`.
* JSONs: `datasets/coco_o/annotations/instances_<domain>.json` with `file_name = <domain>/val2017/<file>`, and
  `instances_all.json` with ids remapped to `(domain_idx+1)·10⁷ + id` and a `domain` field on each image.
  **Image root for all of them: `datasets/coco_o/`.** The 80 COCO categories with COCO ids; no crowd boxes; the
  annotation content is unchanged.
* Totals match the paper: 6,782 images and 26,624 boxes. 330 boxes extend up to 35 px outside the image. They
  are left as released.

| domain | images | boxes | S / M / L |
|---|---|---|---|
| cartoon | 1,996 | 8,774 | 0.110 / 0.306 / 0.583 |
| handmake | 961 | 3,266 | 0.043 / 0.193 / 0.764 |
| painting | 954 | 4,879 | 0.157 / 0.345 / 0.498 |
| sketch | 992 | 3,707 | 0.141 / 0.283 / 0.576 |
| tattoo | 918 | 1,489 | 0.052 / 0.170 / 0.778 |
| weather | 961 | 4,509 | 0.160 / 0.389 / 0.451 |
| all | 6,782 | 26,624 | 0.120 / 0.303 / 0.577 |

* Protocol: the paper reports COCO AP per domain and their mean, plus "effective robustness" against COCO val AP.
  The per-domain JSONs reproduce that. `instances_all.json` (pooled AP) is an extra and is not the paper's
  headline number.

## BDD100K val: ready
* Source: val-only HF mirror by Hirundo, no login:
  https://huggingface.co/datasets/hirundo-io/bdd100k-validation-only (`BDD100K Val from Hirundo.zip`, 575 MB).
  `hirundo-io/bdd100k-val` (570 MB) has the same images and the same label counts. The official download
  (https://bdd-data.berkeley.edu / dl.cv.ethz.ch) needs registration, and its 100k-images zip is several GB (train+val+test).
* License: the mirror ships the BSD-3 LICENSE of the bdd100k code repo. The BDD100K data itself is under the
  BDD100K license (research use).
* Labels: the mirror's CSV is a flat export of the det-2020 val labels (13 classes, 186,033 boxes, coordinates
  rounded to integer pixels). **It has no occluded / truncated / crowd attributes.** Kept in
  `datasets/bdd100k/raw/`.
* Images: `datasets/bdd100k/images/val/` (10,000 images, 1280×720). JSON: `datasets/bdd100k/annotations/instances_val_cocoids.json`.
* Mapping: pedestrian, rider → person(1); car(3); truck(8); bus(6); train(7); motorcycle(4); bicycle(2);
  traffic light → traffic light(10). Ignored (iscrowd, ×8 categories): traffic sign (34,724; it has no COCO
  equivalent, since COCO "stop sign" is a narrow subset), other vehicle (85), other person (1), trailer (2).
* Boxes: 151,221 in total: car 102,837 · traffic light 26,884 · person 14,083 · truck 4,243 · bus 1,660 ·
  bicycle 1,039 · motorcycle 460 · train 15. S/M/L = 0.510 / 0.341 / 0.149. At most 61 boxes per image.
* Deviation: the official BDD det eval uses COCO-style AP over its 10 classes with its own ignore rules. Here
  traffic sign is ignored and the categories are COCO ones.

## CrowdHuman val: script only (not downloaded; over the disk budget)
* Source: HF mirror of the official files (same sizes as the Google Drive originals):
  https://huggingface.co/datasets/sshao0516/CrowdHuman (also `Funtion/CrowdHuman`). `annotation_val.odgt`
  (23 MB) is already downloaded to `datasets/crowdhuman/raw/`. `CrowdHuman_val.zip` (2.49 GB) is not downloaded.
* License: CrowdHuman terms, non-commercial research only.
* odgt stats: 4,370 images; 99,481 person boxes kept; 3,634 person boxes with `extra.ignore = 1`; 24,601 "mask"
  regions. All the ignored ones become iscrowd = 1.
* Conversion: full-body `fbox` → COCO person(1), clipped to the image (raw box in `fbox_raw`, plus `vbox` and
  `hbox`). mask / ignore → iscrowd = 1. With a single category, no replication is needed.
* To finish: `scripts/prepare_crowdhuman.py` (needs about 2.5 GB). Deviation: the official metrics are AP,
  MR⁻² and JI from CrowdHuman's own matcher with no detection cap. Here plain COCOeval is used, and maxDets = 100
  matters because there are about 23 persons per image.
