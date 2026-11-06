"""Model label -> GT category mapping (src/labels.py via src.experiment._category_mapping), CPU only.

PekingU/rtdetr_* configs name 6 of the 80 COCO classes VOC-style (aeroplane, diningtable, motorbike,
pottedplant, sofa, tvmonitor); before 38084a1 their detections were dropped (74/80 classes on COCO).
Checks: every COCO-label model covers all GT categories on every dataset; for models whose names were
already COCO names the mapping equals the old exact-name mapping; the name mapping equals the mapping
implied by each model's label-id convention; a missing class raises."""
import pathlib
import sys
import types

import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src import experiment as ex  # noqa: E402

COCO_DS = ["coco_val", "voc07_test", "visdrone_val", "bdd_val", "coco_o", "crowdhuman_val"]


def _model(name):
    from transformers import AutoConfig
    return types.SimpleNamespace(config=AutoConfig.from_pretrained(ex.MODEL_ZOO[name][0]))


def _exact(model, gt):
    n = {c["name"]: i for i, c in gt.cats.items()}
    return {int(k): n[v] for k, v in model.config.id2label.items() if v in n}


@pytest.fixture(scope="module")
def gts():
    from faster_coco_eval import COCO
    return {ds: COCO(str(ex.DATASETS[ds][1])) for ds in COCO_DS if pathlib.Path(ex.DATASETS[ds][1]).exists()}


@pytest.mark.parametrize("model", ["tiny", "small", "base", "detr_r50", "rtdetr_r18", "rtdetr_r50"])
def test_all_gt_categories_mapped(gts, model):
    m = _model(model)
    for ds, gt in gts.items():
        mapping = ex._category_mapping(m, gt)
        assert set(mapping.values()) == set(gt.cats), (ds, sorted(set(gt.cats) - set(mapping.values())))
        assert len(set(mapping.values())) == len(mapping), ds  # injective


@pytest.mark.parametrize("model", ["tiny", "small", "base", "detr_r50"])
def test_exact_name_models_unchanged(gts, model):
    m = _model(model)
    for ds, gt in gts.items():
        assert ex._category_mapping(m, gt) == _exact(m, gt), ds


def test_rtdetr_aliases(gts):
    gt = gts["coco_val"]
    m = ex._category_mapping(_model("rtdetr_r18"), gt)
    names = {gt.cats[c]["name"] for c in set(m.values()) - set(_exact(_model("rtdetr_r18"), gt).values())}
    assert names == {"airplane", "dining table", "motorcycle", "potted plant", "couch", "tv"}


def _by_id(model_name, model, gt):
    """Mapping implied by the label-id convention: YOLOS/DETR index = COCO category id (91 slots, N/A gaps),
    RT-DETR index k = k-th COCO category id in ascending order (contiguous 0..79)."""
    if model_name.startswith("rtdetr"):
        ids = sorted(gt.cats)
        return {int(k): ids[int(k)] for k in model.config.id2label}
    return {int(k): int(k) for k, v in model.config.id2label.items() if v != "N/A" and int(k) in gt.cats}


@pytest.mark.parametrize("model", ["tiny", "small", "base", "detr_r50", "rtdetr_r18", "rtdetr_r50"])
def test_names_agree_with_id_convention(gts, model):
    m = _model(model)
    assert ex._category_mapping(m, gts["coco_val"]) == _by_id(model, m, gts["coco_val"])


def test_missing_class_raises():
    from src.labels import category_mapping
    cats = {1: {"name": "person"}, 2: {"name": "unicorn"}}
    with pytest.raises(ValueError, match="unicorn"):
        category_mapping({0: "person"}, cats)
    assert category_mapping({0: "person"}, cats, require_all=False) == {0: 1}
