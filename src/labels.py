"""Model label index -> GT category id: one path for every model, dataset and evaluation script.

Both the model's label names (config.id2label) and the GT category names are first normalized to the
canonical COCO name (LABEL_ALIASES: the VOC-style names that PekingU/rtdetr_* configs use for 6 of the 80
COCO classes), then matched exactly. By name rather than by label id, because the id conventions differ:
YOLOS and DETR index the original 91-slot COCO ids (with N/A gaps), RT-DETR uses contiguous 0..79 and a
fine-tuned new head (VisDrone) uses 0..9; names are the only convention shared by all of them.
tests/test_category_mapping.py checks that, for every COCO-label model, the name mapping equals the mapping
derived from that model's id convention.

``require_all``: every GT category must receive at least one model label, otherwise ValueError. A missing
class would silently score AP = 0 for it (the bug fixed in 38084a1).
"""
from typing import Dict, Mapping

LABEL_ALIASES = {"aeroplane": "airplane", "diningtable": "dining table", "motorbike": "motorcycle",
                 "pottedplant": "potted plant", "sofa": "couch", "tvmonitor": "tv"}


def canonical(name: str) -> str:
    return LABEL_ALIASES.get(name, name)


def category_mapping(id2label: Mapping, cats: Mapping[int, dict], require_all: bool = True) -> Dict[int, int]:
    """id2label: model label index -> name; cats: GT category id -> {"name": ...} (COCO.cats)."""
    name_to_cat = {}
    for cid, c in cats.items():
        n = canonical(c["name"])
        if n in name_to_cat:
            raise ValueError(f"two GT categories normalize to {n!r}")
        name_to_cat[n] = cid
    mapping = {int(i): name_to_cat[canonical(n)] for i, n in id2label.items() if canonical(n) in name_to_cat}
    missing = sorted(cats[c]["name"] for c in set(cats) - set(mapping.values()))
    if require_all and missing:
        raise ValueError(f"GT categories without a model label: {missing}")
    return mapping
