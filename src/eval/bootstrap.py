"""Paired image-level bootstrap for COCO AP differences.

COCOeval.evaluate() is run once per detection set; afterwards AP@[.5:.95] (area=all, maxDets=100)
is recomputed for resampled image multisets by weighting every detection / GT with the
multiplicity of its image. With all weights equal to 1 this reproduces pycocotools exactly.
"""
from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Sequence

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class PerImageEval:
    """Per-category sorted detections with their image index and TP/FP flags at all IoU thresholds."""

    def __init__(self, coco_gt: COCO, dets: Dict[str, np.ndarray]):
        anns = [
            {"image_id": int(i), "category_id": int(c), "bbox": [float(x) for x in b], "score": float(s)}
            for i, c, b, s in zip(dets["image_id"], dets["category_id"], dets["bbox"], dets["score"])
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            ev = COCOeval(coco_gt, coco_gt.loadRes(anns), "bbox")
            ev.params.imgIds = sorted(set(int(i) for i in dets["image_ids_evaluated"]))
            ev.evaluate()
        p = ev.params
        self.img_ids = list(p.imgIds)
        img_index = {img: n for n, img in enumerate(self.img_ids)}
        n_img, n_area = len(p.imgIds), len(p.areaRng)
        self.n_thr = len(p.iouThrs)
        self.rec_thrs = p.recThrs
        self.per_cat = []
        max_det = p.maxDets[-1]
        for k in range(len(p.catIds)):
            entries = [ev.evalImgs[k * n_area * n_img + i] for i in range(n_img)]  # area index 0 = all
            entries = [(i, e) for i, e in enumerate(entries) if e is not None]
            if not entries:
                self.per_cat.append(None)
                continue
            scores = np.concatenate([e["dtScores"][:max_det] for _, e in entries])
            order = np.argsort(-scores, kind="mergesort")
            dt_img = np.concatenate([np.full(len(e["dtScores"][:max_det]), i) for i, e in entries])[order]
            dtm = np.concatenate([e["dtMatches"][:, :max_det] for _, e in entries], axis=1)[:, order]
            dt_ig = np.concatenate([e["dtIgnore"][:, :max_det] for _, e in entries], axis=1)[:, order]
            gt_count = np.zeros(n_img)
            for i, e in entries:
                gt_count[i] = np.count_nonzero(e["gtIgnore"] == 0)
            self.per_cat.append({
                "img": dt_img,
                "tp": np.logical_and(dtm, ~dt_ig).astype(np.float32),
                "fp": np.logical_and(np.logical_not(dtm), ~dt_ig).astype(np.float32),
                "gt": gt_count,
            })

    def ap(self, weights: np.ndarray) -> np.ndarray:
        """AP for each row of ``weights`` (B, n_images); returns (B,)."""
        weights = np.atleast_2d(weights).astype(np.float32)
        B = weights.shape[0]
        sums = np.zeros(B)
        counts = np.zeros(B)
        R = len(self.rec_thrs)
        for c in self.per_cat:
            if c is None:
                continue
            npig = weights @ c["gt"]  # (B,)
            valid = npig > 0
            if not valid.any():
                continue
            w = weights[:, c["img"]]  # (B, D)
            for t in range(self.n_thr):
                tp = np.cumsum(w * c["tp"][t], axis=1)
                fp = np.cumsum(w * c["fp"][t], axis=1)
                rc = tp / np.maximum(npig[:, None], 1e-12)
                pr = tp / (tp + fp + np.spacing(1))
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
                q = np.zeros((B, R))
                for b in range(B):
                    if not valid[b]:
                        continue
                    inds = np.searchsorted(rc[b], self.rec_thrs, side="left")
                    ok = inds < pr.shape[1]
                    q[b, ok] = pr[b, inds[ok]]
                sums += np.where(valid, q.mean(axis=1), 0.0)
                counts += valid
        return sums / np.maximum(counts, 1)


def paired_bootstrap(evals: Sequence[PerImageEval], n_boot: int = 1000, seed: int = 0, chunk: int = 50) -> np.ndarray:
    """Returns (n_boot, len(evals)) AP samples computed on identical image resamples."""
    n = len(evals[0].img_ids)
    assert all(e.img_ids == evals[0].img_ids for e in evals)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_boot, len(evals)))
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        weights = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(b)])
        for j, e in enumerate(evals):
            out[s : s + b, j] = e.ap(weights)
    return out


def summarize_delta(samples: np.ndarray, point: float) -> Dict[str, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"delta": point, "ci95_lo": float(lo), "ci95_hi": float(hi), "se": float(samples.std(ddof=1))}
