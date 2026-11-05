"""The diag fix for a varying channel width (src/generic_arith._diag_update, commit b8c7b18) must not change
anything on a model with a fixed channel width: on YOLOS-tiny (COCO val, 64 images, arithmetic "all") the
per-op statistics (n, per-channel sum, sq, sig) and the scripts/x_error_decomp.py summary from the pre-fix
code (4cf4aa0, copied verbatim below) and from the current code are bitwise equal."""
import importlib.util
import pathlib
import sys

import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src import generic_arith as ga  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _diag_update_old(ctrl, op_id, y, y_ref, ch_dim):  # src/generic_arith.py @ 4cf4aa0, verbatim
    d = (y.float() - y_ref.float()).movedim(ch_dim, -1)
    r = y_ref.float().movedim(ch_dim, -1)
    d2, r2 = d.reshape(-1, d.shape[-1]), r.reshape(-1, r.shape[-1])
    st = ctrl.diag.setdefault(op_id, {"n": 0, "sum": torch.zeros(d2.shape[1], dtype=torch.float64, device=d.device),
                                      "sq": 0.0, "sig": 0.0})
    st["n"] += d2.shape[0]
    st["sum"] += d2.sum(0, dtype=torch.float64)
    st["sq"] += float(d2.double().pow(2).sum())
    st["sig"] += float(r2.double().pow(2).sum())


def _summary_old(diag):  # scripts/x_error_decomp.py run() post-processing @ 4cf4aa0, verbatim
    out = {}
    for op, st in diag.items():
        mean = st["sum"] / st["n"]
        sys_energy = float(mean.pow(2).sum()) * st["n"]
        out[op] = {"component": op.split("#")[1], "rel_err": (st["sq"] / st["sig"]) ** 0.5,
                   "systematic_fraction": sys_energy / st["sq"] if st["sq"] > 0 else 0.0}
    tot_sq = sum(st["sq"] for st in diag.values())
    tot_sys = sum(float((st["sum"] / st["n"]).pow(2).sum()) * st["n"] for st in diag.values())
    return {"ops": out, "systematic_fraction_all_ops": tot_sys / tot_sq}


@cuda
def test_diag_fixed_width_bitwise(monkeypatch):
    spec = importlib.util.spec_from_file_location("x_error_decomp", ROOT / "scripts" / "x_error_decomp.py")
    dec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dec)
    ctrls, orig = [], dec.load_model

    def load_model(*a, **k):
        r = orig(*a, **k)
        ctrls.append(r[2])
        return r
    monkeypatch.setattr(dec, "load_model", load_model)

    summary_new = dec.run("tiny", None, 64, "all")  # current _diag_update + current post-processing
    monkeypatch.setattr(ga, "_diag_update", _diag_update_old)
    dec.run("tiny", None, 64, "all")  # pre-fix _diag_update
    new, old = ctrls[0].diag, ctrls[1].diag

    assert len(new) > 0 and list(new) == list(old)
    for op in new:
        a, b = new[op], old[op]
        assert a["sum"] is not None, op  # fixed width: the new branch never drops the channel sum
        assert a["n"] == b["n"] and a["sq"] == b["sq"] and a["sig"] == b["sig"], op
        assert a["sum"].dtype == b["sum"].dtype and torch.equal(a["sum"], b["sum"]), op
    assert summary_new == _summary_old(old)  # exact float equality of every rel_err / systematic fraction
    print(f"DIAG_BITWISE_OK ops={len(new)} sys_frac={summary_new['systematic_fraction_all_ops']!r}")
