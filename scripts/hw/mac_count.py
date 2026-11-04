#!/usr/bin/env python
"""E6 / step 1: MAC accounting for YOLOS (tiny / small) at a fixed input size.

For every matmul-like component we report
  * macs     -- number of multiply-accumulate operations,
  * outputs  -- number of accumulator results (output elements of the matmul),
  * K        -- reduction length (number of products summed into one output).

Components (match the bf16-accum switches of src/bf16_accum.py):
  patch    : Conv2d patch embedding (im2col GEMM, K = C*P*P)          -> "patch" flag
  qkv      : query/key/value nn.Linear                                 -> "lin"   flag
  attn_out : attention output dense nn.Linear                          -> "lin"   flag
  mlp      : intermediate.dense + output.dense nn.Linear               -> "lin"   flag
  head     : class_labels_classifier + bbox_predictor MLPs (nn.Linear) -> "lin"   flag
  qk       : Q @ K^T  (K = head_dim)                                   -> "attn"  flag
  av       : softmax(QK^T) @ V  (K = seq_len)                          -> "attn"  flag

Numbers are computed (a) analytically from the HF config and (b) by instrumenting a
real CPU forward pass (TorchDispatchMode that records every aten mm/addmm/bmm/matmul/
convolution together with the innermost nn.Module that issued it).  A second,
independent check uses torch.utils.flop_counter.FlopCounterMode (total FLOPs / 2).
The script asserts that all three agree exactly.

Run (CPU only):
  CUDA_VISIBLE_DEVICES="" nice -n 19 .venv/bin/python scripts/hw/mac_count.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

COMPONENTS = ["patch", "qkv", "attn_out", "mlp", "head", "qk", "av"]
REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------
# (a) analytic count
# --------------------------------------------------------------------------------------
def analytic(cfg, H: int, W: int, batch: int = 1) -> dict:
    d = cfg.hidden_size
    L = cfg.num_hidden_layers
    nh = cfg.num_attention_heads
    dh = d // nh
    ff = cfg.intermediate_size
    p = cfg.patch_size
    C = cfg.num_channels
    n_patch = (H // p) * (W // p)
    n_det = cfg.num_detection_tokens
    N = 1 + n_patch + n_det  # CLS + patches + detection tokens
    n_cls_out = cfg.num_labels + 1  # + "no object"

    comp = {}
    # patch embedding: conv k=s=p  == GEMM (n_patch x C*p*p) @ (C*p*p x d)
    Kp = C * p * p
    comp["patch"] = dict(macs=n_patch * d * Kp, outputs=n_patch * d, K=[Kp], gemms=1)
    comp["qkv"] = dict(macs=L * 3 * N * d * d, outputs=L * 3 * N * d, K=[d], gemms=3 * L)
    comp["attn_out"] = dict(macs=L * N * d * d, outputs=L * N * d, K=[d], gemms=L)
    comp["mlp"] = dict(
        macs=L * (N * d * ff + N * ff * d), outputs=L * (N * ff + N * d), K=[d, ff], gemms=2 * L
    )
    comp["qk"] = dict(macs=L * nh * N * N * dh, outputs=L * nh * N * N, K=[dh], gemms=L)
    comp["av"] = dict(macs=L * nh * N * N * dh, outputs=L * nh * N * dh, K=[N], gemms=L)
    # heads on the n_det detection tokens: 3-layer MLPs d->d->d->(classes+1) and d->d->d->4
    hm = n_det * (d * d + d * d + d * n_cls_out) + n_det * (d * d + d * d + d * 4)
    ho = n_det * (d + d + n_cls_out) + n_det * (d + d + 4)
    comp["head"] = dict(macs=hm, outputs=ho, K=[d], gemms=6)
    for v in comp.values():
        v["macs"] *= batch
        v["outputs"] *= batch
    geom = dict(
        H=H, W=W, batch=batch, hidden=d, layers=L, heads=nh, head_dim=dh, intermediate=ff,
        patch=p, n_patch=n_patch, n_det=n_det, seq_len=N, n_class_logits=n_cls_out,
        use_mid_position_embeddings=bool(getattr(cfg, "use_mid_position_embeddings", False)),
    )
    return dict(geometry=geom, components=comp)


# --------------------------------------------------------------------------------------
# (b) instrumentation
# --------------------------------------------------------------------------------------
def classify(module_name: str, out_shape, k: int, seq_len: int) -> str:
    n = module_name
    if "patch_embeddings" in n:
        return "patch"
    if n.endswith(("attention.attention.query", "attention.attention.key", "attention.attention.value")):
        return "qkv"
    if n.endswith("attention.output.dense"):
        return "attn_out"
    if n.endswith(("intermediate.dense", ".output.dense")):
        return "mlp"
    if "class_labels_classifier" in n or "bbox_predictor" in n:
        return "head"
    if n.endswith("attention.attention"):  # matmuls issued inside YolosSelfAttention itself
        # QK^T: output (..., N, N), reduction over head_dim; AV: output (..., N, dh), reduction N
        return "av" if k == seq_len else "qk"
    return "other:" + n


class MatmulRecorder(TorchDispatchMode):
    """Records every GEMM-like aten op with (#MACs, #outputs, K) and issuing module."""

    def __init__(self, stack, seq_len):
        super().__init__()
        self.stack = stack
        self.seq_len = seq_len
        self.records = []

    def _rec(self, opname, macs, outputs, k, out_shape):
        mod = self.stack[-1] if self.stack else "<root>"
        self.records.append(
            dict(op=opname, module=mod, macs=int(macs), outputs=int(outputs), K=int(k),
                 out_shape=list(out_shape), comp=classify(mod, out_shape, k, self.seq_len))
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func.overloadpacket.__name__
        if name in ("mm", "addmm", "bmm", "baddbmm"):
            a = args[1] if name in ("addmm", "baddbmm") else args[0]
            k = a.shape[-1]
            self._rec(name, out.numel() * k, out.numel(), k, out.shape)
        elif name == "convolution":
            w = args[1]
            k = w.shape[1] * math.prod(w.shape[2:])  # Cin/groups * kh * kw
            self._rec(name, out.numel() * k, out.numel(), k, out.shape)
        elif name in ("_scaled_dot_product_attention_math", "_scaled_dot_product_flash_attention_for_cpu",
                      "_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention"):
            raise RuntimeError("fused SDPA kernel hit - run with attn_implementation='eager'")
        return out


def instrumented(model_id: str, H: int, W: int, batch: int, seq_len: int, attn_impl: str):
    from transformers import YolosForObjectDetection

    model = YolosForObjectDetection.from_pretrained(model_id, attn_implementation=attn_impl).eval()
    stack = []
    handles = []
    for name, m in model.named_modules():
        handles.append(m.register_forward_pre_hook(lambda mod, inp, n=name: stack.append(n)))
        handles.append(m.register_forward_hook(lambda mod, inp, out: (stack.pop(), None)[1]))
    x = torch.randn(batch, 3, H, W)
    rec = MatmulRecorder(stack, seq_len)
    with torch.no_grad(), rec:
        out = model(pixel_values=x)
    for h in handles:
        h.remove()
    assert out.logits.shape[1] == model.config.num_detection_tokens
    # second independent check: FlopCounterMode (counts 2 FLOPs per MAC)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(pixel_values=x)
    flops_total = fc.get_total_flops()
    agg = defaultdict(lambda: dict(macs=0, outputs=0, K=set(), gemm_calls=0))
    for r in rec.records:
        a = agg[r["comp"]]
        a["macs"] += r["macs"]
        a["outputs"] += r["outputs"]
        a["K"].add(r["K"])
        a["gemm_calls"] += 1
    agg = {c: dict(macs=v["macs"], outputs=v["outputs"], K=sorted(v["K"]), gemm_calls=v["gemm_calls"])
           for c, v in agg.items()}
    return agg, flops_total, len(rec.records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["hustvl/yolos-tiny", "hustvl/yolos-small"])
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default=str(REPO / "results/hw/mac_counts.json"))
    args = ap.parse_args()
    torch.set_num_threads(2)

    from transformers import AutoConfig

    result = dict(input=dict(H=args.height, W=args.width, batch=args.batch), models={})
    for mid in args.models:
        cfg = AutoConfig.from_pretrained(mid)
        an = analytic(cfg, args.height, args.width, args.batch)
        N = an["geometry"]["seq_len"]
        meas, flops_eager, ncalls = instrumented(mid, args.height, args.width, args.batch, N, "eager")
        # FlopCounter on the default (sdpa) path as well -- independent of our dispatch hook.
        # The fused CPU flash-SDPA kernel is not registered in FlopCounterMode, so force the
        # MATH backend (decomposes into bmm, numerically the same graph).
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import YolosForObjectDetection
        m_sdpa = YolosForObjectDetection.from_pretrained(mid, attn_implementation="sdpa").eval()
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
            m_sdpa(pixel_values=torch.randn(args.batch, 3, args.height, args.width))
        flops_sdpa = fc.get_total_flops()
        del m_sdpa

        tot_an = sum(v["macs"] for v in an["components"].values())
        tot_me = sum(v["macs"] for v in meas.values())
        mismatch = []
        for c in COMPONENTS:
            a, m = an["components"][c], meas.get(c)
            if m is None or a["macs"] != m["macs"] or a["outputs"] != m["outputs"] or sorted(a["K"]) != m["K"]:
                mismatch.append((c, a, m))
        others = [c for c in meas if c not in COMPONENTS]
        ok = (not mismatch and not others and tot_an == tot_me
              and flops_eager == 2 * tot_an and flops_sdpa == 2 * tot_an)
        print(f"== {mid}  seq_len={N}  gemm calls={ncalls}")
        print(f"{'comp':9s} {'MACs':>15s} {'share':>7s} {'outputs':>12s} {'share':>7s}  K")
        tot_out = sum(v["outputs"] for v in an["components"].values())
        for c in COMPONENTS:
            v = an["components"][c]
            print(f"{c:9s} {v['macs']:15,d} {100*v['macs']/tot_an:6.2f}% {v['outputs']:12,d} "
                  f"{100*v['outputs']/tot_out:6.2f}%  {v['K']}")
        print(f"total     {tot_an:15,d}          {tot_out:12,d}")
        print(f"instrumented total MACs={tot_me:,d}; FlopCounter eager/2={flops_eager//2:,d}; "
              f"sdpa/2={flops_sdpa//2:,d}; AGREE={ok}")
        if not ok:
            print("MISMATCH:", mismatch, others)
        result["models"][mid] = dict(
            config=dict(hidden_size=cfg.hidden_size, num_hidden_layers=cfg.num_hidden_layers,
                        num_attention_heads=cfg.num_attention_heads, intermediate_size=cfg.intermediate_size,
                        num_detection_tokens=cfg.num_detection_tokens, patch_size=cfg.patch_size,
                        num_labels=cfg.num_labels),
            geometry=an["geometry"],
            analytic=an["components"],
            instrumented=meas,
            totals=dict(macs=tot_an, outputs=tot_out, instrumented_macs=tot_me,
                        flopcounter_eager_flops=flops_eager, flopcounter_sdpa_flops=flops_sdpa),
            agree=ok,
            not_counted=("bias adds, LayerNorm, GELU/ReLU, softmax, 1/sqrt(dh) scaling, residual adds, "
                         "bicubic interpolation of (mid-)position embeddings (input-size dependent "
                         "constant; precomputable for fixed 512x512)"),
        )
        assert ok, f"analytic and instrumented counts disagree for {mid}"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
