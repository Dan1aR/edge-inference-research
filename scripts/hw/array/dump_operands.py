#!/usr/bin/env python
"""E7 / C: dump real bf16 matmul operands of YOLOS-tiny (CPU, batch 1, eager attention, 512x512).

The model runs as in the experiments' FP32-acc baseline (torch.autocast bf16 on CPU; matmul operands
are bf16, softmax in fp32 then cast to bf16).  For a few COCO val2017 images we store, as bf16 bit
patterns (uint16), the two operands A (M x K) and B (K x N) of
  patch : patch-embedding conv as im2col GEMM  (1024 x 768) @ (768 x 192)
  qkv   : [query|key|value] projections of layer L (1125 x 192) @ (192 x 576)
  fc1   : MLP intermediate.dense of layer L    (1125 x 192) @ (192 x 768)
  fc2   : MLP output.dense of layer L          (1125 x 768) @ (768 x 192)
  qk    : Q K^T of layer L, head h              (1125 x 64)  @ (64 x 1125)
  av    : softmax(QK^T/sqrt(d)) V, layer L, h   (1125 x 1125) @ (1125 x 64)
(uint16 bit patterns; ~10 MB in total).

  CUDA_VISIBLE_DEVICES= nice -n 19 .venv/bin/python scripts/hw/array/dump_operands.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "hw" / "array" / "operands"


def bits(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hustvl/yolos-tiny")
    ap.add_argument("--images", type=int, default=2)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--head", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    torch.set_num_threads(2)
    from PIL import Image
    from transformers import YolosForObjectDetection, YolosImageProcessor
    from transformers.models.yolos.modeling_yolos import YolosSelfAttention

    from src.config import FIXED_IMAGE_SIZE

    proc = YolosImageProcessor.from_pretrained(args.model)
    model = YolosForObjectDetection.from_pretrained(args.model, attn_implementation="eager").eval()
    imgs = sorted((REPO / "coco" / "val2017").glob("*.jpg"))[: args.images]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    L, h = args.layer, args.head
    layer = model.vit.encoder.layer[L]
    att: YolosSelfAttention = layer.attention.attention
    cap: dict = {}

    def lin_hook(name):
        def f(mod, inp, outp):
            cap[name] = (inp[0][0].to(torch.bfloat16), mod.weight.t().to(torch.bfloat16), outp[0])
        return f

    hooks = [layer.intermediate.dense.register_forward_hook(lin_hook("fc1")),
             layer.output.dense.register_forward_hook(lin_hook("fc2"))]

    orig_forward = att.forward

    def att_forward(hidden_states, head_mask=None, output_attentions=False, **kw):
        x = hidden_states[0].to(torch.bfloat16)
        wqkv = torch.cat([att.query.weight, att.key.weight, att.value.weight], 0).t().to(torch.bfloat16)
        cap["qkv"] = (x, wqkv, None)
        B = hidden_states.shape[0]
        shp = (B, -1, att.num_attention_heads, att.attention_head_size)
        q = att.query(hidden_states).view(*shp).transpose(1, 2)
        k = att.key(hidden_states).view(*shp).transpose(1, 2)
        v = att.value(hidden_states).view(*shp).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / (att.attention_head_size ** 0.5)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        ctx = torch.matmul(probs, v)
        cap["qk"] = (q[0, h].to(torch.bfloat16), k[0, h].t().to(torch.bfloat16), None)
        cap["av"] = (probs[0, h].to(torch.bfloat16), v[0, h].to(torch.bfloat16), None)
        ctx = ctx.transpose(1, 2).contiguous().reshape(B, -1, att.all_head_size)
        return ctx, (probs if output_attentions else None)

    att.forward = att_forward
    proj = model.vit.embeddings.patch_embeddings.projection
    meta = {}
    for i, path in enumerate(imgs):
        img = Image.open(path).convert("RGB")
        px = proc(images=img, return_tensors="pt", size=FIXED_IMAGE_SIZE)["pixel_values"]
        assert px.shape[-2:] == (512, 512), px.shape
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            model(pixel_values=px)
            cols = F.unfold(px.to(torch.bfloat16), kernel_size=16, stride=16)[0].t()   # (1024, 768)
            wp = proj.weight.to(torch.bfloat16).reshape(proj.out_channels, -1).t()   # (768, 192)
        cap["patch"] = (cols, wp, None)
        for name, (A, Bm, _) in cap.items():
            tag = f"{name}_L{L}" + (f"h{h}" if name in ("qk", "av") else "") if name != "patch" else name
            fn = out / f"{tag}_img{i}.npz"
            np.savez_compressed(fn, A=bits(A), B=bits(Bm))
            meta[fn.name] = dict(image=path.name, A=list(A.shape), B=list(Bm.shape),
                                 A_absmax=float(A.float().abs().max()), B_absmax=float(Bm.float().abs().max()),
                                 A_zero_frac=float((A == 0).float().mean()), B_zero_frac=float((Bm == 0).float().mean()))
            print(fn.name, meta[fn.name], flush=True)
    for hk in hooks:
        hk.remove()
    att.forward = orig_forward
    (out / "meta.json").write_text(json.dumps(dict(model=args.model, layer=L, head=h, input="512x512 YolosImageProcessor",
                                                   autocast="cpu bfloat16, eager attention", files=meta), indent=2))


if __name__ == "__main__":
    main()
