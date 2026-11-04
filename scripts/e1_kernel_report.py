"""E1: numerical characterisation of BF16 accumulation (synthetic K-sweep + real YOLOS activations).

Outputs results/e1/kernel_report.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import eval_loader, load_dataset, load_model  # noqa: E402
from src.target_arith import AccLinear, AccPatchConv  # noqa: E402
from src.triton_kernels.bf16acc_core import AccumConfig, bf16acc_matmul  # noqa: E402

DEV = "cuda"
U_BF16 = 2.0 ** -8
VARIANTS = {
    "fma_seq": AccumConfig(False, 0),
    "roundprod_seq": AccumConfig(True, 0),
    "fma_chunk32_bf16": AccumConfig(False, 32, "bf16"),
    "fma_chunk32_fp32": AccumConfig(False, 32, "fp32"),
}


def rel_err(y: torch.Tensor, exact: torch.Tensor) -> float:
    return ((y.double() - exact).norm() / exact.norm()).item()


def k_sweep():
    rows = []
    g = torch.Generator(device=DEV).manual_seed(0)
    for K in [16, 32, 64, 128, 192, 384, 768, 1125, 1536, 3072]:
        for dist in ("gauss", "positive"):  # positive = softmax-like / no cancellation
            a = torch.randn(256, K, device=DEV, generator=g)
            b = torch.randn(K, 256, device=DEV, generator=g)
            if dist == "positive":
                a, b = a.abs(), b.abs()
            a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
            exact = a.double() @ b.double()
            row = {"K": K, "dist": dist, "bf16_fp32acc": rel_err(a @ b, exact),
                   "bound_ku": K * U_BF16 / (1 - K * U_BF16) if K * U_BF16 < 1 else None,
                   "prob_sqrtk_u": float(np.sqrt(K) * U_BF16)}
            for name, cfg in VARIANTS.items():
                row[name] = rel_err(bf16acc_matmul(a, b, cfg), exact)
            rows.append(row)
    return rows


def real_activations(n_images: int = 16):
    """Per-component relative error of BF16-acc vs FP32-acc outputs on identical bf16 inputs."""
    model, processor, ctrl = load_model("tiny")
    model.eval()
    ds = torch.utils.data.Subset(load_dataset("coco_val"), range(n_images))
    batch = next(iter(eval_loader(ds, processor, n_images)))
    records = []

    def lin_hook(mod, inp, out):
        x = inp[0].to(torch.bfloat16).reshape(-1, mod.in_features)
        w = mod.weight.to(torch.bfloat16)
        records.append((mod.component, mod.in_features, x, w.t()))

    def conv_hook(mod, inp, out):
        cols = F.unfold(inp[0].to(torch.bfloat16), mod.kernel_size, stride=mod.stride).transpose(1, 2).reshape(-1, 768)
        records.append(("patch", 768, cols, mod.weight.to(torch.bfloat16).reshape(mod.out_channels, -1).t()))

    hooks = []
    for m in model.modules():
        if isinstance(m, AccLinear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, AccPatchConv):
            hooks.append(m.register_forward_hook(conv_hook))
    orig_matmul = torch.matmul

    def rec_matmul(a, b):
        if a.dim() == 4:
            comp = "av" if a.shape[-1] == a.shape[-2] else "qk"
            records.append((comp, a.shape[-1], a.to(torch.bfloat16), b.to(torch.bfloat16)))
        return orig_matmul(a, b)

    torch.matmul = rec_matmul
    try:
        ctrl.set(())
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(pixel_values=batch["pixel_values"].to(DEV))
    finally:
        torch.matmul = orig_matmul
        for h in hooks:
            h.remove()

    out = {}
    for comp, K, a, b in records:
        exact = orig_matmul(a.double(), b.double())
        e = out.setdefault(comp, {"K": int(K), "n_ops": 0, "bf16_fp32acc": [], "fma_seq": [], "roundprod_seq": [],
                                  "fma_chunk32_fp32": []})
        e["n_ops"] += 1
        e["bf16_fp32acc"].append(rel_err(orig_matmul(a, b), exact))
        for name in ("fma_seq", "roundprod_seq", "fma_chunk32_fp32"):
            e[name].append(rel_err(bf16acc_matmul(a.contiguous(), b.contiguous(), VARIANTS[name]), exact))
    for e in out.values():
        for k in ("bf16_fp32acc", "fma_seq", "roundprod_seq", "fma_chunk32_fp32"):
            e[k] = float(np.mean(e[k]))
    return out


def throughput():
    """Emulation cost (informational): Triton BF16-acc kernel vs cuBLAS BF16 on YOLOS-tiny shapes, batch 64."""
    shapes = {"qkv/attn_out (192->192)": (64 * 1125, 192, 192), "mlp up (192->768)": (64 * 1125, 192, 768),
              "mlp down (768->192)": (64 * 1125, 768, 192), "qk (bmm)": ((64 * 3, 1125, 64), 1125),
              "av (bmm)": ((64 * 3, 1125, 1125), 64)}
    res = {}
    for name, s in shapes.items():
        if isinstance(s[0], tuple):
            (B, M, K), N = s
            a = torch.randn(B, M, K, device=DEV).to(torch.bfloat16)
            b = torch.randn(B, K, N, device=DEV).to(torch.bfloat16)
        else:
            M, K, N = s
            a = torch.randn(M, K, device=DEV).to(torch.bfloat16)
            b = torch.randn(K, N, device=DEV).to(torch.bfloat16)
        times = {}
        for tag, fn in (("cublas_bf16", lambda: a @ b), ("triton_bf16acc", lambda: bf16acc_matmul(a, b))):
            fn()
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            times[tag] = (time.time() - t) / 5 * 1e3
        times["slowdown"] = times["triton_bf16acc"] / times["cublas_bf16"]
        res[name] = times
    return res


def main():
    out = {"gpu": torch.cuda.get_device_name(), "k_sweep": k_sweep(), "real_activations": real_activations(),
           "throughput_ms": throughput()}
    path = ROOT / "results/e1/kernel_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["real_activations"], indent=1))


if __name__ == "__main__":
    main()
