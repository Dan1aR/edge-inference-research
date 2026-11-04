"""E1: validation of the BF16-accumulation kernels against a sequential reference."""
import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.bf16_accum import bf16_accum_matmul_ref
from src.triton_kernels.bf16acc_core import AccumConfig, bf16acc_matmul, bf16acc_matmul_ste

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DEV = "cuda"

# (M, K, N): reduction lengths that occur in YOLOS-tiny/small at 512x512 + odd tails
SHAPES = [(96, 64, 80), (70, 192, 130), (64, 768, 192), (33, 1125, 64), (50, 37, 19), (65, 384, 72)]
CONFIGS = [AccumConfig(rp, c, m) for rp in (False, True) for (c, m) in
           [(0, "bf16"), (16, "bf16"), (32, "bf16"), (64, "bf16"), (16, "fp32"), (32, "fp32"), (64, "fp32")]]


def _inputs(M, K, N, seed=0, wide=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    a = torch.randn(M, K, device=DEV, generator=g)
    b = torch.randn(K, N, device=DEV, generator=g) * 0.05
    if wide:  # wide dynamic range, like activations with outliers
        a = a * torch.exp(2 * torch.randn(M, K, device=DEV, generator=g))
    return a.to(torch.bfloat16), b.to(torch.bfloat16)


@cuda
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda c: c.describe())
@pytest.mark.parametrize("wide", [False, True])
def test_triton_bit_exact_vs_reference(shape, cfg, wide):
    a, b = _inputs(*shape, wide=wide)
    out = bf16acc_matmul(a, b, cfg)
    ref = bf16_accum_matmul_ref(a, b, round_product=cfg.round_product, chunk=cfg.chunk, chunk_merge=cfg.chunk_merge)
    assert torch.equal(out, ref), f"mismatch rate {(out != ref).float().mean().item():.4f}"


@cuda
@pytest.mark.parametrize("block_k", [8, 16, 32, 64])
@pytest.mark.parametrize("round_product", [False, True])
def test_sequential_independent_of_block_k(block_k, round_product):
    a, b = _inputs(64, 1125, 64, seed=1)
    cfg = AccumConfig(round_product, 0)
    assert torch.equal(bf16acc_matmul(a, b, cfg, block_k=block_k), bf16acc_matmul(a, b, cfg))


@cuda
def test_strided_and_batched_inputs():
    g = torch.Generator(device=DEV).manual_seed(2)
    q = torch.randn(2, 3, 50, 64, device=DEV, generator=g).to(torch.bfloat16)
    k = torch.randn(2, 3, 50, 64, device=DEV, generator=g).to(torch.bfloat16)
    out = bf16acc_matmul(q, k.transpose(-1, -2))
    ref = bf16_accum_matmul_ref(q, k.transpose(-1, -2).contiguous())
    assert torch.equal(out, ref)
    w = torch.randn(130, 70, device=DEV, generator=g).to(torch.bfloat16)  # Linear layout (N, K)
    x = torch.randn(40, 70, device=DEV, generator=g).to(torch.bfloat16)
    assert torch.equal(bf16acc_matmul(x, w.t()), bf16acc_matmul(x, w.t().contiguous()))


@cuda
def test_ste_gradients_equal_exact_matmul_gradients():
    a, b = _inputs(48, 192, 40, seed=3)
    a1, b1 = a.clone().requires_grad_(), b.clone().requires_grad_()
    a2, b2 = a.clone().requires_grad_(), b.clone().requires_grad_()
    a3, b3 = a.clone().requires_grad_(), b.clone().requires_grad_()
    g = torch.randn(48, 40, device=DEV).to(torch.bfloat16)
    bf16acc_matmul_ste(a1, b1).backward(g)
    (a2 @ b2).backward(g)
    # explicit straight-through formulation: y = a@b + sg(BF16acc(a,b) - a@b)
    y3 = a3 @ b3
    (y3 + (bf16acc_matmul(a3.detach(), b3.detach()) - y3).detach()).backward(g)
    for x1, x2, x3 in ((a1, a2, a3), (b1, b2, b3)):
        assert torch.equal(x1.grad, x2.grad) and torch.equal(x1.grad, x3.grad)
    # forward of the STE op is the bf16-acc result, not the exact matmul
    assert torch.equal(bf16acc_matmul_ste(a, b), bf16acc_matmul(a, b))


@pytest.fixture(scope="module")
def yolos_pair():
    from transformers import YolosForObjectDetection

    from src.target_arith import install_target_arithmetic

    ref = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny", attn_implementation="eager").to(DEV).eval()
    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny").to(DEV).eval()
    ctrl = install_target_arithmetic(model)
    return ref, model, ctrl


@cuda
def test_wrapped_model_equals_hf_when_disabled(yolos_pair):
    ref, model, ctrl = yolos_pair
    x = torch.randn(2, 3, 256, 320, device=DEV)
    ctrl.set(())
    with torch.no_grad():
        for autocast in (False, True):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                o_ref, o = ref(pixel_values=x), model(pixel_values=x)
            assert torch.equal(o_ref.logits, o.logits) and torch.equal(o_ref.pred_boxes, o.pred_boxes)


@cuda
def test_components_change_output_and_patch_gets_gradient(yolos_pair):
    from src.target_arith import COMPONENTS

    _, model, ctrl = yolos_pair
    x = torch.randn(2, 3, 256, 256, device=DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ctrl.set(())
        base = model(pixel_values=x).logits
        for comp in COMPONENTS:
            ctrl.set({comp})
            assert not torch.equal(model(pixel_values=x).logits, base), comp
    ctrl.set(set(COMPONENTS))
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(pixel_values=x)
        (out.logits.float().pow(2).mean() + out.pred_boxes.float().sum()).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and p.dtype == torch.float32, name
    proj = model.vit.embeddings.patch_embeddings.projection
    assert proj.weight.grad.abs().sum() > 0
    ctrl.set(())


@cuda
def test_patch_conv_matches_reference(yolos_pair):
    _, model, ctrl = yolos_pair
    proj = model.vit.embeddings.patch_embeddings.projection
    x = torch.randn(1, 3, 64, 96, device=DEV)
    ctrl.set({"patch"})
    with torch.no_grad():
        y = proj(x)
    cols = F.unfold(x.to(torch.bfloat16), 16, stride=16).transpose(1, 2).reshape(-1, 768)
    w = proj.weight.to(torch.bfloat16).reshape(proj.weight.shape[0], -1)
    ref = bf16_accum_matmul_ref(cols, w.t().contiguous()) + proj.bias.to(torch.bfloat16)
    assert torch.equal(y.flatten(2).transpose(1, 2).reshape(-1, w.shape[0]), ref)
    ctrl.set(())


@cuda
@pytest.mark.parametrize("cfg", [AccumConfig(True, 0), AccumConfig(False, 32, "bf16"), AccumConfig(True, 16, "fp32")],
                         ids=lambda c: c.describe())
def test_batched_with_tails_all_modes(cfg):
    g = torch.Generator(device=DEV).manual_seed(5)
    a = torch.randn(3, 2, 67, 100, device=DEV, generator=g).to(torch.bfloat16)
    b = torch.randn(3, 2, 100, 71, device=DEV, generator=g).to(torch.bfloat16)
    ref = bf16_accum_matmul_ref(a, b, round_product=cfg.round_product, chunk=cfg.chunk, chunk_merge=cfg.chunk_merge)
    assert torch.equal(bf16acc_matmul(a, b, cfg), ref)


@cuda
def test_acc_linear_with_bias_matches_reference(yolos_pair):
    _, model, ctrl = yolos_pair
    lin = model.vit.encoder.layer[0].intermediate.dense
    x = torch.randn(2, 37, lin.in_features, device=DEV)
    ctrl.set({"mlp"})
    with torch.no_grad():
        y = lin(x)
    w = lin.weight.to(torch.bfloat16)
    ref = bf16_accum_matmul_ref(x.to(torch.bfloat16).reshape(-1, lin.in_features), w.t().contiguous())
    ref = (ref + lin.bias.to(torch.bfloat16)).reshape(2, 37, -1)
    assert torch.equal(y, ref)
    ctrl.set(())


@cuda
def test_attention_with_qk_av_matches_reference(yolos_pair):
    _, model, ctrl = yolos_pair
    att = model.vit.encoder.layer[0].attention.attention
    h = torch.randn(2, 41, 192, device=DEV)
    ctrl.set({"qk", "av"})
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ctx, _ = att(h)
        shape = (2, -1, att.num_attention_heads, att.attention_head_size)
        q = att.query(h).view(*shape).transpose(1, 2)
        k = att.key(h).view(*shape).transpose(1, 2)
        v = att.value(h).view(*shape).transpose(1, 2)
    scores = bf16_accum_matmul_ref(q, k.transpose(-1, -2).contiguous()) * att.scaling
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
    ref = bf16_accum_matmul_ref(probs, v).transpose(1, 2).reshape(2, 41, -1)
    assert torch.equal(ctx, ref)
    ctrl.set(())


# --- extension: accumulator formats, Kahan, long chunks -------------------------------------------
FMT_CONFIGS = [AccumConfig(rp, c, m, fmt) for fmt in ("e8m10", "e8m13", "e8m16", "fp32", "fp16")
               for rp in (False, True) for (c, m) in [(0, "acc"), (32, "acc"), (128, "fp32")]]
FMT_CONFIGS += [AccumConfig(False, c, m) for (c, m) in [(8, "acc"), (128, "acc"), (256, "fp32")]]
FMT_CONFIGS += [AccumConfig(False, 0, "acc", fmt, kahan=True) for fmt in ("bf16", "e8m13", "fp16")]


@cuda
@pytest.mark.parametrize("shape", [(70, 192, 130), (33, 1125, 64), (50, 37, 19)])
@pytest.mark.parametrize("cfg", FMT_CONFIGS, ids=lambda c: c.describe())
def test_formats_bit_exact_vs_reference(shape, cfg):
    a, b = _inputs(*shape, wide=True)
    out = bf16acc_matmul(a, b, cfg)
    ref = bf16_accum_matmul_ref(a, b, round_product=cfg.round_product, chunk=cfg.chunk, chunk_merge=cfg.chunk_merge,
                                fmt=cfg.fmt, kahan=cfg.kahan)
    assert torch.equal(out, ref), f"mismatch rate {(out != ref).float().mean().item():.4f}"


@cuda
def test_fp16_accumulator_overflows_to_inf():
    a = torch.full((4, 64), 64.0, device=DEV, dtype=torch.bfloat16)
    b = torch.full((64, 4), 32.0, device=DEV, dtype=torch.bfloat16)  # sum = 131072 > 65504
    assert torch.isinf(bf16acc_matmul(a, b, AccumConfig(fmt="fp16"))).all()
    assert torch.isfinite(bf16acc_matmul(a, b, AccumConfig(fmt="bf16"))).all()


@cuda
def test_error_decreases_with_mantissa_and_kahan():
    a, b = _inputs(64, 1125, 64, seed=7)
    a = a.abs()  # same-sign terms: swamping
    exact = a.double() @ b.abs().double()
    err = lambda cfg: ((bf16acc_matmul(a, b.abs(), cfg).double() - exact).norm() / exact.norm()).item()  # noqa: E731
    e = [err(AccumConfig(fmt=f)) for f in ("bf16", "e8m10", "e8m13", "fp32")]
    assert e[0] > e[1] > e[2] >= e[3]
    assert err(AccumConfig(kahan=True)) < 0.2 * e[0]


@cuda
def test_stochastic_rounding_is_unbiased():
    a, b = _inputs(64, 1125, 64, seed=8)
    a, b = a.abs(), b.abs()
    exact = a.double() @ b.double()
    rne = bf16acc_matmul(a, b, AccumConfig()).double()
    srs = torch.stack([bf16acc_matmul(a, b, AccumConfig(rounding="sr")).double() for _ in range(8)])
    bias_rne = ((rne - exact) / exact).mean().abs().item()
    bias_sr = ((srs.mean(0) - exact) / exact).mean().abs().item()
    assert bias_sr < 0.1 * bias_rne
    assert not torch.equal(srs[0], srs[1])  # different random streams per call
