"""The model-agnostic wrapper (src.generic_arith) must reproduce src.target_arith bit for bit on YOLOS,
and centering must be an exact identity up to the extra operand rounding."""
import pathlib
import sys

import pytest
import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src import generic_arith as ga
from src.target_arith import COMPONENTS as LEGACY_COMPONENTS
from src.triton_kernels.bf16acc_core import set_accum_config

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
DEV = "cuda"


@pytest.fixture(scope="module")
def pair():
    from src.experiment import load_model

    legacy, _, lctrl = load_model("tiny", impl="legacy")
    generic, _, gctrl = load_model("tiny", impl="generic")
    return legacy.eval(), lctrl, generic.eval(), gctrl


@cuda
@pytest.mark.parametrize("spec", ["none", "patch", "qkv", "attn_out", "mlp", "head", "qk", "av", "all", "lin,patch"])
def test_generic_equals_legacy_on_yolos(pair, spec):
    from src.target_arith import parse_components as legacy_parse

    legacy, lctrl, generic, gctrl = pair
    x = torch.randn(2, 3, 256, 320, device=DEV, generator=torch.Generator(device=DEV).manual_seed(0))
    lctrl.set(legacy_parse(spec))
    gctrl.set(ga.parse_components(spec))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a, b = legacy(pixel_values=x), generic(pixel_values=x)
    assert torch.equal(a.logits, b.logits) and torch.equal(a.pred_boxes, b.pred_boxes)
    lctrl.set(())
    gctrl.set(())


@cuda
def test_generic_gradients_equal_legacy(pair):
    legacy, lctrl, generic, gctrl = pair
    x = torch.randn(2, 3, 256, 256, device=DEV, generator=torch.Generator(device=DEV).manual_seed(1))
    grads = []
    for model, ctrl, comps in ((legacy, lctrl, set(LEGACY_COMPONENTS)), (generic, gctrl, ga.parse_components("all"))):
        ctrl.set(comps)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(pixel_values=x)
            (out.logits.float().pow(2).mean() + out.pred_boxes.float().sum()).backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
        ctrl.set(())
    assert grads[0].keys() == grads[1].keys()
    for n in grads[0]:
        assert torch.equal(grads[0][n], grads[1][n]), n


@cuda
def test_record_counts_macs_and_no_unhandled(pair):
    _, _, generic, gctrl = pair
    gctrl.record, gctrl.ops = True, {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        generic(pixel_values=torch.randn(1, 3, 512, 512, device=DEV))
    gctrl.record = False
    total = sum(s.macs for s in gctrl.ops.values())
    assert total == 11_968_838_400 or abs(total / 11.97e9 - 1) < 0.01, total  # E6: 11.97 GMAC
    assert not gctrl.unhandled
    comps = {s.component for s in gctrl.ops.values()}
    assert comps == set(LEGACY_COMPONENTS)


@cuda
@pytest.mark.parametrize("mode", ["a", "b"])
def test_centering_is_identity_in_fp32(mode):
    g = torch.Generator(device=DEV).manual_seed(3)
    a = (torch.rand(4, 50, 300, device=DEV, generator=g) + 0.5).to(torch.bfloat16)  # positive, like softmax / pixels
    b = (torch.randn(4, 300, 40, device=DEV, generator=g) + 1.0).to(torch.bfloat16)
    set_accum_config(fmt="fp32")
    try:
        exact = a.double() @ b.double()
        y = ga._centered_matmul(a, b, mode).double()
        y0 = ga._centered_matmul(a, b, None).double()
    finally:
        set_accum_config()
    # with an fp32 accumulator both are bf16-rounded results of the same product
    assert ((y - exact).abs() / exact.abs()).max() < 1e-2
    assert ((y0 - exact).abs() / exact.abs()).max() < 1e-2


@cuda
def test_centering_reduces_bf16acc_error_for_positive_terms():
    g = torch.Generator(device=DEV).manual_seed(4)
    p = torch.softmax(torch.randn(8, 128, 1125, device=DEV, generator=g), -1).to(torch.bfloat16)
    v = (torch.randn(8, 1125, 64, device=DEV, generator=g) + 2.0).to(torch.bfloat16)
    exact = p.double() @ v.double()
    err = lambda y: ((y.double() - exact).norm() / exact.norm()).item()  # noqa: E731
    assert err(ga._centered_matmul(p, v, "b")) < 0.3 * err(ga._centered_matmul(p, v, None))
