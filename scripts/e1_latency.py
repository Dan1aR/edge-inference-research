"""End-to-end GPU latency of YOLOS forward under different arithmetics (emulation cost, NOT hardware gain).

Usage: python scripts/e1_latency.py results/e1/latency.json
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment import arith_context, load_model  # noqa: E402

ARITHS = ["fp32", "none", "all", "lin,patch", "all-av"]


@torch.no_grad()
def bench(model, ctrl, arith, batch, n_warm=10, n_rep=50):
    x = torch.randn(batch, 3, 512, 512, device="cuda")
    model.eval()
    for _ in range(n_warm):
        with arith_context(arith, ctrl):
            model(pixel_values=x)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_rep):
        t = time.perf_counter()
        with arith_context(arith, ctrl):
            model(pixel_values=x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    times.sort()
    return {"median_ms": 1e3 * times[len(times) // 2], "p10_ms": 1e3 * times[len(times) // 10],
            "p90_ms": 1e3 * times[9 * len(times) // 10]}


def main():
    out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "results": {}}
    for name in ("tiny", "small"):
        model, _, ctrl = load_model(name)
        for batch in (1, 8):
            for arith in ARITHS:
                r = bench(model, ctrl, arith, batch)
                out["results"][f"{name}/b{batch}/{arith}"] = r
                print(name, batch, arith, f"{r['median_ms']:.2f} ms", flush=True)
        del model
        torch.cuda.empty_cache()
    Path(sys.argv[1]).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
