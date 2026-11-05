"""Measure the peak GPU memory of a job (to decide whether it fits the light GPU lane, <= 12 GB).

Runs a python script in-process (runpy) and reports
  * torch.cuda.max_memory_reserved / max_memory_allocated (caching-allocator peak),
  * the peak per-process usage seen by nvidia-smi (includes the CUDA context and non-torch allocations),
sampled every 0.25 s for this pid. With --max_batches N every DataLoader stops after N batches (the peak
is reached in the forward passes; whatever the script does with the truncated results afterwards may
fail and is ignored). Result: one JSON line appended to --log.

Usage: python scripts/gpu_peak.py --tag tiny_grid --max_batches 3 --log results/gpu_peak.jsonl -- \
           scripts/eval_grid.py --prefix /tmp/x --batch_size 64
"""
import argparse
import json
import os
import runpy
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--log", default="results/gpu_peak.jsonl")
    p.add_argument("script")
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = p.parse_args()
    args = a.args[1:] if a.args[:1] == ["--"] else a.args

    import torch
    import torch.utils.data as tud

    truncated = {"hit": False}
    if a.max_batches:
        orig_iter = tud.DataLoader.__iter__

        def capped(self):
            for i, b in enumerate(orig_iter(self)):
                if i >= a.max_batches:
                    truncated["hit"] = True
                    return
                yield b
        tud.DataLoader.__iter__ = capped

    pid, peak_smi, stop = os.getpid(), [0], threading.Event()

    def sample():
        while not stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                for line in out.splitlines():
                    q, m = [x.strip() for x in line.split(",")]
                    if int(q) == pid:
                        peak_smi[0] = max(peak_smi[0], int(m))
            except Exception:
                pass
            stop.wait(0.25)
    th = threading.Thread(target=sample, daemon=True)
    th.start()

    sys.argv = [a.script] + args
    status, t0 = "ok", time.time()
    try:
        runpy.run_path(a.script, run_name="__main__")
    except SystemExit as e:
        status = "ok" if not e.code else f"exit {e.code}"
    except Exception as e:  # noqa: BLE001
        oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e)
        status = "OOM" if oom else (f"after_truncation: {type(e).__name__}" if truncated["hit"] else
                                    f"error: {type(e).__name__}: {e}")
        if not truncated["hit"] or oom:
            traceback.print_exc()
    time.sleep(1.0)
    stop.set()
    th.join()
    gib = 2 ** 30
    rec = {"tag": a.tag, "cmd": " ".join([a.script] + args), "status": status, "max_batches": a.max_batches,
           "torch_max_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 2),
           "torch_max_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
           "nvsmi_peak_gib": round(peak_smi[0] / 1024, 2), "seconds": round(time.time() - t0, 1),
           "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    rec["peak_gib"] = max(rec["nvsmi_peak_gib"], rec["torch_max_reserved_gib"])
    Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    with open(a.log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("GPU_PEAK", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
