"""Phase X3: 5-seed confirmation of the X2 variants (adds seeds 456, 789, 1011 to the X2 seeds 42, 123).

Chosen after X2 seed 42 + partial seed 123, by val-A only:
  kd1_feat      - the only FT trick above ref_I on seed 42;
  bitfit_lr1e-4 - mechanistic question (how much of the FT gain is bias/LN only);
  cbest, chunk128 - FT inside a cheaper arithmetic, compared with B weights deployed there (x2b, 5 seeds).
Run dirs stay in runs/x2/ so that the summary sees all five seeds in one place.
Usage: python scripts/make_jobs_x3.py | python scripts/insert_jobs.py --before x6_
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("x2", Path(__file__).with_name("make_jobs_x2.py"))
src = Path(spec.origin).read_text().split("lines = []")[0]
ns = {}
exec(src, ns)
for seed in (456, 789, 1011):
    for name in ("kd1_feat", "bitfit_lr1e-4", "cbest", "chunk128"):
        print(f"x3_{name}_s{seed}\t{ns['PY']} --run_dir runs/x2/{name}_s{seed} --seed {seed} {ns['BASE']} {ns['VARIANTS'][name]}")
