"""Phase X2 (block D): screening of fine-tuning variants on YOLOS-tiny, 2 seeds, selection by val-A only.

All runs: train2017, 300 steps, batch 64, target = full BF16-acc ("all"), LR 1e-5 unless stated
(reference: runs/e8/lr1e-5_{B,I,M1}_s* with 5 seeds). The winners are re-run with 5 seeds (phase X3).
"""
PY = "NUM_WORKERS=6 .venv/bin/python train.py --val_datasets coco_val"
BASE = "--schedule all@0 --target all"
VARIANTS = {
    "I_lr5e-6": "--lr 5e-6",
    "I_lr2e-5": "--lr 2e-5",
    "I_ema99": "--lr 1e-5 --ema 0.99",
    "bitfit_lr1e-4": "--lr 1e-4 --trainable bitfit",
    "bitfit_lr1e-3": "--lr 1e-3 --trainable bitfit",
    "kd1": "--lr 1e-5 --kd 1",
    "kd5": "--lr 1e-5 --kd 5",
    "kdonly": "--lr 1e-5 --kd 1 --kd_only",
    "kd1_feat": "--lr 1e-5 --kd 1 --kd_feat 1",
    "sr": "--lr 1e-5 --train_rounding sr",
    "cbest": '--lr 1e-5 --center "av:b,patch:a,qk:a"',
    "chunk128": "--lr 1e-5 --chunk 128 --chunk_merge fp32",
    "e8m13": "--lr 1e-5 --fmt e8m13",
}
lines = []
for seed in (42, 123):
    for name, args in VARIANTS.items():
        job = f"x2_{name}_s{seed}"
        lines.append(f"{job}\t{PY} --run_dir runs/x2/{name}_s{seed} --seed {seed} {BASE} {args}")
# B weights (LR 1e-5, 5 seeds) deployed in the new arithmetics: the "same arithmetic" control
for seed in (42, 123, 456, 789, 1011):
    for name, args in (("cbest", '--arith all --center "av:b,patch:a,qk:a"'),
                       ("chunk128", "--arith all --chunk 128 --chunk_merge fp32"), ("e8m13", "--arith all --fmt e8m13")):
        lines.append(f"x2b_B_s{seed}_{name}\tNUM_WORKERS=6 .venv/bin/python eval.py --ckpt runs/e8/lr1e-5_B_s{seed}/final.pt "
                     f"{args} --out results/x2b/B_s{seed}_{name}.json")
with open("queue/jobs.tsv", "a") as f:
    f.write("\n".join(lines) + "\n")
print(len(lines), "jobs")
