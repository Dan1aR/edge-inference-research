"""Phase X1 of the extension: inference sweeps (accumulator formats, chunking, Kahan/SR, centering,
real tensor-core arithmetic) -> appends jobs to queue/jobs.tsv."""
import sys

PY = "NUM_WORKERS=6 .venv/bin/python eval.py"
CENTER_BEST = "av:b,patch:a,qk:a"


def configs(model):
    c = [("none", "--arith none"), ("fp32", "--arith fp32"), ("all", "--arith all")]
    c += [(f"all_{f}", f"--arith all --fmt {f}") for f in ("e8m10", "e8m13", "e8m16", "fp16")]
    for ch in (8, 16, 32, 64, 128, 256):
        for m in ("acc", "fp32"):
            c.append((f"all_chunk{ch}_{m}", f"--arith all --chunk {ch} --chunk_merge {m}"))
    c += [("all_e8m13_chunk128_fp32", "--arith all --fmt e8m13 --chunk 128 --chunk_merge fp32")]
    c += [("all_kahan", "--arith all --kahan"), ("all_sr", "--arith all --rounding sr"),
          ("all_fp16_kahan", "--arith all --fmt fp16 --kahan")]
    for name, spec in (("cav", "av:b"), ("cpatch", "patch:a"), ("cqk", "qk:a"), ("clin", "lin:a"),
                       ("cbest", CENTER_BEST), ("cbest_lin", CENTER_BEST + ",lin:a")):
        c.append((f"all_{name}", f'--arith all --center "{spec}"'))
    c += [("av_cav", '--arith av --center "av:b"'), ("patch_cpatch", '--arith patch --center "patch:a"'),
          ("lin_clin", '--arith lin --center "lin:a"'), ("qk_cqk", '--arith qk --center "qk:a"')]
    for f in ("tc_bf16", "tc_fp16", "tc_fp16acc", "fp8_exact", "fp8_tc", "fp8_promote"):
        c.append((f"hw_{f}", f"--arith all --fmt {f}"))
    return c


def main():
    models = sys.argv[1].split(",")
    lines = []
    for model in models:
        for name, args in configs(model):
            job = f"x1_{model}_coco_{name}"
            lines.append(f"{job}\t{PY} --model {model} {args} --out results/x1/{model}_coco_{name}.json")
    with open("queue/jobs.tsv", "a") as f:
        f.write("\n".join(lines) + "\n")
    print(len(lines), "jobs")


if __name__ == "__main__":
    main()
