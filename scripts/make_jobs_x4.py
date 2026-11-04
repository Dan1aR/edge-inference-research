"""Phase X4: other models, datasets, resolutions (inference). Key arithmetic configs fixed in advance."""
PY = "NUM_WORKERS=6 .venv/bin/python eval.py"
KEY = [("none", "--arith none"), ("all", "--arith all"), ("all_e8m13", "--arith all --fmt e8m13"),
       ("all_chunk32_fp32", "--arith all --chunk 32 --chunk_merge fp32"),
       ("all_chunk128_fp32", "--arith all --chunk 128 --chunk_merge fp32"),
       ("hw_fp8_exact", "--arith all --fmt fp8_exact"), ("hw_fp8_tc", "--arith all --fmt fp8_tc"),
       ("hw_fp8_promote", "--arith all --fmt fp8_promote")]
VIT_EXTRA = [("lin+patch", "--arith lin,patch"), ("attn", "--arith attn"), ("lin", "--arith lin"),
             ("all_cbest", '--arith all --center "av:b,patch:a,qk:a"')]
CNN_EXTRA = [("conv", "--arith conv"), ("lin", "--arith lin"), ("attn", "--arith attn"),
             ("all-conv", "--arith all-conv"), ("all_e8m10", "--arith all --fmt e8m10"),
             ("all_fp16", "--arith all --fmt fp16"), ("all_cconv", '--arith all --center "conv:a"'),
             ("hw_tc_fp16acc", "--arith all --fmt tc_fp16acc"), ("hw_tc_fp16", "--arith all --fmt tc_fp16")]
lines = []


def add(model, ds, cfgs, extra=""):
    for name, args in cfgs:
        tag = f"{model}_{ds}_{name}"
        lines.append(f"x4_{tag}\t{PY} --model {model} --dataset {ds} {args} {extra} --out results/x4/{tag}.json")


# CNN-backbone detectors on COCO
add("rtdetr_r18", "coco_val", KEY + CNN_EXTRA)
add("detr_r50", "coco_val", KEY + CNN_EXTRA, "--batch_size 16")
add("rtdetr_r50", "coco_val", KEY + [c for c in CNN_EXTRA if c[0] in ("conv", "all-conv")])
# datasets: YOLOS-tiny/small and RT-DETR-r18
for ds in ("voc07_test", "visdrone_val", "bdd_val", "coco_o"):
    for m in ("tiny", "small"):
        add(m, ds, KEY[:5] + [VIT_EXTRA[0], VIT_EXTRA[3]])
    add("rtdetr_r18", ds, KEY[:5] + [CNN_EXTRA[0]])
# resolution axis (YOLOS-tiny/small): more tokens -> longer softmax.V sums
for size in ("640x640", "800x800", "1024x1024"):
    s = size.split("x")[0]
    for m in ("tiny", "small"):
        for name, args in KEY[:5] + [("av", "--arith av"), ("all-av", "--arith all-av")]:
            tag = f"{m}_coco_r{s}_{name}"
            lines.append(f"x4_{tag}\t{PY} --model {m} --size {size} --batch_size 16 {args} --out results/x4/{tag}.json")
# DeiT on ImageNetV2
for m in ("facebook/deit-tiny-patch16-224", "facebook/deit-small-patch16-224", "facebook/deit-base-patch16-224"):
    short = m.split("/")[1].split("-patch")[0]
    for name, args in [("none", "--arith none"), ("fp32", "--arith fp32"), ("all", "--arith all"),
                       ("all_e8m13", "--arith all --fmt e8m13"), ("all_chunk32_fp32", "--arith all --chunk 32 --chunk_merge fp32"),
                       ("lin+patch", "--arith lin,patch"), ("attn", "--arith attn"), ("hw_fp8_tc", "--arith all --fmt fp8_tc"),
                       ("hw_fp8_exact", "--arith all --fmt fp8_exact")]:
        lines.append(f"x4_{short}_inv2_{name}\tNUM_WORKERS=6 .venv/bin/python eval_cls.py --model {m} {args} "
                     f"--out results/x4/{short}_inv2_{name}.json")
with open("queue/jobs.tsv", "a") as f:
    f.write("\n".join(lines) + "\n")
print(len(lines), "jobs")
