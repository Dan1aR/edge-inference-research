"""Collect all experiment outputs into markdown tables: docs/experiments_tables.md"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/experiments_tables.md"


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def f4(x):
    return "—" if x is None else f"{x:.4f}"


def pp(x):
    return "—" if x is None else f"{100 * x:+.2f}"


def ci_str(vals):
    vals = np.asarray(vals, dtype=float)
    n = len(vals)
    if n == 0:
        return "—"
    if n == 1:
        return f"{vals[0]:.4f} (n=1)"
    m, s = vals.mean(), vals.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return f"{m:.4f} ± {s:.4f} [{m - h:.4f}; {m + h:.4f}] (n={n})"


def paired_str(a, b):
    """mean(a-b) in pp, 95% CI, paired t-test p."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    if n < 2:
        return f"{100 * d.mean():+.2f} (n={n})" if n else "—"
    m, s = d.mean(), d.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    p = stats.ttest_1samp(d, 0).pvalue if s > 0 else 0.0
    return f"{100 * m:+.2f} [{100 * (m - h):+.2f}; {100 * (m + h):+.2f}], p={p:.3g} (n={n})"


E2_LABELS = [
    ("fp32", "FP32 (без autocast)"), ("none", "BF16, FP32-накопление (autocast)"),
    ("all_fma", "BF16-accum: всё"), ("lin_fma", "BF16-accum: lin"), ("patch_fma", "BF16-accum: patch"),
    ("attn_fma", "BF16-accum: attn (QKᵀ+AV)"), ("lin+patch_fma", "BF16-accum: lin+patch (= M1 без FT)"),
    ("lin+attn_fma", "BF16-accum: lin+attn"), ("patch+attn_fma", "BF16-accum: patch+attn"),
    ("all-av_fma", "BF16-accum: всё, кроме AV (= M2 без FT)"),
    ("qkv_fma", "только qkv"), ("attn_out_fma", "только attn_out"), ("mlp_fma", "только mlp"),
    ("head_fma", "только head"), ("qk_fma", "только QKᵀ"), ("av_fma", "только AV"),
]


def e2_section(lines):
    d = ROOT / "results/e2"
    boot = load(d / "bootstrap.json") or {"deltas": {}}
    lines += ["## Э2. Абляция на инференсе (YOLOS-tiny, COCO val2017, 5000 изобр., L40S)", "",
              "Семантика MAC: FMA (s ← bf16(s + a·b)), последовательное накопление. ΔAP в п.п.; "
              "95% ДИ — парный bootstrap по изображениям.", "",
              "| Конфигурация | AP | AP50 | AP75 | AP_S | AP_M | AP_L | ΔAP vs BF16 [95% ДИ] | ΔAP vs FP32 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, label in E2_LABELS:
        r = load(d / f"{key}.json")
        if r is None:
            continue
        m = r["metrics"]
        dn = boot["deltas"].get(f"{key} - none")
        df = boot["deltas"].get(f"{key} - fp32")
        dn_s = "—" if dn is None else f"{100 * dn['delta']:+.2f} [{100 * dn['ci95_lo']:+.2f}; {100 * dn['ci95_hi']:+.2f}]"
        df_s = "—" if df is None else f"{100 * df['delta']:+.2f}"
        lines.append(f"| {label} | {m['AP']:.4f} | {m['AP50']:.4f} | {m['AP75']:.4f} | {m['AP_small']:.4f} | "
                     f"{m['AP_medium']:.4f} | {m['AP_large']:.4f} | {dn_s} | {df_s} |")
    lines += ["", "### Семантика MAC и порядок суммирования (всё в BF16-accum)", "",
              "| Вариант | AP | ΔAP vs FMA-последоват. [95% ДИ] |", "|---|---|---|"]
    variants = [("all_fma", "FMA, последовательно"), ("all_rp", "округление произведения, последовательно")]
    for c in (16, 32, 64):
        variants += [(f"all_chunk{c}_mergebf16", f"FMA, блоки по {c}, слияние в BF16"),
                     (f"all_chunk{c}_mergefp32", f"FMA, блоки по {c}, слияние в FP32")]
    for key, label in variants:
        r = load(d / f"{key}.json")
        if r is None:
            continue
        dd = boot["deltas"].get(f"{key} - all_fma")
        ds = "—" if dd is None else f"{100 * dd['delta']:+.2f} [{100 * dd['ci95_lo']:+.2f}; {100 * dd['ci95_hi']:+.2f}]"
        lines.append(f"| {label} | {r['metrics']['AP']:.4f} | {ds} |")
    lines += ["", "### Комбинации lin/patch/attn: FMA против округления произведения", "",
              "| lin/patch/attn | AP (FMA) | AP (round product) |", "|---|---|---|"]
    for key in ("lin", "patch", "attn", "lin+patch", "lin+attn", "patch+attn", "all"):
        a, b = load(d / f"{key}_fma.json"), load(d / f"{key}_rp.json")
        lines.append(f"| {key} | {f4(a and a['metrics']['AP'])} | {f4(b and b['metrics']['AP'])} |")
    lines.append("")


# method -> (label, pre-FT reference arithmetic file in results/e2, ref in results/e4)
METHODS = {
    "B": ("B: BF16, FP32-накопление", "none", "none"),
    "I": ("I: immediate, BF16-accum всё", "all_fma", "all"),
    "PA": ("P-A: progressive, 300 шагов (60 в полной конф.)", "all_fma", "all"),
    "PB": ("P-B: progressive, 540 шагов (300 в полной конф.)", "all_fma", "all"),
    "M1": ("M1: FP32-накопление в attention (QKᵀ, AV)", "lin+patch_fma", "lin+patch"),
    "M2": ("M2: FP32-накопление только в AV", "all-av_fma", "all-av"),
    "NP": ("I без patch (как в старой статье)", "lin+attn_fma", "all-patch"),
}


def collect_runs(run_root, methods, prefix=""):
    out = {}
    for m in methods:
        rows = []
        for p in sorted(Path(run_root).glob(f"{prefix}{m}_s*/summary.json")):
            s = load(p)
            rows.append({"seed": int(p.parent.name.split("_s")[-1]), "s": s})
        out[m] = rows
    return out


def ft_section(lines, title, run_root, e2_dir, voc_dir, methods, prefix="", b_root=None, b_prefix=None):
    """b_root/b_prefix: where the baseline (B) runs for the deployment comparison live (default: same section)."""
    runs = collect_runs(run_root, methods, prefix)
    b_root = b_root or run_root
    b_prefix = prefix if b_prefix is None else b_prefix
    b_runs = runs.get("B") if (b_root == run_root and b_prefix == prefix) else collect_runs(b_root, ["B"], b_prefix)["B"]
    if not any(runs.values()):
        return
    lines += [f"## {title}", "",
              "Основной результат — AP финального (фиксированного) шага на полном COCO val2017 (5000 изобр.). "
              "Выбор чекпойнта: по половине val-A (2500 изобр.) в целевой арифметике, отчёт — на другой половине val-B; "
              "для сравнения там же приведён final на val-B. Все модели оцениваются в своей целевой арифметике. "
              "Среднее ± std [95% ДИ по t-распределению].", "",
              "| Метод | AP до FT | AP final (val) | AP final (val-B) | AP best (val-B) | шаг best (медиана) | восст. доли разрыва* | выигрыш над B в той же арифм.** |",
              "|---|---|---|---|---|---|---|---|"]
    none_pre = load(Path(e2_dir) / "none.json")
    none_pre = none_pre and none_pre["metrics"]["AP"]
    for m, rows in runs.items():
        if not rows:
            continue
        label, ref, _ = METHODS[m]
        pre = load(Path(e2_dir) / f"{ref}.json")
        pre = pre and pre["metrics"]["AP"]
        fin = [r["s"]["val"]["coco_val"]["final"]["AP"] for r in rows]
        fin_b = [r["s"]["val"]["coco_val"]["final"]["valB"]["AP"] for r in rows]
        best = [r["s"]["val"]["coco_val"]["best"]["valB"]["AP"] for r in rows]
        steps = [r["s"]["best_step"] for r in rows]
        rec = "—"
        if pre is not None and none_pre is not None and ref != "none" and none_pre > pre:
            rec = f"{100 * (np.mean(fin) - pre) / (none_pre - pre):.0f}%"
        gain = "—"
        if m != "B":
            b_dep = {}
            for r in b_runs or []:
                v = deployed_B(r, METHODS[m][2], b_root, b_prefix)
                if v is not None:
                    b_dep[r["seed"]] = v
            common = [r for r in rows if r["seed"] in b_dep]
            if common:
                gain = paired_str([r["s"]["val"]["coco_val"]["final"]["AP"] for r in common],
                                  [b_dep[r["seed"]] for r in common])
        lines.append(f"| {label} | {f4(pre)} | {ci_str(fin)} | {ci_str(fin_b)} | {ci_str(best)} | {int(np.median(steps))} | {rec} | {gain} |")
    lines += ["", "\\* (AP final − AP до FT) / (AP BF16 до FT − AP до FT): какую долю потерь от BF16-accum закрыло дообучение.",
              "\\*\\* ΔAP final метода минус AP весов B (тот же сид, тот же порядок данных, то же число шагов), развёрнутых в целевой "
              "арифметике метода: эффект именно адаптации к арифметике, без общего дрейфа от дообучения. п.п. [95% ДИ], парный t-тест.", ""]

    # B / M1 / M2 weights deployed on full BF16-accum
    extra = []
    for m in ("B", "M1", "M2"):
        rows = runs.get(m) or []
        vals = [r["s"]["val"]["coco_val"].get("final@all", {}).get("AP") for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            extra.append(f"| {METHODS[m][0]} → оценка в полном BF16-accum | {ci_str(vals)} |")
    if extra:
        lines += ["Контроль: веса после дообучения, развёрнутые на полном BF16-accum:", "",
                  "| Модель | AP (всё в BF16-accum) |", "|---|---|", *extra, ""]

    # paired comparisons (same seed = same data order)
    by_seed = {m: {r["seed"]: r["s"]["val"]["coco_val"]["final"]["AP"] for r in rows} for m, rows in runs.items()}
    pairs = [("I", "PA"), ("I", "PB"), ("PB", "PA"), ("M1", "I"), ("M2", "I"), ("B", "M1"), ("B", "I")]
    plines = []
    for a, b in pairs:
        common = sorted(set(by_seed.get(a, {})) & set(by_seed.get(b, {})))
        if common:
            plines.append(f"| {a} − {b} | {paired_str([by_seed[a][s] for s in common], [by_seed[b][s] for s in common])} |")
    if plines:
        lines += ["Парные сравнения (одинаковый сид = одинаковый порядок данных), ΔAP final в п.п. [95% ДИ], парный t-тест:", "",
                  "| Сравнение | ΔAP |", "|---|---|", *plines, ""]

    # VOC transfer
    vlines = []
    for m, rows in runs.items():
        vals = [r["s"]["val"].get("voc07_test", {}).get("final", {}).get("AP") for r in rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        ref = load(Path(voc_dir) / f"voc_{METHODS[m][2]}.json")
        vlines.append(f"| {METHODS[m][0]} | {f4(ref and ref['metrics']['AP'])} | {ci_str(vals)} |")
    if vlines:
        lines += ["Перенос на Pascal VOC 2007 test (без дообучения на VOC; AP по 20 классам, протокол COCO):", "",
                  "| Метод | AP до FT | AP final |", "|---|---|---|", *vlines, ""]


def deployed_B(row, target, run_root, prefix=""):
    """AP of baseline (B) final weights evaluated in another arithmetic (from train.py extra evals or e*x evals)."""
    for key in (f"final@{target}", f"final@{target.replace('+', ',')}"):
        v = row["s"]["val"]["coco_val"].get(key, {}).get("AP")
        if v is not None:
            return v
    tag = target.replace(",", "+")
    r = load(ROOT / f"results/{Path(run_root).name}x/{prefix}B_s{row['seed']}_{tag}.json")
    return r and r["metrics"]["AP"]


def e8_section(lines):
    root = ROOT / "runs/e8"
    ft_section(lines, "Э8a. Дообучение YOLOS-tiny с LR 1e-5 (300 шагов)", root, ROOT / "results/e2", ROOT / "results/e4",
               ["B", "I", "M1"], prefix="lr1e-5_")
    ft_section(lines, "Э8b. Конфигурация FT старой статьи: всё, кроме patch, LR 5e-5", root, ROOT / "results/e2",
               ROOT / "results/e4", ["NP"], b_root=ROOT / "runs/e3", b_prefix="")
    rows = []
    for p in sorted(root.glob("lr1.5e-4_*/summary.json")):
        s = load(p)
        v = s["val"]["coco_val"]
        rows.append((p.parent.name, s["args"]["target"], v["final"]["AP"], v.get("final@all", {}).get("AP"),
                     v["final"]["valB"]["AP"], v["best"]["valB"]["AP"], s["best_step"]))
    if rows:
        lines += ["## Э8c. LR 1.5e-4 (1 сид)", "",
                  "| Прогон | целевая арифм. | AP final (целевая) | AP final (всё BF16-accum) | AP final (val-B) | AP best (val-B) | шаг best |",
                  "|---|---|---|---|---|---|---|"]
        for name, tgt, fin, fin_all, fin_b, best, step in rows:
            lines.append(f"| {name} | {tgt} | {fin:.4f} | {f4(fin_all)} | {fin_b:.4f} | {best:.4f} | {step} |")
        lines.append("")


def simple_eval_section(lines, title, d, prefix):
    rows = []
    keys = {}
    boot = load(Path(d) / "bootstrap.json") or {"deltas": {}}
    for key, label in [("fp32", "FP32"), ("none", "BF16, FP32-накопление"), ("all", "BF16-accum: всё"),
                       ("lin+patch", "BF16-accum: lin+patch (M1)"), ("all-av", "BF16-accum: всё, кроме AV (M2)"),
                       ("lin", "BF16-accum: lin"), ("attn", "BF16-accum: attn"), ("patch", "BF16-accum: patch")]:
        r = load(Path(d) / f"{prefix}{key}.json")
        if r:
            rows.append((label, r["metrics"]))
            keys[label] = key
    if not rows:
        return
    base = rows[1][1]["AP"] if len(rows) > 1 else None
    lines += [f"## {title}", "", "| Конфигурация | AP | AP50 | AP_S | AP_M | AP_L | ΔAP vs BF16, п.п. [95% ДИ] |", "|---|---|---|---|---|---|---|"]
    for label, m in rows:
        key = keys[label]
        dk = f"{key if key in ('fp32', 'none') else key + '_fma'} - none"
        ci = boot["deltas"].get(dk)
        ci = f" [{pp(ci['ci95_lo'])}; {pp(ci['ci95_hi'])}]" if ci else ""
        lines.append(f"| {label} | {m['AP']:.4f} | {m['AP50']:.4f} | {m['AP_small']:.4f} | {m['AP_medium']:.4f} | "
                     f"{m['AP_large']:.4f} | {pp(m['AP'] - base) if base is not None else '—'}{ci} |")
    lines.append("")


def e1_section(lines):
    r = load(ROOT / "results/e1/kernel_report.json")
    if not r:
        return
    lines += ["## Э1. Численная характеристика BF16-накопления", "",
              "Относительная ошибка ‖Y − Y_exact‖/‖Y_exact‖ на реальных активациях YOLOS-tiny (16 изобр. val2017), "
              "одинаковые bf16-входы:", "",
              "| Компонент | K | BF16 + FP32-накопл. | BF16-accum FMA | BF16-accum round-product | блоки 32 + FP32-слияние |",
              "|---|---|---|---|---|---|"]
    for c, e in r["real_activations"].items():
        lines.append(f"| {c} | {e['K']} | {e['bf16_fp32acc']:.2e} | {e['fma_seq']:.2e} | {e['roundprod_seq']:.2e} | {e['fma_chunk32_fp32']:.2e} |")
    lines += ["", "Синтетика (256×K @ K×256), относительная ошибка; positive = слагаемые одного знака (как softmax·V, пиксели·веса):", "",
              "| K | распр. | BF16+FP32 | FMA | round-product | блоки32/BF16 | блоки32/FP32 | γ_K = Ku/(1−Ku) | √K·u |",
              "|---|---|---|---|---|---|---|---|---|"]
    for row in r["k_sweep"]:
        b = row["bound_ku"]
        lines.append(f"| {row['K']} | {row['dist']} | {row['bf16_fp32acc']:.1e} | {row['fma_seq']:.1e} | {row['roundprod_seq']:.1e} | "
                     f"{row['fma_chunk32_bf16']:.1e} | {row['fma_chunk32_fp32']:.1e} | {'—' if b is None else f'{b:.2f}'} | {row['prob_sqrtk_u']:.1e} |")
    lines += ["", "Стоимость эмуляции (справочно, L40S, batch 64): время Triton BF16-acc / cuBLAS BF16:", "",
              "| Операция | cuBLAS, мс | Triton BF16-acc, мс | замедление |", "|---|---|---|---|"]
    for k, v in r["throughput_ms"].items():
        lines.append(f"| {k} | {v['cublas_bf16']:.2f} | {v['triton_bf16acc']:.2f} | ×{v['slowdown']:.1f} |")
    lines.append("")


def main():
    lines = ["# Сводные таблицы экспериментов (генерируется scripts/aggregate.py)", ""]
    e1_section(lines)
    e2_section(lines)
    ft_section(lines, "Э3. Дообучение YOLOS-tiny (300 шагов, 5 сидов)", ROOT / "runs/e3", ROOT / "results/e2",
               ROOT / "results/e4", ["B", "I", "PA", "PB", "M1", "M2"])
    simple_eval_section(lines, "Э4. Pascal VOC 2007 test без дообучения (YOLOS-tiny)", ROOT / "results/e4", "voc_")
    simple_eval_section(lines, "Э5. YOLOS-small, COCO val2017, инференс", ROOT / "results/e5", "small_coco_")
    ft_section(lines, "Э5. Дообучение YOLOS-small (300 шагов, LR 1e-5, 3 сида)", ROOT / "runs/e5", ROOT / "results/e5_as_e2",
               ROOT / "results/e4", ["B", "I", "M1"])
    ft_section(lines, "Э7. Длинное дообучение YOLOS-tiny (1500 шагов, LR 1e-5, 3 сида)", ROOT / "runs/e7", ROOT / "results/e2",
               ROOT / "results/e4", ["B", "I", "M1"])
    e8_section(lines)
    OUT.write_text("\n".join(lines))
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
