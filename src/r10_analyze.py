"""R10 HSTC analysis — rigorous comparison vs v2a + baseline."""
from __future__ import annotations

import json
import math
import pathlib
import random
import statistics as st
from scipy.stats import wilcoxon, norm

ROWS = pathlib.Path(__file__).parent.parent / "results" / "r10" / "rows.json"
OUT = pathlib.Path(__file__).parent.parent / "results" / "r10" / "analysis.json"


def bootstrap_ci(vals, fn=st.median, B=10000, seed=42):
    random.seed(seed)
    n = len(vals)
    bs = [fn([random.choice(vals) for _ in range(n)]) for _ in range(B)]
    bs.sort()
    return bs[int(B * 0.025)], bs[int(B * 0.975)]


def cohen_d_paired(diffs):
    sd = st.stdev(diffs) if len(diffs) > 1 else 0
    return st.mean(diffs) / sd if sd > 0 else 0.0


def main():
    rows = json.load(open(ROWS, encoding="utf-8"))
    by = {}
    for r in rows:
        if r.get("ok"):
            by.setdefault(r["sample_id"], {})[r["mode"]] = r

    # Triples
    trips = [(d["baseline"], d["v2a"], d["hstc"]) for d in by.values()
             if all(k in d for k in ("baseline", "v2a", "hstc"))]
    n = len(trips)
    print(f"=== R10 HSTC vs v2a vs baseline (n={n}) ===\n")

    # Pull token counts
    tin_b = [t[0]["tokens_in"] for t in trips]
    tin_v = [t[1]["tokens_in"] for t in trips]
    tin_h = [t[2]["tokens_in"] for t in trips]

    rouge_b = [t[0]["rouge_l_f1"] for t in trips]
    rouge_v = [t[1]["rouge_l_f1"] for t in trips]
    rouge_h = [t[2]["rouge_l_f1"] for t in trips]

    wall_b = [t[0]["wall"] for t in trips]
    wall_v = [t[1]["wall"] for t in trips]
    wall_h = [t[2]["wall"] for t in trips]

    # Per-sample reductions
    v_red = [(b - v) / b for b, v in zip(tin_b, tin_v)]
    h_red = [(b - h) / b for b, h in zip(tin_b, tin_h)]
    h_vs_v_red = [(v - h) / v for v, h in zip(tin_v, tin_h)]

    print("[Token-in reduction vs baseline]")
    print(f"  v2a:  median {st.median(v_red)*100:.1f}%  mean {st.mean(v_red)*100:.1f}%  CI {bootstrap_ci(v_red, st.median)}")
    print(f"  hstc: median {st.median(h_red)*100:.1f}%  mean {st.mean(h_red)*100:.1f}%  CI {bootstrap_ci(h_red, st.median)}")
    print(f"  Extra reduction HSTC gives over v2a (paired):")
    print(f"    median {st.median(h_vs_v_red)*100:.1f}%  mean {st.mean(h_vs_v_red)*100:.1f}%")

    # Paired Wilcoxon: hstc < v2a on tokens_in?
    wx_tokens = wilcoxon(tin_v, tin_h)
    d_tokens = cohen_d_paired([h - v for v, h in zip(tin_v, tin_h)])
    print(f"\n[Paired test HSTC vs v2a on tokens_in]")
    print(f"  Wilcoxon: W={wx_tokens.statistic:.0f}, p={wx_tokens.pvalue:.2e}")
    print(f"  Cohen's d (hstc - v2a, negative = fewer tokens): {d_tokens:.3f}")

    # ROUGE-L quality comparison
    print(f"\n[ROUGE-L F1 vs reference]")
    print(f"  baseline median: {st.median(rouge_b):.4f}  mean: {st.mean(rouge_b):.4f}")
    print(f"  v2a      median: {st.median(rouge_v):.4f}  mean: {st.mean(rouge_v):.4f}")
    print(f"  hstc     median: {st.median(rouge_h):.4f}  mean: {st.mean(rouge_h):.4f}")

    # Paired Wilcoxon ROUGE: hstc vs baseline, hstc vs v2a
    wx_rouge_hb = wilcoxon(rouge_b, rouge_h)
    wx_rouge_hv = wilcoxon(rouge_v, rouge_h)
    d_rouge_hb = cohen_d_paired([h - b for b, h in zip(rouge_b, rouge_h)])
    d_rouge_hv = cohen_d_paired([h - v for v, h in zip(rouge_v, rouge_h)])
    print(f"  Wilcoxon baseline vs hstc: p={wx_rouge_hb.pvalue:.4f}  Cohen d(paired)={d_rouge_hb:.3f}")
    print(f"  Wilcoxon v2a vs hstc:      p={wx_rouge_hv.pvalue:.4f}  Cohen d(paired)={d_rouge_hv:.3f}")

    # Wall
    print(f"\n[Wall-clock mean, seconds]")
    print(f"  baseline: {st.mean(wall_b):.2f}s")
    print(f"  v2a:      {st.mean(wall_v):.2f}s")
    print(f"  hstc:     {st.mean(wall_h):.2f}s")

    out = {
        "n_paired": n,
        "tokens_in": {
            "baseline_median": st.median(tin_b),
            "v2a_median": st.median(tin_v),
            "hstc_median": st.median(tin_h),
            "v2a_reduction_median_pct": st.median(v_red) * 100,
            "hstc_reduction_median_pct": st.median(h_red) * 100,
            "hstc_reduction_mean_pct": st.mean(h_red) * 100,
            "hstc_reduction_ci_95_median_pct": list(bootstrap_ci(h_red, st.median)),
            "hstc_extra_vs_v2a_median_pct": st.median(h_vs_v_red) * 100,
            "wilcoxon_hstc_vs_v2a_p": float(wx_tokens.pvalue),
            "cohen_d_hstc_vs_v2a": d_tokens,
        },
        "rouge_l_quality": {
            "baseline_median": st.median(rouge_b),
            "v2a_median": st.median(rouge_v),
            "hstc_median": st.median(rouge_h),
            "wilcoxon_baseline_vs_hstc_p": float(wx_rouge_hb.pvalue),
            "cohen_d_baseline_vs_hstc": d_rouge_hb,
            "wilcoxon_v2a_vs_hstc_p": float(wx_rouge_hv.pvalue),
            "cohen_d_v2a_vs_hstc": d_rouge_hv,
        },
        "wall_clock_mean_s": {
            "baseline": st.mean(wall_b),
            "v2a": st.mean(wall_v),
            "hstc": st.mean(wall_h),
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✓ Written to {OUT}")


if __name__ == "__main__":
    main()
