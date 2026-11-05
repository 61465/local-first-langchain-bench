"""R11 HSTC n=100 analysis — combines with R8 baseline/v2a for 3-way comparison.

Produces:
- HSTC n=100 reduction stats (median, mean, 95% CI, Wilcoxon, Cohen's d)
- HSTC vs v2a paired comparison on same 100 sample_ids
- Power analysis at n=100
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import statistics as st
from scipy.stats import wilcoxon, norm

R8_ROWS = pathlib.Path(__file__).parent.parent / "results" / "r8" / "rows.json"
R11_ROWS = pathlib.Path(__file__).parent.parent / "results" / "r11" / "rows.json"
OUT = pathlib.Path(__file__).parent.parent / "results" / "r11" / "analysis.json"


def bootstrap_ci(vals, fn=st.median, B=10000, seed=42):
    random.seed(seed)
    n = len(vals)
    bs = [fn([random.choice(vals) for _ in range(n)]) for _ in range(B)]
    bs.sort()
    return bs[int(B * 0.025)], bs[int(B * 0.975)]


def cohen_d_paired(diffs):
    sd = st.stdev(diffs) if len(diffs) > 1 else 0
    return st.mean(diffs) / sd if sd > 0 else 0.0


def observed_power(d, n, alpha=0.05, two_sided=True):
    z_alpha = norm.ppf(1 - alpha / (2 if two_sided else 1))
    z = abs(d) * math.sqrt(n) - z_alpha
    return float(norm.cdf(z))


def main():
    # Load R8 (baseline + v2a) and R11 (HSTC)
    r8 = json.load(open(R8_ROWS, encoding="utf-8"))
    r11 = json.load(open(R11_ROWS, encoding="utf-8"))

    r8_by = {}
    for r in r8:
        if r.get("ok"):
            r8_by.setdefault(r["sample_id"], {})[r["mode"]] = r

    r11_by = {r["sample_id"]: r for r in r11 if r.get("ok")}

    # Triples: baseline + v2a + hstc on same sample_id
    sids = [sid for sid in r11_by if sid in r8_by and "baseline" in r8_by[sid] and "compressor" in r8_by[sid]]
    n = len(sids)
    print(f"=== R11 HSTC n={n} vs R8 baseline/v2a (same sample_ids) ===\n")

    tin_b = [r8_by[sid]["baseline"]["tokens_in"] for sid in sids]
    tin_v = [r8_by[sid]["compressor"]["tokens_in"] for sid in sids]
    tin_h = [r11_by[sid]["tokens_in"] for sid in sids]

    red_v = [(b - v) / b for b, v in zip(tin_b, tin_v)]
    red_h = [(b - h) / b for b, h in zip(tin_b, tin_h)]
    red_hv = [(v - h) / v for v, h in zip(tin_v, tin_h)]

    print("[Token-in reduction vs baseline, per-sample paired]")
    for name, reds in [("v2a", red_v), ("hstc", red_h)]:
        ci = bootstrap_ci(reds, st.median)
        print(f"  {name:5s}: median={st.median(reds)*100:5.1f}%  mean={st.mean(reds)*100:5.1f}%  CI_med=[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")

    # HSTC vs v2a paired
    print(f"\n[HSTC vs v2a (paired)]")
    print(f"  Extra reduction HSTC gives over v2a: median={st.median(red_hv)*100:.1f}%  mean={st.mean(red_hv)*100:.1f}%")
    diffs_hv = [h - v for v, h in zip(tin_v, tin_h)]
    d_hv = cohen_d_paired(diffs_hv)
    wx_hv = wilcoxon(tin_v, tin_h)
    direction_stable = sum(1 for d in diffs_hv if d < 0) / len(diffs_hv) * 100
    print(f"  Wilcoxon: W={wx_hv.statistic:.0f}, p={wx_hv.pvalue:.2e}")
    print(f"  Cohen's d (hstc - v2a): {d_hv:.3f}")
    print(f"  Direction stability: {direction_stable:.0f}% samples have HSTC < v2a")
    print(f"  Power @ n={n}: {observed_power(abs(d_hv), n)*100:.1f}%")

    # Save
    out = {
        "n_paired": n,
        "hstc_reduction": {
            "median_pct": st.median(red_h) * 100,
            "mean_pct": st.mean(red_h) * 100,
            "ci_95_median_pct": list(bootstrap_ci(red_h, st.median)),
            "range_pct": [min(red_h) * 100, max(red_h) * 100],
        },
        "v2a_reduction": {
            "median_pct": st.median(red_v) * 100,
            "mean_pct": st.mean(red_v) * 100,
            "ci_95_median_pct": list(bootstrap_ci(red_v, st.median)),
        },
        "hstc_vs_v2a_paired": {
            "extra_reduction_median_pct": st.median(red_hv) * 100,
            "extra_reduction_mean_pct": st.mean(red_hv) * 100,
            "wilcoxon_statistic": float(wx_hv.statistic),
            "wilcoxon_p": float(wx_hv.pvalue),
            "cohen_d_paired": d_hv,
            "direction_stability_pct": direction_stable,
            "power_observed_pct": observed_power(abs(d_hv), n) * 100,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✓ Written to {OUT}")


if __name__ == "__main__":
    main()
