"""R8 n=100 analysis — power-upgraded statistical validation."""
from __future__ import annotations

import json
import math
import pathlib
import random
import statistics as st
from scipy.stats import wilcoxon, norm

ROWS = pathlib.Path(__file__).parent.parent / "results" / "r8" / "rows.json"
OUT = pathlib.Path(__file__).parent.parent / "results" / "r8" / "analysis.json"


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
    z_test = abs(d) * math.sqrt(n) - z_alpha
    return float(norm.cdf(z_test))


def main():
    rows = json.load(open(ROWS, encoding="utf-8"))
    by = {}
    for r in rows:
        if r.get("ok"):
            by.setdefault(r["sample_id"], {})[r["mode"]] = r
    pairs = [(d["baseline"], d["compressor"]) for d in by.values() if "baseline" in d and "compressor" in d]
    n = len(pairs)
    print(f"R8: n={n} paired samples (target was 100)")

    # Token-in
    tin_b = [p[0]["tokens_in"] for p in pairs]
    tin_c = [p[1]["tokens_in"] for p in pairs]
    red = [(b - c) / b for b, c in zip(tin_b, tin_c)]
    ci_med = bootstrap_ci(red, st.median)
    ci_mean = bootstrap_ci(red, st.mean)
    print(f"\n[Token-in reduction on gpt-oss-20b, n={n}]")
    print(f"  median: {st.median(red)*100:.1f}%  95% CI [{ci_med[0]*100:.1f}%, {ci_med[1]*100:.1f}%]")
    print(f"  mean:   {st.mean(red)*100:.1f}%  95% CI [{ci_mean[0]*100:.1f}%, {ci_mean[1]*100:.1f}%]")
    print(f"  range:  [{min(red)*100:.1f}%, {max(red)*100:.1f}%]")
    print(f"  CI width (median): {(ci_med[1]-ci_med[0])*100:.1f} pts (R3 n=11 was 10.5 pts)")

    # Wilcoxon on tokens_in
    wx = wilcoxon(tin_b, tin_c)
    diffs = [c - b for c, b in zip(tin_c, tin_b)]
    d_tokens = cohen_d_paired(diffs)
    print(f"\n[Statistical power at n={n}]")
    print(f"  Wilcoxon paired on tokens_in: W={wx.statistic:.0f}, p={wx.pvalue:.2e}")
    print(f"  Cohen's d (tokens_in): {d_tokens:.3f}")
    print(f"  Power to detect d=0.3: {observed_power(0.3, n)*100:.1f}% (R3 n=11 was 26%)")

    # Wall
    w_b = [p[0].get("wall", 0) for p in pairs]
    w_c = [p[1].get("wall", 0) for p in pairs]
    w_red = [(b - c) / b for b, c in zip(w_b, w_c) if b > 0]
    if w_red:
        ci_w = bootstrap_ci(w_red, st.mean)
        print(f"\n[Wall-clock reduction, n={len(w_red)}]")
        print(f"  mean: {st.mean(w_red)*100:.1f}%  95% CI [{ci_w[0]*100:.1f}%, {ci_w[1]*100:.1f}%]")

    out = {
        "n": n,
        "model": "openai/gpt-oss-20b",
        "tokens_in": {
            "baseline_median": st.median(tin_b),
            "compressor_median": st.median(tin_c),
            "reduction_median_pct": st.median(red) * 100,
            "reduction_mean_pct": st.mean(red) * 100,
            "reduction_ci_95_median_pct": [ci_med[0] * 100, ci_med[1] * 100],
            "reduction_ci_95_mean_pct": [ci_mean[0] * 100, ci_mean[1] * 100],
            "ci_width_pts": (ci_med[1] - ci_med[0]) * 100,
            "wilcoxon_p": float(wx.pvalue),
            "cohen_d": d_tokens,
            "power_at_d_0_3_pct": observed_power(0.3, n) * 100,
        },
        "wall_clock": {
            "mean_reduction_pct": st.mean(w_red) * 100 if w_red else None,
            "ci_95_mean_pct": [ci_w[0] * 100, ci_w[1] * 100] if w_red else None,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✓ Written to {OUT}")


if __name__ == "__main__":
    main()
