"""R7 analysis — multi-workload token reduction + per-workload stats."""
from __future__ import annotations

import json
import pathlib
import random
import statistics as st
from scipy.stats import wilcoxon

ROWS = pathlib.Path(__file__).parent.parent / "results" / "r7" / "rows.json"
OUT = pathlib.Path(__file__).parent.parent / "results" / "r7" / "analysis.json"


def bootstrap_ci(vals, fn=st.median, B=10000, seed=42):
    random.seed(seed)
    n = len(vals)
    bs = [fn([random.choice(vals) for _ in range(n)]) for _ in range(B)]
    bs.sort()
    return bs[int(B * 0.025)], bs[int(B * 0.975)]


def cohen_d_paired(diffs):
    sd = st.stdev(diffs) if len(diffs) > 1 else 0
    return st.mean(diffs) / sd if sd > 0 else 0.0


def analyze_workload(rows, workload):
    rs = [r for r in rows if r.get("workload") == workload and r.get("ok")]
    by = {}
    key_fn = (lambda r: r["sample_id"]) if workload == "dialogue" else (lambda r: r["scenario_idx"])
    for r in rs:
        by.setdefault(key_fn(r), {})[r["mode"]] = r
    tin_b, tin_c, w_b, w_c = [], [], [], []
    for sid, m in by.items():
        if "baseline" in m and "compressor" in m:
            tin_b.append(m["baseline"]["tokens_in"])
            tin_c.append(m["compressor"]["tokens_in"])
            w_b.append(m["baseline"]["wall"])
            w_c.append(m["compressor"]["wall"])
    n = len(tin_b)
    red = [(b - c) / b for b, c in zip(tin_b, tin_c)]
    w_red = [(b - c) / b for b, c in zip(w_b, w_c)]
    ci = bootstrap_ci(red, st.median)
    wx = wilcoxon(tin_b, tin_c) if n >= 3 and any(c != b for b, c in zip(tin_b, tin_c)) else None
    diffs = [c - b for c, b in zip(tin_c, tin_b)]
    d = cohen_d_paired(diffs)
    return {
        "n": n,
        "baseline_median_tokens": st.median(tin_b) if tin_b else 0,
        "compressor_median_tokens": st.median(tin_c) if tin_c else 0,
        "reduction_median_pct": st.median(red) * 100 if red else 0,
        "reduction_mean_pct": st.mean(red) * 100 if red else 0,
        "reduction_ci_95_pct": [ci[0] * 100, ci[1] * 100] if red else [0, 0],
        "wall_reduction_mean_pct": st.mean(w_red) * 100 if w_red else 0,
        "wilcoxon_pvalue": float(wx.pvalue) if wx else None,
        "cohen_d_tokens_in": d,
        "unchanged_samples": sum(1 for r in red if abs(r) < 0.001),
    }


def main():
    rows = json.load(open(ROWS, encoding="utf-8"))
    print("=== R7 Multi-Workload Compression ===\n")
    print(f"{'Workload':20s}  {'n':>3s}  {'median red%':>11s}  {'mean red%':>10s}  {'95% CI':>18s}  {'wall red%':>9s}  {'Wilcoxon p':>11s}  {'Cohen d':>7s}  {'noop':>5s}")
    print("-" * 115)
    results = {}
    for wl in ["dialogue", "rag_qa"]:
        a = analyze_workload(rows, wl)
        results[wl] = a
        p = f"{a['wilcoxon_pvalue']:.2e}" if a["wilcoxon_pvalue"] else "-"
        print(f"{wl:20s}  {a['n']:>3d}  {a['reduction_median_pct']:>10.1f}%  {a['reduction_mean_pct']:>9.1f}%  [{a['reduction_ci_95_pct'][0]:>5.1f}%, {a['reduction_ci_95_pct'][1]:>5.1f}%]  {a['wall_reduction_mean_pct']:>8.1f}%  {p:>11s}  {a['cohen_d_tokens_in']:>7.3f}  {a['unchanged_samples']:>5d}")

    print()
    print("Interpretation:")
    print("  dialogue:  the news-tuned extractive compressor UNDER-compresses short dialogues;")
    print(f"             {results['dialogue']['unchanged_samples']}/{results['dialogue']['n']} are kept verbatim (pass-through), consistent with v2a safe-guard for short inputs.")
    print("             Honest conclusion: v2a is a NO-OP on short dialogue, not harmful — workload needs a dialogue-specialized compressor (Phase 2).")
    print("  rag_qa:    long retrieved-context compression works exactly as expected;")
    print(f"             median {results['rag_qa']['reduction_median_pct']:.1f}% reduction with Wilcoxon p={results['rag_qa']['wilcoxon_pvalue']:.2e}.")
    print("             This is the dominant LangChain-production shape: large static system + small dynamic user.")
    print("             NOTE: 10 samples share the SAME compressed system (1 compression × 10 distinct user queries).")
    print("             Tight CI reflects compression determinism, not 10 independent reduction measurements.")

    results["honest_framing"] = {
        "dialogue": "n=10 short dialogues (median 490 chars). 7/10 under v2a's short-prompt safe-guard = pass-through. v2a is a news-tuned compressor; dialogue-tuned variant is Phase 2 scope.",
        "rag_qa": "n=10 RAG queries over SAME 4281-char system prompt. The reduction is deterministic (same compression applied); 10 queries vary. Tight CI reflects determinism, NOT 10 independent samples. The reduction number (45.6%) is the single compression-ratio measurement on this system prompt.",
        "aggregate_sample_size_across_phases": "R3 n=11 paired + R6 n=33 (3 models × 11) + R7 n=20 = 64 paired measurements; one compressor applied across all.",
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n✓ Written to {OUT}")


if __name__ == "__main__":
    main()
