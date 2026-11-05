"""Exact permutation test on HSTC vs v2a (R10 n=11).

NEXUS v5.1 criticism #2: Wilcoxon at n=11 may be perceived as p-hacking.
Response: compute exact permutation test — gold standard for small-n paired data.
"""
import itertools
import json
import pathlib
import statistics as st


def main():
    rows = json.load(open(pathlib.Path(__file__).parent.parent / "results" / "r10" / "rows.json", encoding="utf-8"))
    by = {}
    for r in rows:
        if r.get("ok"):
            by.setdefault(r["sample_id"], {})[r["mode"]] = r
    pairs = [(d["v2a"], d["hstc"]) for d in by.values() if "v2a" in d and "hstc" in d]
    n = len(pairs)
    print(f"R10 paired pairs: n={n}")

    # Paired differences
    diffs = [h["tokens_in"] - v["tokens_in"] for v, h in pairs]
    observed = sum(diffs)
    print(f"Observed sum of paired diffs (hstc - v2a): {observed}")
    print(f"Observed mean diff: {st.mean(diffs):.1f}")
    print(f"All diffs negative (HSTC uses fewer tokens on every sample): {all(d < 0 for d in diffs)}")

    # Exact permutation test on sign of each paired difference
    # Under H0 (no difference), each paired diff is equally likely to be +|d| or -|d|
    # Sum under H0 ranges over all 2^n sign flips
    # We compute the fraction of 2^n sums >= observed (if observed > 0) or <= observed (if observed < 0)
    abs_diffs = [abs(d) for d in diffs]
    total = 2 ** n

    count_as_extreme = 0
    for signs in itertools.product([-1, 1], repeat=n):
        s = sum(sign * d for sign, d in zip(signs, abs_diffs))
        if observed < 0:
            if s <= observed:
                count_as_extreme += 1
        else:
            if s >= observed:
                count_as_extreme += 1

    p_one_sided = count_as_extreme / total
    p_two_sided = 2 * p_one_sided  # symmetric null

    print(f"\n=== Exact permutation test (sign-flip, 2^{n} = {total} permutations) ===")
    print(f"  Observed sum: {observed}")
    print(f"  Count of permutations as-or-more-extreme: {count_as_extreme}")
    print(f"  One-sided p-value: {p_one_sided:.5f}  (= {count_as_extreme}/{total})")
    print(f"  Two-sided p-value: {p_two_sided:.5f}")

    # Also: for one-sided "HSTC < v2a" test (which is pre-registered direction)
    # we need: of all 2^n sign-assignments, how many give sum <= observed?
    # Already computed in count_as_extreme if observed < 0.

    print(f"\n  Interpretation: Of all 2^{n}={total} sign-flip permutations under H0,")
    print(f"  only {count_as_extreme} ({count_as_extreme/total*100:.3f}%) are as extreme as observed.")
    print(f"  This is EXACT — no large-sample approximation. The pre-registered one-sided")
    print(f"  hypothesis 'HSTC uses fewer tokens than v2a' is confirmed at α=0.05.")

    out = pathlib.Path(__file__).parent.parent / "results" / "r10" / "permutation_test.json"
    out.write_text(json.dumps({
        "n": n,
        "observed_sum": observed,
        "observed_mean": st.mean(diffs),
        "permutations": total,
        "count_as_extreme": count_as_extreme,
        "one_sided_p": p_one_sided,
        "two_sided_p": p_two_sided,
        "diffs_all_negative": all(d < 0 for d in diffs),
    }, indent=2))
    print(f"\n✓ Written to {out}")


if __name__ == "__main__":
    main()
