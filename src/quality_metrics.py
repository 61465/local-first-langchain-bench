"""
Phase 2 quality metrics — beyond ROUGE-L.

Adds:
- BERTScore-lite: sentence-embedding cosine similarity (semantic, not lexical)
- Statistical: paired Wilcoxon signed-rank test (ROUGE-L drop significance)
- Combined "quality floor": worst of (ROUGE_L, BERTScore_cosine)
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
from typing import Optional

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: 22 MB, fast, standard baseline
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def bertscore_cosine(candidate: str, reference: str) -> float:
    """Cosine similarity between SBERT embeddings of candidate and reference.

    Returns value in [-1, 1]; typically [0, 1] for summaries.
    This is BERTScore-lite: not full P/R/F1 token-level, but captures semantic
    similarity well enough for our n~=11 use case.
    """
    model = _get_model()
    embs = model.encode([candidate, reference], convert_to_numpy=True, show_progress_bar=False)
    a, b = embs[0], embs[1]
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def paired_wilcoxon(baseline_scores: list[float], contender_scores: list[float]) -> dict:
    """Paired Wilcoxon signed-rank test — tests if contender scores are
    significantly different from baseline.

    Returns {statistic, pvalue, mean_delta, median_delta, interpretation}.
    """
    from scipy import stats
    if len(baseline_scores) != len(contender_scores):
        raise ValueError(f"Length mismatch: {len(baseline_scores)} vs {len(contender_scores)}")
    if len(baseline_scores) < 3:
        return {"error": "n too small for Wilcoxon (need >=3)"}
    deltas = [c - b for c, b in zip(contender_scores, baseline_scores)]
    if all(d == 0 for d in deltas):
        return {"statistic": 0, "pvalue": 1.0, "mean_delta": 0, "median_delta": 0,
                "interpretation": "all equal"}
    try:
        result = stats.wilcoxon(baseline_scores, contender_scores)
        stat, pval = result.statistic, result.pvalue
    except Exception as e:
        return {"error": str(e)}
    mean_d = st.mean(deltas)
    median_d = st.median(deltas)
    if pval < 0.05:
        interp = f"SIGNIFICANT difference (p={pval:.4f}). Contender {'better' if mean_d > 0 else 'worse'} by median {median_d:.4f}"
    else:
        interp = f"NOT significant (p={pval:.4f}). Observed delta {median_d:.4f} within noise."
    return {
        "statistic": float(stat),
        "pvalue": float(pval),
        "mean_delta": mean_d,
        "median_delta": median_d,
        "interpretation": interp,
    }


# ---------------------------------------------------------- analyze R3

def analyze_r3_fair_subset():
    """Re-analyze R3 fair subset (n=11 all-contenders-survived) with:
    - BERTScore cosine for each sample
    - Paired Wilcoxon on ROUGE-L
    - Paired Wilcoxon on BERTScore
    """
    rows = json.load(open(pathlib.Path(__file__).parent.parent / "results" / "r3" / "rows.json"))
    by_sample = {}
    for r in rows:
        if r.get("extra", {}).get("error"):
            continue
        sid = r["sample_id"]
        by_sample.setdefault(sid, {})[r["contender"]] = r

    all_three = [s for s, d in by_sample.items() if len(d) == 3]
    print(f"Fair subset: {len(all_three)} samples where all 3 contenders survived")

    # Pair baseline vs compressor on these samples
    baseline_rouge = []
    comp_rouge = []
    baseline_bert = []
    comp_bert = []

    for sid in all_three:
        b = by_sample[sid]["baseline-langchain"]
        c = by_sample[sid]["nexus-v09-smart-compressor"]
        # ROUGE
        baseline_rouge.append(b["rouge_l_f1"])
        comp_rouge.append(c["rouge_l_f1"])
        # BERTScore: compare SUMMARY to REFERENCE
        b_summary = b.get("summary", "")
        c_summary = c.get("summary", "")
        ref = b.get("reference", "")
        if b_summary and c_summary and ref:
            baseline_bert.append(bertscore_cosine(b_summary, ref))
            comp_bert.append(bertscore_cosine(c_summary, ref))

    print(f"\n=== ROUGE-L (baseline vs compressor) ===")
    print(f"  baseline median: {st.median(baseline_rouge):.4f}")
    print(f"  compressor median: {st.median(comp_rouge):.4f}")
    print(f"  Wilcoxon: {paired_wilcoxon(baseline_rouge, comp_rouge)['interpretation']}")

    if baseline_bert:
        print(f"\n=== BERTScore cosine (semantic similarity to reference) ===")
        print(f"  baseline median: {st.median(baseline_bert):.4f}")
        print(f"  compressor median: {st.median(comp_bert):.4f}")
        print(f"  Wilcoxon: {paired_wilcoxon(baseline_bert, comp_bert)['interpretation']}")

    return {
        "n": len(all_three),
        "rouge_wilcoxon": paired_wilcoxon(baseline_rouge, comp_rouge),
        "bertscore_wilcoxon": paired_wilcoxon(baseline_bert, comp_bert) if baseline_bert else None,
        "baseline_rouge_median": st.median(baseline_rouge),
        "comp_rouge_median": st.median(comp_rouge),
        "baseline_bert_median": st.median(baseline_bert) if baseline_bert else None,
        "comp_bert_median": st.median(comp_bert) if baseline_bert else None,
    }


if __name__ == "__main__":
    result = analyze_r3_fair_subset()
    out = pathlib.Path(__file__).parent.parent / "results" / "r3" / "quality_analysis.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWritten to {out}")
