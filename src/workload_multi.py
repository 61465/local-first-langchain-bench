"""
Multi-domain workloads for Phase 2 cross-domain validation.

License verification (per HuggingFace Datasets metadata):
- samsum: CC-BY-NC-4.0 (NON-COMMERCIAL — use for research only)
- billsum: CC0-1.0 (public domain — safe for RFC)
- multi_news: Apache 2.0 (safe for RFC)

For RFC commercial-safe publication, we use:
- CNN/DailyMail (Apache 2.0) — primary
- billsum (CC0) — legal domain
- multi_news (Apache) — multi-doc news
- xsum (optional, Apache) — alternative single-doc

samsum excluded from commercial publication but can be cited as "tested
internally" if we add it.
"""
from __future__ import annotations

import random
from typing import Optional


def _safe_load(name: str, config: Optional[str], split: str, n: int, seed: int) -> list[dict]:
    try:
        from datasets import load_dataset
        ds = load_dataset(name, config, split=split, streaming=False) if config else \
             load_dataset(name, split=split, streaming=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load {name}: {e}")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    return [ds[i] for i in idxs]


def load_billsum(n: int = 20, seed: int = 42) -> list[dict]:
    """US congressional bills. CC0 license. Long legal text."""
    raw = _safe_load("billsum", None, "test", n * 3, seed)
    out = []
    for item in raw:
        article = item.get("text", "")
        ref = item.get("summary", "")
        if 500 <= len(article) <= 25000 and len(ref) > 100:
            out.append({
                "id": item.get("title", "")[:40] or f"billsum_{len(out)}",
                "article": article,
                "reference_summary": ref,
            })
        if len(out) >= n:
            break
    return out


def load_multi_news(n: int = 20, seed: int = 42) -> list[dict]:
    """Multi-document news summarization. Apache 2.0. Cluster of 2-10 articles."""
    raw = _safe_load("multi_news", None, "test", n * 3, seed)
    out = []
    for item in raw:
        article = item.get("document", "") or item.get("article", "")
        ref = item.get("summary", "")
        if 500 <= len(article) <= 25000 and len(ref) > 100:
            out.append({
                "id": f"multinews_{len(out)}",
                "article": article,
                "reference_summary": ref,
            })
        if len(out) >= n:
            break
    return out


def load_xsum(n: int = 20, seed: int = 42) -> list[dict]:
    """Extreme single-document summarization (very short summaries).
    Apache 2.0."""
    raw = _safe_load("xsum", None, "test", n * 3, seed)
    out = []
    for item in raw:
        article = item.get("document", "")
        ref = item.get("summary", "")
        if 500 <= len(article) <= 15000 and len(ref) > 30:
            out.append({
                "id": item.get("id", f"xsum_{len(out)}"),
                "article": article,
                "reference_summary": ref,
            })
        if len(out) >= n:
            break
    return out


# Registry — the datasets the RFC publishes against
COMMERCIAL_SAFE_WORKLOADS = {
    "cnn_dailymail": None,  # loaded via workload_summarization
    "billsum": load_billsum,
    "multi_news": load_multi_news,
    "xsum": load_xsum,
}


if __name__ == "__main__":
    import statistics as st, sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from compressor_v2 import extractive_compress_v2
    from token_measurement import compression_ratio

    for name in ["billsum", "multi_news", "xsum"]:
        print(f"\n=== {name} (n=10) ===")
        try:
            data = COMMERCIAL_SAFE_WORKLOADS[name](n=10)
            print(f"  loaded {len(data)} samples")
            ratios = []
            for d in data:
                orig = d["article"]
                comp = extractive_compress_v2(orig)
                m = compression_ratio(orig, comp)
                ratios.append(m["token_reduction"] * 100)
            print(f"  v2 token reduction median: {st.median(ratios):.1f}% (range {min(ratios):.1f}-{max(ratios):.1f}%)")
        except Exception as e:
            print(f"  FAILED: {e}")
