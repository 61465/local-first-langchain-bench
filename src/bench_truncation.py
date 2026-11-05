"""LangChain-style naive truncation baseline — Fix #7 for NEXUS v5.2.

Compares HSTC against the first-N-chars truncation strategy that LangChain users
typically reach for by default (e.g., ConversationBufferWindowMemory).
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import tiktoken

# Use same R3 fair subset articles
from datasets import load_dataset
from runner_r3 import extractive_compress as v2a
from compressor_hstc import hstc_compress

OUT = pathlib.Path(__file__).parent.parent / "results" / "truncation_baseline.json"


def truncate_tokens(text: str, max_tokens: int = 485, encoder=None) -> str:
    """Truncate to first N tokens — the naive LangChain default."""
    if encoder is None:
        encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    truncated = tokens[:max_tokens]
    return encoder.decode(truncated)


def main():
    rows = json.load(open(pathlib.Path(__file__).parent.parent / "results" / "r3" / "rows.json", encoding="utf-8"))
    by = {}
    for r in rows:
        if r.get("extra", {}).get("error"):
            continue
        by.setdefault(r["sample_id"], {})[r["contender"]] = r
    fair_ids = {sid for sid, d in by.items()
                if "baseline-langchain" in d and "nexus-v09-smart-compressor" in d}

    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    articles = [row for row in ds if row["id"] in fair_ids][:11]
    print(f"Loaded {len(articles)} fair-subset articles\n")

    encoder = tiktoken.get_encoding("cl100k_base")
    baseline_tokens, trunc_tokens, v2a_tokens, hstc_tokens = [], [], [], []

    for a in articles:
        art = a["article"]
        b = len(encoder.encode(art))
        baseline_tokens.append(b)

        # Truncation to match v2a's median (485 tokens)
        tr = truncate_tokens(art, max_tokens=485, encoder=encoder)
        trunc_tokens.append(len(encoder.encode(tr)))

        v = v2a(art)
        v2a_tokens.append(len(encoder.encode(v)))

        h = hstc_compress(art)
        hstc_tokens.append(len(encoder.encode(h)))

    def stats(name, tokens, base=None):
        red = [(b - t) / b for b, t in zip(baseline_tokens, tokens)] if base is None else [(base[i] - t) / base[i] for i, t in enumerate(tokens)]
        print(f"  {name:25s} median tokens={st.median(tokens):4.0f}  reduction vs baseline: median={st.median(red)*100:5.1f}%")
        return st.median(red) * 100

    print("Token-count comparison (n=11 fair-subset articles):")
    stats("baseline (full article)", baseline_tokens)
    trunc_red = stats("LangChain truncate(485)", trunc_tokens)
    v2a_red = stats("v2a extractive", v2a_tokens)
    hstc_red = stats("HSTC (ours)", hstc_tokens)

    print(f"\n[Semantic coverage note]")
    print(f"  Truncation keeps the FIRST 485 tokens — drops everything after (lead/tail lost).")
    print(f"  Extractive/HSTC keep high-information sentences distributed across the article.")
    print(f"  For news articles, this means truncation LOSES the inverted-pyramid tail.")
    print(f"  Comparable reduction % does NOT equal comparable quality. See §5.7 ROUGE comparison.")

    out = {
        "n": len(articles),
        "baseline_median_tokens": st.median(baseline_tokens),
        "methods": {
            "truncation_485": {"median_tokens": st.median(trunc_tokens), "median_reduction_pct": trunc_red},
            "v2a": {"median_tokens": st.median(v2a_tokens), "median_reduction_pct": v2a_red},
            "hstc": {"median_tokens": st.median(hstc_tokens), "median_reduction_pct": hstc_red},
        },
        "interpretation": "Truncation and extractive compression can target similar reduction %, but truncation unconditionally loses article tail content. HSTC preserves high-information sentences distributed across the article, trading some stopword content for structural coverage.",
    }
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✓ Written to {OUT}")


if __name__ == "__main__":
    main()
