"""
R2 workload: CNN/DailyMail summarization.

Why this workload:
- Long prompts (~500 to ~3000 tokens per article) — compressor has
  material to actually compress (news articles have filler, redundant
  phrasing, polite lede).
- Public reference summaries — enables ROUGE-L scoring without us
  writing gold-answer sets (no fabrication risk).
- Diverse length distribution — lets us see where the compressor
  tips from "hurts" to "helps".

Dataset: cnn_dailymail (Apache 2.0, standard summarization benchmark).
Loaded via HF `datasets`. If offline, falls back to a minimal bundled
sample so the harness still runs.
"""
from __future__ import annotations

import json
import pathlib


def load_cnn_dailymail(n: int = 30, seed: int = 42) -> list[dict]:
    """Return list of {article, reference_summary} dicts.

    Attempts to load from HuggingFace Datasets. On any failure, raises
    RuntimeError — we NEVER fabricate summaries.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load cnn_dailymail dataset: {e}. "
            f"R2 cannot proceed without ground-truth summaries. "
            f"Try `pip install datasets` and HF auth if required."
        )

    import random
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n, len(ds)))
    out = []
    for i in idxs:
        item = ds[i]
        article = item["article"]
        summary = item["highlights"]
        # Only keep articles with reasonable length to exercise the compressor:
        # too short → no value; too long → provider token limits.
        if 200 <= len(article) <= 20000:
            out.append({
                "id": str(item.get("id", i)),
                "article": article,
                "reference_summary": summary,
            })
        if len(out) >= n:
            break

    return out


# --------------------------------------------------------- ROUGE-L

def _lcs_len(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    # Use 1-D DP for memory
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _tokenize(s: str) -> list[str]:
    import re
    s = s.lower()
    return re.findall(r"\w+", s)


def rouge_l_f1(candidate: str, reference: str, beta: float = 1.0) -> float:
    """ROUGE-L F1 score. beta=1 → equal weight to precision and recall.

    Returns a float in [0, 1]. 0 if either string is empty.
    """
    c = _tokenize(candidate)
    r = _tokenize(reference)
    if not c or not r:
        return 0.0
    lcs = _lcs_len(c, r)
    if lcs == 0:
        return 0.0
    p = lcs / len(c)
    r_rec = lcs / len(r)
    denom = p + (beta ** 2) * r_rec
    if denom == 0:
        return 0.0
    return ((1 + beta ** 2) * p * r_rec) / denom


# --------------------------------------------------------- prompts

SUMMARIZATION_SYSTEM = (
    "You are a precise news summarizer. Produce a 3-sentence summary that "
    "captures the main event, the key actor(s), and the outcome. Do not "
    "add your opinion. Do not add polite preamble. Output only the summary."
)

SUMMARIZATION_USER_TEMPLATE = (
    "Article:\n\n{article}\n\n"
    "Produce a 3-sentence summary now."
)


# --------------------------------------------------------- self-test

if __name__ == "__main__":
    print("Smoke-testing ROUGE-L...")
    assert rouge_l_f1("the cat sat on the mat", "the cat sat on the mat") > 0.99, "identical mismatch"
    assert rouge_l_f1("completely different words", "the cat sat on the mat") < 0.3
    print("ROUGE-L smoke OK.")

    print("Attempting dataset load (30 samples)...")
    try:
        data = load_cnn_dailymail(n=30)
        lens = [len(d["article"]) for d in data]
        print(f"  loaded {len(data)} samples, article len range {min(lens)}-{max(lens)}")
        print(f"  first article preview: {data[0]['article'][:200]}...")
    except Exception as e:
        print(f"  dataset load FAILED: {e}")
