"""R15: Third-provider validation via tiktoken cl100k_base (OpenAI/Claude-family BPE approximation).

This is explicitly an APPROXIMATION of Claude's tokenizer — Anthropic's exact
tokenizer is not public; cl100k_base is the same BPE family and gives the
closest open-source approximation.
"""
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compressor_hstc import hstc_compress

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def main():
    r11_rows = json.load(open("D:/project/local-first-langchain-bench/results/r11/rows.json"))
    sample_ids = [r["sample_id"] for r in r11_rows if r.get("ok") and r.get("mode") == "hstc"]
    print(f"Target article count: {len(sample_ids)}")

    from datasets import load_dataset
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    article_by_id = {row["id"]: row["article"] for row in ds if row["id"] in set(sample_ids)}
    print(f"Matched {len(article_by_id)}/{len(sample_ids)} articles.")

    results = []
    for i, sid in enumerate(sample_ids, 1):
        article = article_by_id.get(sid)
        if not article:
            continue
        compressed = hstc_compress(article)
        base_tok = len(ENC.encode(article))
        comp_tok = len(ENC.encode(compressed))
        reduction = (base_tok - comp_tok) / base_tok
        results.append({
            "sample_id": sid,
            "base_tokens": base_tok,
            "hstc_tokens": comp_tok,
            "reduction": reduction,
        })

    reductions = [r["reduction"] for r in results]
    summary = {
        "provider": "tiktoken-cl100k-approximation",
        "tokenizer": "cl100k_base (BPE, OpenAI GPT-4 family; approximates Claude)",
        "n": len(results),
        "median_reduction": statistics.median(reductions),
        "mean_reduction": statistics.mean(reductions),
        "stdev_reduction": statistics.stdev(reductions),
        "direction_stable": f"{sum(1 for r in reductions if r>0)}/{len(reductions)}",
        "min": min(reductions),
        "max": max(reductions),
    }

    out_dir = "D:/project/local-first-langchain-bench/results/r15"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(results, open(f"{out_dir}/rows.json", "w"), indent=2)
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2)

    print("\n=== TIKTOKEN cl100k VALIDATION (Claude approximation) ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
