"""R14: Cross-provider validation of HSTC via Gemini's native countTokens API.

Uses the OAuth bearer token supplied at runtime (never saved to disk).
Loads the same 86 CNN/DailyMail articles as R11.
For each article, counts tokens for baseline vs HSTC-compressed via Gemini's
tokenizer. Reports median reduction + distribution.
"""
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compressor_hstc import hstc_compress

TOKEN = os.environ.get("GEMINI_OAUTH_TOKEN", "").strip()
MODEL = "gemini-2.5-flash"
ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:countTokens?key={TOKEN}"


def count_tokens(text: str) -> int | None:
    payload = json.dumps({"contents": [{"parts": [{"text": text}]}]}).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["totalTokens"]
    except urllib.error.HTTPError as e:
        print(f"[HTTP {e.code}] {e.read()[:200].decode('utf-8', errors='replace')}")
        return None
    except Exception as e:
        print(f"[err] {type(e).__name__}: {e}")
        return None


def main():
    if not TOKEN:
        print("ERROR: set GEMINI_OAUTH_TOKEN in env.")
        sys.exit(1)

    # Load the SAME 86 articles R11 used (fail-closed on dataset miss).
    r11_rows = json.load(open("D:/project/local-first-langchain-bench/results/r11/rows.json"))
    sample_ids = [r["sample_id"] for r in r11_rows if r.get("ok") and r.get("mode") == "hstc"]
    print(f"Target article count: {len(sample_ids)}")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    article_by_id = {row["id"]: row["article"] for row in ds if row["id"] in set(sample_ids)}
    print(f"Matched {len(article_by_id)}/{len(sample_ids)} articles.")

    results = []
    for i, sid in enumerate(sample_ids, 1):
        article = article_by_id.get(sid)
        if not article:
            continue
        compressed = hstc_compress(article)
        base_tok = count_tokens(article)
        time.sleep(0.5)  # rate-friendly
        comp_tok = count_tokens(compressed)
        if base_tok is None or comp_tok is None:
            continue
        reduction = (base_tok - comp_tok) / base_tok
        results.append({
            "sample_id": sid,
            "base_tokens": base_tok,
            "hstc_tokens": comp_tok,
            "reduction": reduction,
            "char_reduction": 1 - len(compressed) / len(article),
        })
        print(f"[{i:>3}/{len(sample_ids)}] base={base_tok} hstc={comp_tok} reduction={reduction:.1%}")
        time.sleep(0.5)

    if not results:
        print("No successful samples.")
        sys.exit(1)

    reductions = [r["reduction"] for r in results]
    median = statistics.median(reductions)
    mean = statistics.mean(reductions)
    stdev = statistics.stdev(reductions) if len(reductions) > 1 else 0.0
    direction_stable = sum(1 for r in reductions if r > 0)

    summary = {
        "provider": "google-gemini",
        "model": MODEL,
        "endpoint": "countTokens",
        "n": len(results),
        "median_reduction": median,
        "mean_reduction": mean,
        "stdev_reduction": stdev,
        "direction_stable": f"{direction_stable}/{len(results)}",
        "min": min(reductions),
        "max": max(reductions),
    }

    out_dir = "D:/project/local-first-langchain-bench/results/r14"
    os.makedirs(out_dir, exist_ok=True)
    json.dump(results, open(f"{out_dir}/rows.json", "w"), indent=2)
    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2)

    print("\n=== GEMINI CROSS-PROVIDER VALIDATION ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
