"""R6 — Multi-model-family benchmark.

Reuses the SAME 11 fair-subset articles from R3. Runs baseline (uncompressed)
and compressor (v2a extractive) through 3 Groq-hosted model families:
  - openai/gpt-oss-120b (OpenAI family, large)
  - openai/gpt-oss-20b (OpenAI family, small)
  - qwen/qwen3.8-27b (Alibaba family)

Measures tokens_in reduction per sample per model.
Honest framing: same provider (Groq infrastructure), 3 model families.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import statistics as st
import time
import urllib.error
import urllib.request

from runner_r3 import extractive_compress  # v2a — the production compressor

# Load env
_ENV_FILE = pathlib.Path("D:/project/ai/.env")
for line in _ENV_FILE.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

GROQ_KEY = os.environ["GROQ_API_KEY"]
R3_ROWS = pathlib.Path(__file__).parent.parent / "results" / "r3" / "rows.json"
OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "r6"
OUT_DIR.mkdir(exist_ok=True)

MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]


def groq_call(model: str, article: str) -> dict:
    """Single summarization call. Returns tokens_in/out/wall."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise news summarizer. Produce a 3-sentence summary."},
            {"role": "user", "content": f"Summarize this article:\n\n{article}"},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {GROQ_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "lclb-benchmark/1.0 (+https://github.com/61465)",
        },
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
        wall = time.monotonic() - t0
        return {
            "ok": True,
            "tokens_in": resp["usage"]["prompt_tokens"],
            "tokens_out": resp["usage"]["completion_tokens"],
            "summary": resp["choices"][0]["message"]["content"],
            "wall": wall,
        }
    except urllib.error.HTTPError as e:
        wall = time.monotonic() - t0
        body_txt = e.read().decode(errors="replace")[:200]
        return {"ok": False, "status": e.code, "err": body_txt, "wall": wall}
    except Exception as e:
        return {"ok": False, "err": str(e), "wall": time.monotonic() - t0}


def load_fair_articles():
    """Grab the 11 articles from R3 fair-subset — one per sample_id.
    Articles are not stored in R3 rows (only metadata). Reload CNN/DailyMail
    and filter by sample_id (which is the CNN/DM `id` field).
    """
    rows = json.load(open(R3_ROWS, encoding="utf-8"))
    by = {}
    for r in rows:
        if r.get("extra", {}).get("error"):
            continue
        by.setdefault(r["sample_id"], {})[r["contender"]] = r
    fair_ids = {sid for sid, d in by.items()
                if "baseline-langchain" in d and "nexus-v09-smart-compressor" in d}

    from datasets import load_dataset
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    ds_by_id = {row["id"]: row for row in ds if row["id"] in fair_ids}
    out = []
    for sid in fair_ids:
        if sid in ds_by_id:
            r = ds_by_id[sid]
            out.append({"sample_id": sid, "article": r["article"], "reference": r["highlights"]})
    print(f"Matched {len(out)}/{len(fair_ids)} fair-subset articles from CNN/DM test split")
    return out


def main():
    articles = load_fair_articles()
    print(f"Loaded {len(articles)} fair-subset articles")
    if not articles:
        print("No articles found in R3 rows — bailing.")
        return

    rows = []
    for i, a in enumerate(articles):
        print(f"\n--- sample {i+1}/{len(articles)} {a['sample_id'][:8]} ---")
        compressed = extractive_compress(a["article"])
        comp_ratio = len(compressed) / max(1, len(a["article"]))
        print(f"  article={len(a['article'])}ch, compressed={len(compressed)}ch ({comp_ratio*100:.1f}%)")
        for model in MODELS:
            for mode, text in [("baseline", a["article"]), ("compressor", compressed)]:
                res = groq_call(model, text)
                row = {
                    "sample_id": a["sample_id"],
                    "model": model,
                    "mode": mode,
                    "article_len_chars": len(a["article"]),
                    "sent_len_chars": len(text),
                    "sha256_input": hashlib.sha256(text.encode()).hexdigest(),
                    **res,
                }
                rows.append(row)
                status = "OK" if res["ok"] else f"ERR[{res.get('status','')}]"
                tok = res.get("tokens_in", "-")
                print(f"  {model[:30]:30s}  {mode:10s}  {status}  tokens_in={tok}  wall={res['wall']:.2f}s")
                time.sleep(2)  # gentle rate-limit respect

    out_path = OUT_DIR / "rows.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n✓ Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
