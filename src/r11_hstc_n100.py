"""R11 — HSTC n=100 expansion.

Run HSTC on the 100 CNN/DM samples already cached by R8.
Avoids re-fetching the dataset; reuses R8 sample_ids to maintain audit trail.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.request

from runner_r3 import extractive_compress as v2a
from compressor_hstc import hstc_compress

_ENV = pathlib.Path("D:/project/ai/.env")
for line in _ENV.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

GROQ_KEY = os.environ["GROQ_API_KEY"]
MODEL = "openai/gpt-oss-20b"
OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "r11"
OUT_DIR.mkdir(exist_ok=True)


def groq(system: str, user: str):
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
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
        return {
            "ok": True,
            "tokens_in": resp["usage"]["prompt_tokens"],
            "tokens_out": resp["usage"]["completion_tokens"],
            "summary": resp["choices"][0]["message"]["content"],
            "wall": time.monotonic() - t0,
        }
    except Exception as e:
        body = e.read().decode(errors="replace")[:200] if hasattr(e, "read") else ""
        return {"ok": False, "err": str(e) + " " + body, "wall": time.monotonic() - t0}


def main():
    # Load R8 articles (already cached) — reuse same sample set for direct comparison
    r8_rows = json.load(open(pathlib.Path(__file__).parent.parent / "results" / "r8" / "rows.json", encoding="utf-8"))
    r8_baseline = {r["sample_id"]: r for r in r8_rows if r.get("mode") == "baseline" and r.get("ok")}
    print(f"Loaded {len(r8_baseline)} R8 baseline articles")

    # Need the raw article text, which we didn't store in R8 rows. Load from CNN/DM.
    from datasets import load_dataset
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    article_by_id = {row["id"]: row for row in ds if row["id"] in r8_baseline}
    print(f"Matched {len(article_by_id)}/{len(r8_baseline)} articles from CNN/DM")

    # Resume support
    out_path = OUT_DIR / "rows.json"
    rows = []
    done_sids = set()
    if out_path.exists():
        try:
            rows = json.load(open(out_path, encoding="utf-8"))
            done_sids = {r["sample_id"] for r in rows if r.get("ok")}
        except Exception:
            pass
    print(f"Resuming: {len(done_sids)} already processed")

    target = list(article_by_id.keys())
    processed = 0
    for i, sid in enumerate(target):
        if sid in done_sids:
            continue
        item = article_by_id[sid]
        article = item["article"]
        ref = item["highlights"]

        hstc_text = hstc_compress(article)
        processed += 1
        print(f"\n--- {processed}/{len(target)-len(done_sids)} sid={sid[:8]} article={len(article)}ch hstc={len(hstc_text)}ch ({(1-len(hstc_text)/len(article))*100:.1f}%) ---")

        res = groq(
            "You are a concise news summarizer. Produce a 3-sentence summary.",
            f"Summarize this article:\n\n{hstc_text}",
        )
        attempts = 0
        while not res["ok"] and "429" in str(res.get("err","")) and attempts < 3:
            attempts += 1
            time.sleep(15 * (2 ** attempts))
            res = groq(
                "You are a concise news summarizer. Produce a 3-sentence summary.",
                f"Summarize this article:\n\n{hstc_text}",
            )

        rows.append({
            "sample_id": sid,
            "mode": "hstc",
            "article_len_chars": len(article),
            "sent_len_chars": len(hstc_text),
            "sha256_input": hashlib.sha256(hstc_text.encode()).hexdigest(),
            "reference": ref,
            **res,
        })
        if res["ok"]:
            print(f"  OK  tokens_in={res['tokens_in']} wall={res['wall']:.2f}s")
        else:
            print(f"  FAIL {res.get('err','')[:70]}")

        out_path.write_text(json.dumps(rows, indent=2, default=str))
        time.sleep(4)

    print(f"\n✓ R11 complete. Total HSTC rows: {len(rows)}")


if __name__ == "__main__":
    main()
