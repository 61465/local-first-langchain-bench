"""R8 — n=100 expansion on gpt-oss-20b (cheap tier, large sample).

Purpose: raise statistical power from 13% to 80%+ for small effect sizes.
Previous analysis: n=11 needs n>=88 for 80% power at d=0.3 (Cohen small).
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import time
import urllib.request
from datasets import load_dataset

from runner_r3 import extractive_compress

# env
_ENV = pathlib.Path("D:/project/ai/.env")
for line in _ENV.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

GROQ_KEY = os.environ["GROQ_API_KEY"]
MODEL = "openai/gpt-oss-20b"
OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "r8"
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


def main(n_target=100):
    print(f"Loading CNN/DM test split...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    # Deterministic shuffle, seed 42
    random.seed(42)
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    # Resume support
    out_path = OUT_DIR / "rows.json"
    rows = []
    done_sample_ids = set()
    if out_path.exists():
        try:
            rows = json.load(open(out_path, encoding="utf-8"))
            done_sample_ids = {(r["sample_id"], r["mode"]) for r in rows if r.get("ok")}
            print(f"Resuming — {len(done_sample_ids)} entries already complete")
        except Exception:
            pass

    done_count = len({sid for (sid, _) in done_sample_ids})
    target_remaining = n_target - done_count
    print(f"Need {target_remaining} more paired samples")

    collected = 0
    for idx in idxs:
        if collected >= target_remaining:
            break
        item = ds[idx]
        sid = item["id"]
        if (sid, "baseline") in done_sample_ids and (sid, "compressor") in done_sample_ids:
            continue

        article = item["article"]
        if len(article) < 500 or len(article) > 6000:
            continue  # keep sample size manageable

        compressed = extractive_compress(article)
        print(f"\n--- {collected+1}/{target_remaining} id={sid[:8]} article={len(article)}ch comp={len(compressed)}ch ---")

        pair_ok = True
        for mode, text in [("baseline", article), ("compressor", compressed)]:
            if (sid, mode) in done_sample_ids:
                print(f"  {mode:10s} already done, skip")
                continue
            res = groq(
                "You are a concise news summarizer. Produce a 3-sentence summary.",
                f"Summarize this article:\n\n{text}",
            )
            # exponential backoff on rate-limit
            attempts = 0
            while not res["ok"] and "429" in str(res.get("err", "")) and attempts < 3:
                attempts += 1
                wait = 15 * (2 ** attempts)
                print(f"    429 backoff {wait}s")
                time.sleep(wait)
                res = groq(
                    "You are a concise news summarizer. Produce a 3-sentence summary.",
                    f"Summarize this article:\n\n{text}",
                )

            row = {
                "sample_id": sid,
                "mode": mode,
                "article_len_chars": len(article),
                "sent_len_chars": len(text),
                "sha256_input": hashlib.sha256(text.encode()).hexdigest(),
                "reference": item["highlights"],
                **res,
            }
            rows.append(row)
            if res["ok"]:
                print(f"  {mode:10s} OK  tokens_in={res['tokens_in']} wall={res['wall']:.2f}s")
            else:
                print(f"  {mode:10s} FAIL {res.get('err','')[:70]}")
                pair_ok = False

            # persist after each call
            out_path.write_text(json.dumps(rows, indent=2, default=str))
            time.sleep(4)

        if pair_ok:
            collected += 1

    print(f"\n✓ R8 complete. Total rows: {len(rows)}")


if __name__ == "__main__":
    main(n_target=100)
