"""R13 — Multi-model HSTC validation at n=50 per model.

Replaces the "multi-provider" critique by showing HSTC works on 3 DIFFERENT
Groq-hosted model families at scale (not just 3 x 11 like R6).

Models tested:
  - openai/gpt-oss-120b   (OpenAI, 120B params)
  - openai/gpt-oss-20b    (OpenAI, 20B params)
  - qwen/qwen3.8-27b      (Alibaba, 27B params — different tokenizer family!)
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import statistics as st
import time
import urllib.request
from scipy.stats import wilcoxon

from runner_r3 import extractive_compress as v2a
from compressor_hstc import hstc_compress

# env
_ENV = pathlib.Path("D:/project/ai/.env")
for line in _ENV.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

GROQ_KEY = os.environ["GROQ_API_KEY"]
OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "r13"
OUT_DIR.mkdir(exist_ok=True)

MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]


def groq(model: str, text: str):
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise news summarizer. Produce a 3-sentence summary."},
            {"role": "user", "content": f"Summarize this article:\n\n{text}"},
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json",
                 "User-Agent": "lclb-benchmark/1.0 (+https://github.com/61465)"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
        return {"ok": True, "tokens_in": resp["usage"]["prompt_tokens"],
                "tokens_out": resp["usage"]["completion_tokens"],
                "summary": resp["choices"][0]["message"]["content"],
                "wall": time.monotonic() - t0}
    except Exception as e:
        body = e.read().decode(errors="replace")[:150] if hasattr(e, "read") else ""
        return {"ok": False, "err": str(e) + " " + body, "wall": time.monotonic() - t0}


def main(n_per_model=50):
    from datasets import load_dataset
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    random.seed(42)
    idxs = random.sample(range(len(ds)), n_per_model)
    articles = [ds[i] for i in idxs if 500 <= len(ds[i]["article"]) <= 6000][:n_per_model]
    print(f"Loaded {len(articles)} articles (500-6000 chars)")

    out_path = OUT_DIR / "rows.json"
    rows = []
    done = set()
    if out_path.exists():
        try:
            rows = json.load(open(out_path, encoding="utf-8"))
            done = {(r["sample_id"], r["model"], r["mode"]) for r in rows if r.get("ok")}
            print(f"Resume: {len(done)} already done")
        except Exception:
            pass

    total_target = len(articles) * len(MODELS) * 3  # baseline + v2a + hstc
    processed = len(done)

    for model in MODELS:
        print(f"\n=== Model: {model} ===")
        for i, item in enumerate(articles):
            sid = item["id"]
            art = item["article"]
            v_text = v2a(art)
            h_text = hstc_compress(art)

            for mode, text in [("baseline", art), ("v2a", v_text), ("hstc", h_text)]:
                key = (sid, model, mode)
                if key in done:
                    continue
                res = groq(model, text)
                attempts = 0
                while not res["ok"] and "429" in str(res.get("err", "")) and attempts < 2:
                    attempts += 1
                    time.sleep(10 * (2 ** attempts))
                    res = groq(model, text)

                row = {"sample_id": sid, "model": model, "mode": mode,
                       "article_len_chars": len(art),
                       "sent_len_chars": len(text),
                       "sha256_input": hashlib.sha256(text.encode()).hexdigest(),
                       **res}
                rows.append(row)
                done.add(key)
                processed += 1

                if res["ok"]:
                    print(f"  {i+1:3d}/{len(articles)} {model[:20]:20s} {mode:10s} tokens_in={res['tokens_in']} ({processed}/{total_target})")
                else:
                    print(f"  FAIL {model[:20]:20s} {mode:10s} {res.get('err','')[:50]}")

                out_path.write_text(json.dumps(rows, indent=2, default=str))
                time.sleep(4)

    print(f"\n✓ R13 complete. {len(rows)} total rows.")


if __name__ == "__main__":
    main(n_per_model=50)
