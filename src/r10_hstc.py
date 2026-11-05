"""R10 — HSTC benchmark on R3 fair subset.

Compare: baseline-langchain, v2a (current production), HSTC (new hybrid).
Same model (gpt-oss-120b), same 11 fair-subset articles.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
import urllib.request
from datasets import load_dataset

from runner_r3 import extractive_compress as v2a
from compressor_hstc import hstc_compress

# env
_ENV = pathlib.Path("D:/project/ai/.env")
for line in _ENV.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

GROQ_KEY = os.environ["GROQ_API_KEY"]
MODEL = "openai/gpt-oss-120b"
OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "r10"
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


def rouge_l_f1(candidate: str, reference: str) -> float:
    """Simple LCS-based ROUGE-L F1 — pure Python."""
    c_tokens = candidate.lower().split()
    r_tokens = reference.lower().split()
    if not c_tokens or not r_tokens:
        return 0.0
    m, n = len(c_tokens), len(r_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if c_tokens[i - 1] == r_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision = lcs / len(c_tokens)
    recall = lcs / len(r_tokens)
    return 2 * precision * recall / (precision + recall)


def load_fair_articles():
    rows = json.load(open(pathlib.Path(__file__).parent.parent / "results" / "r3" / "rows.json", encoding="utf-8"))
    by = {}
    for r in rows:
        if r.get("extra", {}).get("error"):
            continue
        by.setdefault(r["sample_id"], {})[r["contender"]] = r
    fair_ids = {sid for sid, d in by.items()
                if "baseline-langchain" in d and "nexus-v09-smart-compressor" in d}
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test", trust_remote_code=True)
    out = []
    for row in ds:
        if row["id"] in fair_ids:
            out.append({"sample_id": row["id"], "article": row["article"], "reference": row["highlights"]})
    return out


def main():
    articles = load_fair_articles()
    print(f"Loaded {len(articles)} fair-subset articles\n")

    rows = []
    for i, a in enumerate(articles):
        sid = a["sample_id"]
        art = a["article"]
        ref = a["reference"]

        # Compress with both methods
        t0 = time.monotonic()
        v2a_text = v2a(art)
        v2a_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        hstc_text = hstc_compress(art)
        hstc_ms = (time.monotonic() - t0) * 1000

        print(f"--- {i+1}/{len(articles)} {sid[:8]} article={len(art)}ch ---")
        print(f"  v2a:  {len(v2a_text)}ch ({(1-len(v2a_text)/len(art))*100:.1f}%)  {v2a_ms:.1f}ms")
        print(f"  hstc: {len(hstc_text)}ch ({(1-len(hstc_text)/len(art))*100:.1f}%)  {hstc_ms:.1f}ms")

        for mode, text, comp_ms in [("baseline", art, 0), ("v2a", v2a_text, v2a_ms), ("hstc", hstc_text, hstc_ms)]:
            res = groq("You are a concise news summarizer. Produce a 3-sentence summary.",
                       f"Summarize this article:\n\n{text}")
            # backoff
            attempts = 0
            while not res["ok"] and "429" in str(res.get("err", "")) and attempts < 3:
                attempts += 1
                time.sleep(15 * (2 ** attempts))
                res = groq("You are a concise news summarizer. Produce a 3-sentence summary.",
                           f"Summarize this article:\n\n{text}")
            row = {
                "sample_id": sid,
                "mode": mode,
                "article_len_chars": len(art),
                "sent_len_chars": len(text),
                "compression_wall_ms": comp_ms,
                "sha256_input": hashlib.sha256(text.encode()).hexdigest(),
                "reference": ref,
                **res,
            }
            if res.get("ok"):
                row["rouge_l_f1"] = rouge_l_f1(res["summary"], ref)
            rows.append(row)
            status = "OK" if res["ok"] else f"FAIL[{res.get('err','')[:40]}]"
            tok = res.get("tokens_in", "-")
            rouge = f" ROUGE={row.get('rouge_l_f1', 0):.3f}" if res["ok"] else ""
            print(f"    {mode:10s} {status}  tokens_in={tok}  wall={res['wall']:.2f}s{rouge}")
            time.sleep(5)

    out = OUT_DIR / "rows.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n✓ Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
