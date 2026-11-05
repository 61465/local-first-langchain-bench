"""R9 — LLMLingua head-to-head comparison.

Compare our deterministic extractive compressor (v2a) against Microsoft's
LLMLingua (EMNLP 2023) on the same R3 fair-subset articles.

Metric: tokens_in reduction + API latency + quality (ROUGE-L via API output).
Same Groq model (gpt-oss-120b) across both compressors for apples-to-apples.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import statistics as st
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
MODEL = "openai/gpt-oss-120b"
OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "r9"
OUT_DIR.mkdir(exist_ok=True)


# ------------------------------- LLMLingua ---------------------------------

_LLMLINGUA = None


def get_llmlingua():
    global _LLMLINGUA
    if _LLMLINGUA is None:
        from llmlingua import PromptCompressor
        # Small lightweight model — llmlingua default + offline-friendly
        _LLMLINGUA = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu",
        )
    return _LLMLINGUA


def compress_llmlingua(article: str, target_ratio: float = 0.65) -> str:
    """Compress with LLMLingua-2 to retain roughly target_ratio of tokens."""
    pc = get_llmlingua()
    result = pc.compress_prompt(
        article,
        rate=target_ratio,
        force_tokens=["\n", "?", ".", "!"],
    )
    return result["compressed_prompt"]


# ------------------------------- Groq ---------------------------------

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


# ------------------------------- Load fair subset ---------------------------

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


# ------------------------------- Main ---------------------------------

def main():
    articles = load_fair_articles()
    print(f"Loaded {len(articles)} fair-subset articles")

    # Preload LLMLingua once (slow — model download)
    print("Loading LLMLingua-2 (first run downloads ~470MB)...")
    get_llmlingua()
    print("✓ LLMLingua ready")

    rows = []
    for i, a in enumerate(articles):
        sid = a["sample_id"]
        print(f"\n--- {i+1}/{len(articles)} sample {sid[:8]} article={len(a['article'])}ch ---")

        # Three contenders, same article
        t0 = time.monotonic()
        ours = extractive_compress(a["article"])
        ours_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        try:
            ll = compress_llmlingua(a["article"], target_ratio=0.65)
            ll_ms = (time.monotonic() - t0) * 1000
            ll_ok = True
        except Exception as e:
            ll = a["article"]
            ll_ms = 0
            ll_ok = False
            print(f"  LLMLingua ERR: {e}")

        print(f"  ours:      {len(a['article'])}→{len(ours)}ch ({len(ours)/len(a['article'])*100:.1f}%)  {ours_ms:.1f}ms")
        print(f"  llmlingua: {len(a['article'])}→{len(ll)}ch ({len(ll)/len(a['article'])*100:.1f}%)  {ll_ms:.1f}ms")

        for mode, text, comp_ms in [("baseline", a["article"], 0),
                                     ("ours", ours, ours_ms),
                                     ("llmlingua", ll, ll_ms)]:
            res = groq("You are a concise news summarizer. Produce a 3-sentence summary.",
                       f"Summarize this article:\n\n{text}")
            # backoff on 429
            attempts = 0
            while not res["ok"] and "429" in str(res.get("err","")) and attempts < 2:
                attempts += 1
                time.sleep(15 * (2 ** attempts))
                res = groq("You are a concise news summarizer. Produce a 3-sentence summary.",
                           f"Summarize this article:\n\n{text}")
            row = {
                "sample_id": sid,
                "mode": mode,
                "article_len_chars": len(a["article"]),
                "sent_len_chars": len(text),
                "compression_wall_ms": comp_ms,
                "sha256_input": hashlib.sha256(text.encode()).hexdigest(),
                "reference": a["reference"],
                **res,
            }
            rows.append(row)
            status = "OK" if res["ok"] else "FAIL"
            print(f"    {mode:10s} {status}  tokens_in={res.get('tokens_in','-')}  wall={res['wall']:.2f}s")
            time.sleep(6)

    out = OUT_DIR / "rows.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n✓ Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
