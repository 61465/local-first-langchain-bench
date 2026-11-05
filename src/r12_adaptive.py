"""R12 — HSTC-Adaptive benchmark on dialogue workload.

Compare v2a vs HSTC vs HSTC-Adaptive on 10 DialogSum samples.
Target: HSTC-Adaptive should (a) detect "dialogue" correctly and
(b) deliver SOME compression while preserving speaker structure.
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

from runner_r3 import extractive_compress as v2a
from compressor_hstc import hstc_compress
from compressor_hstc_adaptive import hstc_adaptive_compress, detect_workload

_ENV = pathlib.Path("D:/project/ai/.env")
for line in _ENV.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

GROQ_KEY = os.environ["GROQ_API_KEY"]
MODEL = "openai/gpt-oss-120b"
OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "r12"
OUT_DIR.mkdir(exist_ok=True)


def groq(system, user):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
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
        body = e.read().decode(errors="replace")[:200] if hasattr(e, "read") else ""
        return {"ok": False, "err": str(e) + " " + body, "wall": time.monotonic() - t0}


def main():
    from datasets import load_dataset
    ds = load_dataset("knkarthick/dialogsum", split="test", trust_remote_code=True)
    random.seed(42)
    idxs = random.sample(range(len(ds)), 10)

    rows = []
    for i, idx in enumerate(idxs):
        item = ds[idx]
        dialogue = item["dialogue"]
        ref = item["summary"]

        v_text = v2a(dialogue)
        h_text = hstc_compress(dialogue)
        a_text = hstc_adaptive_compress(dialogue)
        wl_detected = detect_workload(dialogue)

        print(f"\n--- {i+1}/10 id={item['id']} dlg={len(dialogue)}ch detected={wl_detected} ---")
        print(f"  v2a:    {len(v_text)}ch ({(1-len(v_text)/len(dialogue))*100:.1f}%)")
        print(f"  hstc:   {len(h_text)}ch ({(1-len(h_text)/len(dialogue))*100:.1f}%)")
        print(f"  adapt:  {len(a_text)}ch ({(1-len(a_text)/len(dialogue))*100:.1f}%)")

        for mode, text in [("baseline", dialogue), ("v2a", v_text), ("hstc", h_text), ("adaptive", a_text)]:
            res = groq("You summarize short dialogues in 1-2 sentences.",
                       f"Summarize this dialogue:\n\n{text}")
            attempts = 0
            while not res["ok"] and "429" in str(res.get("err", "")) and attempts < 2:
                attempts += 1
                time.sleep(15 * (2 ** attempts))
                res = groq("You summarize short dialogues in 1-2 sentences.",
                           f"Summarize this dialogue:\n\n{text}")
            rows.append({
                "sample_id": item["id"], "mode": mode,
                "sent_len_chars": len(text),
                "detected_workload": wl_detected,
                "sha256_input": hashlib.sha256(text.encode()).hexdigest(),
                "reference": ref, **res,
            })
            if res["ok"]:
                print(f"    {mode:10s} OK  tokens_in={res['tokens_in']} wall={res['wall']:.2f}s")
            time.sleep(4)

    out = OUT_DIR / "rows.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n✓ Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
