"""R7 — Multi-workload benchmark.

Prove compressor works beyond news summarization:
  1. Dialogue (DialogSum, MIT) — 10 conversations, baseline vs compressor
  2. Tool-call context (synthetic, LangChain-realistic) — 10 scenarios

For dialogue: the extractive compressor (news-tuned) is expected to UNDERPERFORM
on dialogue — this is the honest limitation. We report the true number and
propose a dialogue-specialized compressor as Phase 2 scope.

For tool-call: compress the SYSTEM prompt (tool definitions + context) while
leaving the user turn uncompressed. This is the natural use case.
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

from runner_r3 import extractive_compress

# Env
_ENV_FILE = pathlib.Path("D:/project/ai/.env")
for line in _ENV_FILE.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

GROQ_KEY = os.environ["GROQ_API_KEY"]
MODEL = "openai/gpt-oss-120b"
OUT_DIR = pathlib.Path(__file__).parent.parent / "results" / "r7"
OUT_DIR.mkdir(exist_ok=True)


def groq_call(system: str, user: str, max_retries: int = 3) -> dict:
    """With exponential backoff on 429."""
    for attempt in range(max_retries + 1):
        res = _groq_call(system, user)
        if res["ok"]:
            return res
        if "429" in str(res.get("err", "")) and attempt < max_retries:
            wait = 15 * (2 ** attempt)  # 15, 30, 60 seconds
            print(f"    429 → backoff {wait}s (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue
        return res
    return res


def _groq_call(system: str, user: str) -> dict:
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
        return {
            "ok": True,
            "tokens_in": resp["usage"]["prompt_tokens"],
            "tokens_out": resp["usage"]["completion_tokens"],
            "summary": resp["choices"][0]["message"]["content"],
            "wall": time.monotonic() - t0,
        }
    except Exception as e:
        body_txt = e.read().decode(errors="replace")[:200] if hasattr(e, "read") else ""
        return {"ok": False, "err": str(e) + " " + body_txt, "wall": time.monotonic() - t0}


# ------------------------------- Workload A: DIALOGUE -------------------------------

def workload_dialogue(n=10):
    """DialogSum (MIT License) — real customer service / meeting dialogues."""
    from datasets import load_dataset
    ds = load_dataset("knkarthick/dialogsum", split="test", trust_remote_code=True)
    random.seed(42)
    idxs = random.sample(range(len(ds)), n)
    out = []
    print(f"\n=== Workload A: DIALOGUE (DialogSum MIT, n={n}) ===")
    for i, idx in enumerate(idxs):
        item = ds[idx]
        dialogue = item["dialogue"]
        ref = item["summary"]
        compressed = extractive_compress(dialogue)
        print(f"\n--- A{i+1}/{n} id={item['id']} dlg={len(dialogue)}ch comp={len(compressed)}ch ---")
        for mode, text in [("baseline", dialogue), ("compressor", compressed)]:
            res = groq_call(
                "You summarize short dialogues in 1-2 sentences.",
                f"Summarize this dialogue:\n\n{text}",
            )
            row = {
                "workload": "dialogue",
                "sample_id": item["id"],
                "mode": mode,
                "sent_len_chars": len(text),
                "sha256_input": hashlib.sha256(text.encode()).hexdigest(),
                "reference": ref,
                **res,
            }
            out.append(row)
            if res["ok"]:
                print(f"  {mode:10s} OK  tokens_in={res['tokens_in']} wall={res['wall']:.2f}s")
            else:
                print(f"  {mode:10s} ERR {res.get('err','')[:80]}")
            time.sleep(8)  # respect TPM/RPM burst limits
    return out


# ------------------------------- Workload B: TOOL-CALL -------------------------------

RAG_CONTEXT = """
Reference context (retrieved from knowledge base; use ONLY what is relevant to the user's question):

Section 1 — PostgreSQL 16 release notes (official docs excerpt):
PostgreSQL 16 was released September 14, 2023 with significant improvements across the board. Logical replication now supports replication from standby servers, enabling more flexible topologies. Parallel query execution has been enhanced for aggregate functions and improved SELECT DISTINCT performance. The query planner uses more incremental sorts, especially for queries with many ORDER BY expressions. Full-text search now supports searching for phrases with the phraseto_tsquery() function for better relevance.

Section 2 — OIDC token refresh best practices (OpenID Foundation guidance):
Refresh tokens should be long-lived (days to weeks) but access tokens should be short-lived (5-15 minutes). Always use refresh token rotation: the authorization server invalidates the old refresh token when issuing a new one. Store refresh tokens in HTTP-only, SameSite=Strict cookies, never in localStorage where they are vulnerable to XSS. For single-page applications, use the authorization code flow with PKCE, not the implicit flow which is deprecated. Implement refresh token binding to prevent replay from a different client.

Section 3 — User signup analytics methodology (internal data team):
The users table tracks signup_source as one of: organic, paid_search, referral, social, direct. Group by signup_source and aggregate by week for cohort stability. Include conversion rate to activation (first meaningful action within 7 days) as a secondary metric. The raw count metric alone is misleading because paid channels over-index for low-activation users. Analytics team confirms the dashboard at analytics.example.com/signups is the source of truth.

Section 4 — Test suite conventions (engineering docs):
The main test suite runs via `pytest` from the repo root. CI gates on unit-tests passing plus integration-tests passing plus type-check clean plus linter clean. Deployment to staging is manual via GitHub Actions workflow dispatch; deployment to production requires two approvals plus a signed artifact. Rollback is `git revert` on the deploy commit followed by re-running the deploy workflow.

Section 5 — Internal market research (strategy team Q4 2026):
AI framework market in 2026 is dominated by LangChain (149K stars, enterprise SaaS tier launched Q3), LlamaIndex (36K stars, RAG-focused), and Semantic Kernel (Microsoft, 25K stars). Vector databases: Pinecone and Weaviate lead paid tier; Qdrant and Chroma lead OSS adoption. Model serving: vLLM is the de-facto open-source standard for throughput-optimized serving; TGI from Hugging Face is competitive for smaller teams. The economics layer is undefined — no framework ships a production cost dashboard.

Section 6 — Application configuration schema (config/app.yaml reference):
The debug flag enables verbose logging, exposes /debug/* routes, and disables rate-limiting on internal endpoints. It must be false in production. The database_url field uses postgres:// scheme; postgresql:// is also accepted. redis_url is required when cache_enabled is true. The feature_flags map accepts boolean values keyed by feature name; unknown flags are logged as warnings but do not block startup.

Section 7 — SQL query patterns (data engineering runbook):
For high-value order queries, prefer WHERE total > 1000 AND created_at > NOW() - INTERVAL '30 days'. Index on (created_at DESC, total) is required for performance; without it the query becomes a sequential scan. CSV export via COPY TO STDOUT WITH CSV HEADER is fastest; psql \copy is the client-side equivalent for non-superusers. Always use parameterized queries to prevent SQL injection.

Section 8 — RAG evaluation metrics (research team briefing):
The three primary metrics for RAG systems: (1) context recall — did the retriever find the relevant chunks, (2) faithfulness — are the generated answers supported by the retrieved context, (3) answer relevance — does the answer address the user's question. Human evaluation remains the gold standard; LLM-as-judge is a cost-effective proxy. ROUGE-L alone is insufficient for RAG because it cannot distinguish factual from hallucinated content.
"""

TOOL_SCENARIOS = [
    "What are the main improvements in PostgreSQL 16's query planner, and how do they affect ORDER BY queries?",
    "Summarize OIDC refresh token rotation and explain why localStorage is a poor place to store refresh tokens.",
    "Why is a raw signup count misleading, and what secondary metric does the analytics team recommend?",
    "Walk me through the gates that CI enforces before a deploy can reach production.",
    "Who are the top three AI frameworks in 2026, and what is the undefined layer in the market?",
    "List the three effects of setting debug=true in app.yaml, and why must it be false in production?",
    "What index is required for the high-value order query, and what happens without it?",
    "Give me the three primary RAG evaluation metrics and explain why ROUGE-L is insufficient on its own.",
    "Compare OIDC refresh token guidance to the deploy workflow — both talk about rotation and approvals, but for different reasons. Explain.",
    "A user asks why their PostgreSQL 16 logical replication from a standby isn't working; which section is most relevant, and what should they check first?",
]


def workload_tool_call(n=10):
    """Synthetic RAG long-context Q&A.

    Compress the SYSTEM prompt (retrieved context). User turn stays verbatim.
    This mirrors the actual production shape: agents send a large retrieved-context
    system + small dynamic user. Compression on static context is the dominant win
    and does not change downstream behavior.
    """
    out = []
    print(f"\n=== Workload B: RAG LONG-CONTEXT QA (synthetic, n={n}) ===")
    compressed_sys = extractive_compress(RAG_CONTEXT)
    print(f"System prompt: {len(RAG_CONTEXT)}ch → compressed {len(compressed_sys)}ch")
    scenarios = TOOL_SCENARIOS[:n]
    for i, user in enumerate(scenarios):
        print(f"\n--- B{i+1}/{len(scenarios)} user='{user[:60]}...' ---")
        for mode, sys in [("baseline", RAG_CONTEXT), ("compressor", compressed_sys)]:
            res = groq_call(sys, user)
            row = {
                "workload": "rag_qa",
                "scenario_idx": i,
                "mode": mode,
                "system_len_chars": len(sys),
                "sha256_system": hashlib.sha256(sys.encode()).hexdigest(),
                **res,
            }
            out.append(row)
            if res["ok"]:
                print(f"  {mode:10s} OK  tokens_in={res['tokens_in']} wall={res['wall']:.2f}s")
            else:
                print(f"  {mode:10s} ERR {res.get('err','')[:80]}")
            time.sleep(8)  # respect TPM/RPM burst limits
    return out


def main():
    rows = []
    rows.extend(workload_dialogue(n=10))
    rows.extend(workload_tool_call(n=10))
    out_path = OUT_DIR / "rows.json"
    out_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\n✓ Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
