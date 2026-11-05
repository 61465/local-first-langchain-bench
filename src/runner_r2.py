"""
R2 runner — summarization workload, n=30, long-context stress.

Same 3 contenders as R1 but:
- Workload: CNN/DailyMail summarization (not short QA)
- Prompts: 400-1400 tokens per article (vs ~15 in R1)
- Quality metric: ROUGE-L F1 vs reference summary (not exact match)
- Samples: n=30 (vs 10 in R1)
- max_tokens_out: 512 (room for 3-sentence summary)

Zero-fabrication policy: every number is a real measurement. Compressor
is the SAME static-regex `_compress()` from R1 — if it doesn't help on
long articles (likely — it strips polite filler that news articles
rarely contain), the honest result is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import time
from dataclasses import asdict, dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from runner import _load_env, groq_call, _verify_response, _compress, sha256_text, GROQ_LLAMA33_INPUT_USD_PER_M, GROQ_LLAMA33_OUTPUT_USD_PER_M  # noqa
from workload_summarization import (  # noqa
    load_cnn_dailymail,
    rouge_l_f1,
    SUMMARIZATION_SYSTEM,
    SUMMARIZATION_USER_TEMPLATE,
)

_load_env()


@dataclass
class SumResult:
    contender: str
    sample_id: str
    article_len_chars: int
    tokens_in: int
    tokens_out: int
    tokens_billed_usd: float
    wall_clock_seconds: float
    retry_count: int
    rouge_l_f1: float
    summary: str
    reference: str
    sha256_input: str
    sha256_output: str
    extra: dict = field(default_factory=dict)


def _build_prompt(article: str) -> str:
    return SUMMARIZATION_USER_TEMPLATE.format(article=article)


def run_baseline_sum(article: str):
    prompt = _build_prompt(article)
    content, tin, tout = groq_call(prompt, system=SUMMARIZATION_SYSTEM)
    return content, tin, tout, 0, prompt


def run_nexus_v09_sum(article: str):
    prompt = _build_prompt(article)
    content, tin, tout = groq_call(prompt, system=SUMMARIZATION_SYSTEM)
    ok, _ = _verify_response(content, article)
    retries = 0
    if not ok:
        retries = 1
        retry_sys = SUMMARIZATION_SYSTEM + (
            "\n\nRETRY WITH STRONGER GROUNDING: output only the 3-sentence "
            "summary. Do not refuse. Do not add preamble."
        )
        content2, tin2, tout2 = groq_call(prompt, system=retry_sys)
        content, tin, tout = content2, tin + tin2, tout + tout2
    return content, tin, tout, retries, prompt


def run_nexus_v09_compressor_sum(article: str):
    """Compressor applied to the article text itself before prompt build."""
    compressed_article = _compress(article)
    prompt = _build_prompt(compressed_article)
    content, tin, tout = groq_call(prompt, system=SUMMARIZATION_SYSTEM)
    ok, _ = _verify_response(content, compressed_article)
    retries = 0
    if not ok:
        retries = 1
        retry_sys = SUMMARIZATION_SYSTEM + (
            "\n\nRETRY WITH STRONGER GROUNDING: output only the 3-sentence "
            "summary. Do not refuse. Do not add preamble."
        )
        content2, tin2, tout2 = groq_call(prompt, system=retry_sys)
        content, tin, tout = content2, tin + tin2, tout + tout2
    return content, tin, tout, retries, prompt


CONTENDERS = {
    "baseline-langchain": run_baseline_sum,
    "nexus-v09": run_nexus_v09_sum,
    "nexus-v09-compressor": run_nexus_v09_compressor_sum,
}


def execute(contender: str, dataset: list[dict], out_dir: pathlib.Path) -> list[SumResult]:
    fn = CONTENDERS[contender]
    results: list[SumResult] = []
    for i, item in enumerate(dataset):
        article = item["article"]
        reference = item["reference_summary"]
        sample_id = str(item["id"])
        sha_in = sha256_text(article)

        t0 = time.monotonic()
        try:
            summary, tin, tout, retries, _prompt = fn(article)
            err = None
        except Exception as e:
            summary, tin, tout, retries = "", 0, 0, 0
            err = f"{type(e).__name__}: {str(e)[:150]}"
        dur = time.monotonic() - t0
        billed = (tin * GROQ_LLAMA33_INPUT_USD_PER_M + tout * GROQ_LLAMA33_OUTPUT_USD_PER_M) / 1_000_000

        r_score = rouge_l_f1(summary, reference) if summary else 0.0

        result = SumResult(
            contender=contender,
            sample_id=sample_id,
            article_len_chars=len(article),
            tokens_in=tin,
            tokens_out=tout,
            tokens_billed_usd=round(billed, 6),
            wall_clock_seconds=round(dur, 3),
            retry_count=retries,
            rouge_l_f1=round(r_score, 4),
            summary=summary,
            reference=reference,
            sha256_input=sha_in,
            sha256_output=sha256_text(summary),
            extra={"error": err} if err else {},
        )
        results.append(result)
        line = (f"  [{contender}] sample {i+1:02d}/{len(dataset)} "
                f"({dur:.2f}s, tin={tin}, tout={tout}, rougeL={r_score:.3f})")
        if err:
            line += f" ERR:{err}"
        print(line.encode("ascii", "replace").decode("ascii"))

    (out_dir / f"{contender}_results.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2)
    )
    return results


def summarize(all_results: dict[str, list[SumResult]]) -> dict:
    out = {}
    for c, rs in all_results.items():
        n = len(rs)
        valid = [r for r in rs if not r.extra.get("error")]
        nv = len(valid)
        if nv == 0:
            out[c] = {"n": n, "valid": 0, "error": "all samples failed"}
            continue
        rouges = [r.rouge_l_f1 for r in valid]
        walls = [r.wall_clock_seconds for r in valid]
        tins = [r.tokens_in for r in valid]
        touts = [r.tokens_out for r in valid]
        usds = [r.tokens_billed_usd for r in valid]
        import statistics as st
        out[c] = {
            "n": n, "valid": nv,
            "rouge_l_mean": round(st.mean(rouges), 4),
            "rouge_l_stdev": round(st.stdev(rouges) if nv > 1 else 0.0, 4),
            "wall_mean_s": round(st.mean(walls), 3),
            "wall_stdev_s": round(st.stdev(walls) if nv > 1 else 0.0, 3),
            "wall_p50_s": round(st.median(walls), 3),
            "wall_p95_s": round(sorted(walls)[max(0, int(0.95 * nv) - 1)], 3),
            "total_tokens_in": sum(tins),
            "total_tokens_out": sum(touts),
            "total_usd": round(sum(usds), 6),
            "avg_tokens_in": round(st.mean(tins), 1),
            "avg_tokens_out": round(st.mean(touts), 1),
            "total_retries": sum(r.retry_count for r in valid),
            "retry_rate": round(sum(r.retry_count for r in valid) / nv, 3),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--contender", choices=list(CONTENDERS.keys()) + ["ALL"], default="ALL")
    ap.add_argument("--out", type=pathlib.Path, default=HERE.parent / "results" / "r2")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.samples} CNN/DailyMail samples...")
    dataset = load_cnn_dailymail(n=args.samples)
    print(f"Loaded {len(dataset)} samples.")

    contenders = list(CONTENDERS.keys()) if args.contender == "ALL" else [args.contender]
    all_results: dict[str, list[SumResult]] = {}

    for c in contenders:
        print(f"\n=== {c} ===")
        all_results[c] = execute(c, dataset, args.out)

    summary = summarize(all_results)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== R2 SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
