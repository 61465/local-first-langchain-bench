"""
R5 runner — v2 compressor (adaptive + dedup + high-info preservation).

Changes from R3:
- Uses compressor_v2.extractive_compress_v2 (adaptive aggressive mode)
- Increased inter-call sleep to 2.5s (R4 showed rate-storm at 1.5s)
- Longer samples: 20 articles × 2 trials × 3 contenders = 120 runs
- Same interleaved execution to avoid time-of-day bias
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from runner_r3 import (  # noqa: E402
    RowResult, _load_env, groq_call, _verify_response, sha256_text,
    GROQ_LLAMA33_INPUT_USD_PER_M, GROQ_LLAMA33_OUTPUT_USD_PER_M,
    contender_baseline, contender_nexus_v09,
    execute_interleaved, summarize, _run_once,
)
from workload_summarization import (  # noqa: E402
    load_cnn_dailymail, rouge_l_f1,
    SUMMARIZATION_SYSTEM, SUMMARIZATION_USER_TEMPLATE,
)
from compressor_v2 import extractive_compress_v2  # noqa: E402

_load_env()


def _build_prompt(article: str) -> str:
    return SUMMARIZATION_USER_TEMPLATE.format(article=article)


def contender_nexus_v09_compressor_v2(article: str):
    """v2 compressor + verified-fast."""
    compressed = extractive_compress_v2(article)
    prompt = _build_prompt(compressed)
    c, tin, tout, r = _run_once(prompt, verify=True, source_for_verify=compressed)
    return c, tin, tout, r, compressed


CONTENDERS_V2 = {
    "baseline-langchain": contender_baseline,
    "nexus-v09": contender_nexus_v09,
    "nexus-v09-compressor-v2": contender_nexus_v09_compressor_v2,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--inter-call-sleep", type=float, default=2.5)
    ap.add_argument("--out", type=pathlib.Path, default=HERE.parent / "results" / "r5")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.samples} samples × {args.trials} trials × {len(CONTENDERS_V2)} contenders "
          f"= {args.samples * args.trials * len(CONTENDERS_V2)} total measurements")
    dataset = load_cnn_dailymail(n=args.samples)
    print(f"Loaded {len(dataset)} samples.")

    # Show v2 compression on first 5
    print("\n=== v2 Compressor preview ===")
    for i, d in enumerate(dataset[:5]):
        orig = d["article"]
        comp = extractive_compress_v2(orig)
        reduction = 100 * (1 - len(comp) / len(orig))
        print(f"  sample {i}: {len(orig)} -> {len(comp)} chars ({reduction:.1f}% reduction)")
    print()

    # Patch the contender dict in runner_r3
    import runner_r3
    runner_r3.CONTENDERS = CONTENDERS_V2

    rows = execute_interleaved(list(CONTENDERS_V2.keys()), dataset, args.trials, args.out,
                               inter_call_sleep=args.inter_call_sleep)
    (args.out / "rows.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))

    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== R5 SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
