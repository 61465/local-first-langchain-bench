"""R5 staged — one contender per stage, fast sequential runs.

Stage 1: baseline only, n=10 samples × 2 trials = 20 runs
Stage 2: nexus-v09 only, n=10 × 2 = 20 runs
Stage 3: nexus-v09-compressor-v2 only, n=10 × 2 = 20 runs

Each stage completes independently, writes its own results.
If any stage fails, previous stages remain usable.
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

from runner_r3 import RowResult, sha256_text, execute_interleaved, summarize  # noqa
from runner_r3 import GROQ_LLAMA33_INPUT_USD_PER_M, GROQ_LLAMA33_OUTPUT_USD_PER_M  # noqa
from runner_r5 import contender_nexus_v09_compressor_v2  # noqa
from runner_r3 import contender_baseline, contender_nexus_v09  # noqa
from workload_summarization import load_cnn_dailymail  # noqa
from runner import _load_env  # noqa
_load_env()

STAGE_CONTENDERS = {
    "stage1-baseline": {"baseline-langchain": contender_baseline},
    "stage2-nexus": {"nexus-v09": contender_nexus_v09},
    "stage3-compressor-v2": {"nexus-v09-compressor-v2": contender_nexus_v09_compressor_v2},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGE_CONTENDERS.keys()))
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--inter-call-sleep", type=float, default=2.0)
    ap.add_argument("--out", type=pathlib.Path, default=HERE.parent / "results" / "r5_staged")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    contenders_dict = STAGE_CONTENDERS[args.stage]
    print(f"Stage: {args.stage}")
    print(f"Contenders: {list(contenders_dict.keys())}")

    dataset = load_cnn_dailymail(n=args.samples)
    print(f"Loaded {len(dataset)} samples")

    # Patch runner_r3 CONTENDERS to just our stage's entry
    import runner_r3
    runner_r3.CONTENDERS = contenders_dict

    rows = execute_interleaved(list(contenders_dict.keys()), dataset, args.trials, args.out,
                               inter_call_sleep=args.inter_call_sleep)
    (args.out / f"{args.stage}_rows.json").write_text(json.dumps([asdict(r) for r in rows], indent=2))

    summary = summarize(rows)
    (args.out / f"{args.stage}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== {args.stage} SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
