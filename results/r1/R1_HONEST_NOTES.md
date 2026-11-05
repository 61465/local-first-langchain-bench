# R1 — Honest Notes (2026-11-04)

## What ran
- 10 QA samples × 3 contenders × 1 stress (W0 — no stress applied)
- Model: `openai/gpt-oss-120b` on Groq
- Temperature 0.0, max_tokens 256, n=1 per sample

## Results (after normalization fix)

| Contender | Exact | Partial | Tin | Tout | Wall avg | Retries |
|---|---|---|---|---|---|---|
| baseline-langchain | 1.0 | 1.0 | 1080 | 729 | 1.953s | 0 |
| nexus-v09 | 0.9 | 1.0 | 1176 | 726 | 1.550s | 1 |
| nexus-v09-compressor | 0.9 | 1.0 | 1176 | 730 | 3.511s | 1 |

## What this does NOT show

- Statistical significance (n=10 too small)
- Impact at scale (short-QA ≠ production agent workload)
- Compressor benefit (QA is ~10 tokens; nothing to compress)
- Cost reduction (all 3 cost same $0.0012 total)

## What this DOES show

- NEXUS v0.9 verified-fast is ~21% faster on short QA at single-trial
  (1.55s vs 1.95s average)
- nexus-v09 exact-accuracy was 0.9 vs baseline 1.0 — the retry sometimes
  returns a wordier answer that fails exact-match (but hits partial)
- Compressor adds overhead with no benefit when input is already short

## Negatives recorded (per Gate 1 no-fab rule)

1. Compressor regressed wall-clock from 1.55s → 3.51s on short QA
2. nexus-v09 lost 1 exact-match (q08 "The smallest prime number is **2**. It i...")
   because retry produced wordier output
3. Token-in count higher on nexus because retry adds a second prompt send
4. n=10 is too small; 30+ needed for stat-sig claims

## R2 target

- Multi-paragraph QA (long context, 2k-16k tokens — where compressor
  could help)
- n=30+ per contender for standard-error bounds
- Separate "accuracy" from "verbosity" in scoring

## Artifacts

- `baseline-langchain_results.json` — SHA256 per sample
- `nexus-v09_results.json` — SHA256 per sample
- `nexus-v09-compressor_results.json` — SHA256 per sample
- `summary.json` — aggregated

Every row carries `sha256_input` + `sha256_output` for audit replay.
