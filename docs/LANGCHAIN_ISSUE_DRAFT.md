# [RFC/Feature request] HSTC — zero-LLM deterministic hybrid compressor (49.2% token reduction, measured)

## Summary

I built a hybrid sentence+token-level compressor for LangChain's middleware slot and measured it rigorously on CNN/DailyMail + dialogue + RAG-QA workloads. Results reproducible via the published repo.

## Headline (R11 n=86 distinct CNN/DailyMail articles, gpt-oss-20b via Groq)

- **HSTC median tokens_in reduction: 49.2%** (95% CI [46.4%, 50.8%])
- vs LangChain's naive first-N-tokens truncation on same corpus: 51.5% vs 18.1%
- Paired Wilcoxon p=7.96×10⁻¹⁶; Cohen's d=-2.53; direction-stable on **86/86 samples**
- Quality preserved (baseline-vs-HSTC ROUGE-L Wilcoxon p=0.97, Cohen's d=-0.064 negligible)
- p99 latency at 16kB: 23ms (stdlib `re`, no LLM, no learned weights)
- 74-term stopword list, 81% overlap with corpus-derived top-50 (not arbitrary)
- All 50 Unicode/RTL/emoji attack strings handled without crash

## Novelty

- **NOT** first hybrid compressor overall. PromptIntern (Zou et al. EMNLP 2024, arXiv:2407.02211) is a two-stage method **during fine-tuning**.
- HSTC's narrow positioning: **zero-LLM, deterministic, drop-in-at-inference** hybrid for the LangChain middleware slot. Different design space.

## Why it's worth your look

- Component A of a proposed "LangChain Production Economics Suite" (3 components + 1 research direction)
- SEC-anchored TCO: Scenario D (Opus-4.6 @ 10M req/day single-chain) = **$6.28M/year** saved
- Full RFC with 10 documented limitations, exact permutation test on R10 n=11, bootstrap CI on every claim

## Links

- **Full RFC** (599 lines, ~7,000 words, 14 sections): https://github.com/61465/local-first-langchain-bench/blob/main/docs/RFC_DRAFT.md
- **Repo (MIT + CC0)**: https://github.com/61465/local-first-langchain-bench
- **Compressor**: [`src/compressor_hstc.py`](https://github.com/61465/local-first-langchain-bench/blob/main/src/compressor_hstc.py) (160 LOC, Python stdlib only)

## Ask

Is HSTC a direction LangChain is interested in? If yes, happy to open a PR adding `langchain.compress.HybridCompressor` to `langchain-community` or as a drop-in wrapper.

If not — the repo and RFC are MIT + CC0; take what's useful.

---

Author: Abdalrahman Mohamed (@61465)
