# local-first-langchain-bench

**Measurement-first foundation for a LangChain Production Economics Suite.**
Rigorous, reproducible benchmark + two novel compressor components measured on 164 paired API samples.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: CC0](https://img.shields.io/badge/Data-CC0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

---

## Headline numbers (R11 n=86 distinct CNN/DailyMail test-split articles, gpt-oss-20b via Groq)

| Metric | v2a extractive (prior) | **HSTC hybrid (new)** |
|---|---|---|
| Median tokens_in reduction | 36.2% | **49.2%** |
| 95% CI | [32.2%, 37.5%] | **[46.4%, 50.8%]** |
| Paired Wilcoxon p (vs v2a) | — | **7.96 × 10⁻¹⁶** |
| Cohen's d | — | **-2.53 (very large)** |
| Direction stability | — | **86/86 (100%)** |
| vs LangChain truncation baseline | — | **51.5% vs truncation's 18.1%** (same corpus) |
| p99 latency at 4 kB input | 15.88 ms | **13.87 ms** |
| p99 latency at 16 kB input | 60.89 ms | **23.19 ms (2.6× better)** |
| Unicode / RTL / emoji attack suite | — | **50/50 strings, 5/5 categories pass** |
| Lines of production code | 150 | **310 total (150 + 160)** |
| Runtime deps | Python stdlib | **Python stdlib only** |

## RFC

The full RFC proposal with experimental setup, statistical methodology,
TCO analysis with SEC-anchored scale, and 10 documented limitations:
→ [`docs/RFC_DRAFT.md`](docs/RFC_DRAFT.md) (v5.5, 2026-11-05, ~7,000 words)

## Companion documents

- [`docs/TCO_CALCULATOR.md`](docs/TCO_CALCULATOR.md) — CFO-facing total-cost worksheet
- [`docs/TCO_ANCHORS.md`](docs/TCO_ANCHORS.md) — SEC-filing citations (Klarna / Shopify / Microsoft)

## Reproducibility

```bash
git clone https://github.com/61465/local-first-langchain-bench
cd local-first-langchain-bench
pip install -r requirements.txt

# Unit tests
python src/test_compressor_v2.py        # 12/12 pass
python src/test_hstc_unicode.py         # 5/5 pass, 50/50 attack strings

# Offline — tiktoken-only compression ratio
python src/token_measurement.py

# Full live benchmark (needs GROQ_API_KEY)
python src/runner_r3.py                 # R3: 3 contenders × 15 samples × 3 trials
python src/quality_metrics.py           # BERTScore + Wilcoxon on R3
python src/stats_v2.py                  # 95% CI + Cohen's d + power analysis
python src/r6_multimodel.py             # Cross-model-family (gpt-oss-120b/20b/qwen)
python src/r7_multiworkload.py          # Dialogue + RAG-QA workloads
python src/r8_n100_expand.py            # n=100 v2a expansion (power-upgrade)
python src/r10_hstc.py                  # HSTC vs v2a vs baseline (n=11, R3 fair subset)
python src/r11_hstc_n100.py             # HSTC n=86 distinct docs (R11)
python src/bench_latency.py             # p50/p95/p99 at 1k/4k/16k
python src/bench_truncation.py          # LangChain truncation baseline
python src/hstc_ablation.py             # Stopword-list sensitivity
python src/stopwords_derivation.py      # Corpus-derived vs hand-curated overlap
```

## Compressor usage (zero-LLM)

```python
from compressor_hstc import hstc_compress

article = "Your long RAG context or document..."
compressed = hstc_compress(article)
# Returns deterministically compressed string, ~49% fewer tokens, zero LLM calls
```

## Headline reproduction

Every reported number maps to a file:line or a published paper. Row-level JSON with SHA256
anchors is committed in `results/{r3,r6,r7,r8,r10,r11}/rows.json`.

## License

- Code: MIT ([`LICENSE`](LICENSE))
- Data: CC0 (we distribute no original text; see dataset citations in RFC §4)

## Status

v5.5 — public RFC for LangChain community review. Published 2026-11-05.

## Author

Abdalrahman Mohamed · [@61465](https://github.com/61465) · abdarahman10555@gmail.com
