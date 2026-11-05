### Submission checklist

- [x] This is a feature request, not a bug report or usage question.
- [x] I added a clear and descriptive title that summarizes the feature request.
- [x] I used the GitHub search to find a similar feature request and didn't find it.
- [x] I checked the LangChain documentation and API reference to see if this feature already exists.
- [x] This is not related to the langchain-community package.

### Package

`langchain-core` (or community via `langchain.compress.HybridCompressor` wrapper).

### Feature description

Propose adding **HSTC (Hybrid Sentence+Token Compression)** as a middleware-slot context compressor — zero-LLM, deterministic, drop-in Runnable. Measured on n=86 distinct CNN/DailyMail articles with full statistical reporting (paired Wilcoxon, exact permutation test, 95% bootstrap CIs, Cohen's d, power analysis, 50-string Unicode attack suite).

**Headline numbers** (R11 n=86 via Groq gpt-oss-20b):

- HSTC median tokens_in reduction: **49.2%** (95% CI [46.4%, 50.8%])
- vs naive first-N-tokens truncation on same corpus: **51.5% vs 18.1%**
- Paired Wilcoxon p=7.96×10⁻¹⁶; Cohen's d=-2.53; direction-stable on **86/86 samples**
- Quality preserved (baseline-vs-HSTC ROUGE-L Wilcoxon p=0.97, Cohen's d=-0.064 negligible)
- Observed output-token drift (R10 n=11 paired): median +0.5% (essentially unchanged)
- p99 latency at 16 kB: **23 ms** (stdlib `re`, no LLM, no learned weights)
- 74-term stopword list — 81% overlap with corpus-derived top-50 (not arbitrary)
- All 50 Unicode/RTL/emoji attack strings handled without crash

**LangChain integration** already implemented (`src/langchain_integration.py` in the repo):

```python
from langchain_integration import HybridCompressor
compressor = HybridCompressor.for_rag().as_runnable()  # returns RunnableLambda
chain = compressor | prompt | llm  # standard LangChain composition
```

Factories: `for_rag()`, `for_summarization()`, `for_dialogue()`. `invoke`/`ainvoke`/`batch` verified.

### Motivation

Agent-chain economics: the three questions every CFO asks before scaling — *"what will this cost at 10M req/day? what's the p99 under rate-pressure? how much quality do I lose if I cut 50% of tokens?"* — are currently unanswered by any component in `langchain-core`. Truncation drops article tails unconditionally. LLMLingua needs a 560M-param scoring model. HSTC occupies the zero-LLM deterministic drop-in slot.

**TCO scale** (input-only reduction, output tokens unchanged; measured output drift honestly disclosed in RFC §6):

- Scenario A (gpt-4o-mini @ 1M req/day): ~$18.6K/year HSTC median
- Scenario B (Sonnet 4.6 @ 5M req/day): ~$1.85M/year
- Scenario C (gpt-4o @ 10M req/day): ~$3.10M/year
- Scenario D (Opus 4.6 @ 10M req/day single-chain): ~$6.18M/year

Scale anchored by Klarna 20-F ($59M saved), Shopify 10-Q (+$40M LLM-infra line), Microsoft FY26 10-K ($115.95B capex).

### Prior art — honest framing

HSTC is **not** the first hybrid compressor overall. PromptIntern (Zou et al. EMNLP 2024, arXiv:2407.02211) is a two-stage method during fine-tuning. HSTC's narrow positioning:

- **Zero-LLM** at compression time (unlike LLMLingua-2)
- **Deterministic** (same input → same output; no stochastic weights)
- **Drop-in-at-inference** for the LangChain middleware slot
- **Single-provider measurement** honestly disclosed as a limitation

### Links

- **Full RFC** (~630 lines, 14 sections): https://github.com/61465/local-first-langchain-bench/blob/main/docs/RFC_DRAFT.md
- **Repo** (MIT + CC0): https://github.com/61465/local-first-langchain-bench
- **Compressor**: [`src/compressor_hstc.py`](https://github.com/61465/local-first-langchain-bench/blob/main/src/compressor_hstc.py) (stdlib only, 160 LOC)
- **LangChain Runnable wrapper**: [`src/langchain_integration.py`](https://github.com/61465/local-first-langchain-bench/blob/main/src/langchain_integration.py)

### System Info

Not applicable — this is a feature proposal, not a bug report. Measurement environment documented in RFC §5.7 SBOM (AMD64 / Windows 11 / Python 3.12.10).

---

### Not the ceiling — room to grow if accepted

The 49.2% number is the **starting point**, not a limit. If maintainers are interested in the direction, several measured levers can push the number higher:

- **Multi-provider validation** (Claude/GPT-4/Gemini direct) — current numbers are single-provider (Groq); re-measurement on Anthropic/OpenAI direct is a straightforward follow-up.
- **Workload-tuned stopword sets** — `HybridCompressor.for_rag()/for_dialogue()/for_summarization()` already exist; tuned stopword lists per domain (legal, medical, code) could push reduction into the 55-60% band on domain-matched workloads.
- **HSTC × static-context injection (Component D)** — PromptIntern's 90%+ reduction on recurrent prompts is complementary to HSTC's dynamic-content reduction. The two levers haven't been combined on a single workload in published literature; we have the measurement infrastructure to run that experiment.
- **Phase 2 LangSmith integration (Component B)** — plug the benchmark harness into LangSmith traces for per-chain `$/req`, p99, quality retention dashboards.
- **n≥500 power-upgrade** — current R11 n=86 (limited by free-tier rate limits); paid-tier re-run would narrow CIs further.

Happy to prioritize whichever direction the maintainers see most value in. The RFC's §10 roadmap lays out five components; Component A (HSTC itself) is the first and most mature. Everything else is runway if the direction is right.
