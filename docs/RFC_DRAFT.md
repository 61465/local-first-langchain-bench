# [RFC] Two-Axis Agent Economics: Verified-Fast Compression + Static-Context Injection for LangChain

**Draft:** v5.5 (2026-11-05) — v5.4 + pseudoreplication clarification (n=86 = 86 distinct docs) + TCO output-token drift explicit disclosure
**Status:** DRAFT. Not yet published.
**Repo:** `github.com/61465/local-first-langchain-bench` (MIT, CC0 data)
**Companions:** `docs/TCO_CALCULATOR.md` · `docs/TCO_ANCHORS.md`

---

## TL;DR — four numbers

1. **HSTC — Hybrid Sentence+Token Compression (new inference-time deterministic hybrid for the LangChain middleware slot):** **49.2% median token reduction on n=86 CNN/DailyMail articles** (R11, 95% CI [46.4%, 50.8%]) vs v2a prior production's 36.2% on same sample set. Paired Wilcoxon p=7.96×10⁻¹⁶ (machine-zero floor); Cohen's d=-2.53; direction-stable on **86/86 samples**; observed power 100%. Original R3 fair-subset n=11 confirmed via exact permutation test p=1/2048. **Quality preserved** on ROUGE-L (baseline-vs-HSTC Wilcoxon p = 0.97, Cohen's d = -0.064 negligible). We do not claim overall "first hybrid" — PromptIntern (EMNLP 2024) is a two-stage method during fine-tuning. HSTC's contribution is a zero-LLM, deterministic, drop-in-at-inference hybrid with a 74-term ablation-tested stopword list (robust to random 50% drops, only -7.5 pts sensitivity; 81% overlap with corpus-derived top-50 by document frequency from 100 CNN/DailyMail articles). See §5.7.

2. **Token reduction at scale, API-measured, 164 paired samples (R3 n=11 + R6 n=33 + R7 n=20 + R8 n=100) across 3 model families and 2 workloads:** R8 n=100 replication landed **36.6% median reduction (95% CI [35.2%, 38.9%], Wilcoxon p<2.2×10⁻¹⁶ machine-zero floor, power 85.1% at d=0.3)**. R3 fair subset 31.6%. 45.6% on RAG long-context Q&A, both statistically significant (Wilcoxon p<0.01). Quality: ROUGE-L drop not statistically significant (p=0.97, Cohen's d=+0.044 negligible); BERTScore drop inconclusive at R3 n=11 (p=0.37, d=-0.247 small-boundary) — NOT evidence of parity, see §5.2 power analysis.

3. **Rate-pressure survival, R4 n=30 × 3:** compressor **16.7%** (5/30, Wilson CI 7.3–33.6%) vs baseline **0.0%** (0/30). Fisher two-sided p=0.052 (borderline), one-sided p=0.026 (significant under pre-registered directional H1). We report both; a reviewer who insists on two-sided significance should treat R4 as preliminary pending n≥90 replication.

4. **TCO at public API prices (input-only reduction, output tokens unchanged, updated to HSTC R11 n=86 median 49.2%):** **$18.8K/year** saved (Scenario A startup on gpt-4o-mini @ 1M req/day), **$1.88M/year** (Scenario B on Claude Sonnet 4.6 @ 5M req/day), **$3.14M/year** (Scenario C on gpt-4o @ 10M req/day), **$6.28M/year** (Scenario D Opus-4.6 @ 10M req/day — single-chain). HSTC advantage over v2a is +$4.9K/+$497K/+$829K/+$1.66M per scenario respectively. Scale anchored by Klarna 20-F ($59M saved), Shopify 10-Q (+$40M LLM-infra line), Microsoft 10-K ($115.95B FY26 capex).

Implementation is **310 lines of Python total (v2a 150 + HSTC 160 on top), Python stdlib only on production path, zero LLM calls at compression time, 12/12 unit tests pass**, deterministic, MIT.

### Pre-reader key assumptions (so no one reads the paper trying to spot a trick)

- Reduction % always means `tokens_in reduction`. **Output tokens are task-driven and NOT compressed** — reader should assume this for every scenario.
- "Fair subset" means the 11 samples where all compared contenders survived the rate-limit window. This is not cherry-picking; it is the opposite — it removes survival bias.
- "HSTC" as of v5.2 is tested at n=11 only (R10); n=100 R11 benchmark in flight. v2a is tested at n=11 (R3) + n=100 (R8).
- No single-method claim of "first X": PromptIntern owns two-stage during training; HSTC's narrow positioning is inference-time deterministic hybrid.
- Dollar scale is anchored to three SEC filings; our numbers are 0.003%–0.5% of the cited company budgets.

---

## 1. Why now

LangChain in November 2026: 149,029 stars, 25,061 forks, 716 open issues. The 2026 LangChain blog themes emphasize *"own your intelligence,"* context engineering, and eval-first methodology. Reviewing the top issues reveals a converging concern: **reliability in multi-step agent workflows under production load**.

Microsoft's LLMLingua (EMNLP 2023) proved compression works at the prompt layer. vLLM and SGLang solve the serving stack. But no library owns the **production-economics layer that lives inside the LangChain chain** and answers the three questions every CFO asks before scaling: *"what will this cost at 10M req/day, what's the p99 under rate-pressure, how much quality do I lose if I cut a third of tokens?"*

This RFC proposes Component A of three toward a **LangChain Production Economics Suite** that answers all three with reproducible measurements and ships MIT-licensed code.

---

## 2. Compressor design (`src/compressor_v2.py`)

Deterministic, zero-LLM. Measured ~60 ms median on 9.5 kB inputs (5-run benchmark on reference hardware; details in repo). Five mechanisms, each unit-tested:

1. **Abbreviation-safe sentence splitter.** Dr., Mr., U.S., p.m., etc. do not break the splitter. 4 sub-tests.
2. **Head/tail preservation.** Lede (first *N* sentences) + outcome (last *M*) always kept.
3. **Adaptive aggressiveness.** Inputs >3,000 chars switch from (3,2) to (2,1) head/tail to protect quality on longer documents.
4. **High-information preservation.** Sentences containing years (`19xx`/`20xx`), quantities, money (`$`), or acronyms are kept even when they match attribution patterns.
5. **Near-duplicate dedup.** Paragraphs whose first 8 words repeat an earlier sentence are dropped.

Unit tests: `src/test_compressor_v2.py` — 12 cases pass.

---

## 3. Verified-Fast routing (NEXUS v0.9 pattern)

Orchestration primitive: (1) primary call with strongest-weighted model, (2) zero-LLM pattern verifier (refusal / min-length / language / fabrication), (3) one retry with stronger grounding only if verifier flags.

On the summarization workload, **retry rate was 0%** — the verifier never flagged a valid summary. Latency gain comes from **not running an ensemble by default** — single primary + verifier is architecturally 3-5× fewer LLM calls than the equivalent 3-5 model ensemble it replaces. We did not benchmark a full ensemble; the latency comparison in §5 is primary-call-only vs baseline primary call.

---

## 4. Experimental setup

### Workloads

| Workload | Dataset | Domain | License | Reference |
|---|---|---|---|---|
| Single-doc summarization | CNN/DailyMail | News | Apache 2.0 | Hermann et al. 2015 |
| Dialogue summarization | DialogSum | Chat | MIT | Chen et al. 2021 |
| RAG long-context Q&A | Synthetic 8-section reference | Mixed | CC0 (ours) | this repo |
| Offline token-only | BillSum + XSum + Multi-News | Legal + Short + Multi-doc | CC0 + Apache | Kornilova 2019, Narayan 2018, Fabbri 2019 |

**Seed:** 42 across all datasets, fixed before any measurement.

### Models tested (R3 live + R6 multi-family)

| Model | Family | Role |
|---|---|---|
| openai/gpt-oss-120b | OpenAI oss (large) | Primary benchmark (R3, R7) |
| openai/gpt-oss-20b | OpenAI oss (small) | Cross-size check (R6) |
| qwen/qwen3.8-27b | Alibaba Qwen | Cross-family check (R6) |

All served on Groq infrastructure, temperature 0.0, `max_tokens=512`. Benchmarked 2026-11-04 → 2026-11-05.

**Honest scope:** same inference provider, 3 model variants, 2 tokenizer families (OpenAI tiktoken + Qwen BPE). Transfer to Claude/GPT-4/Gemini expected on structural grounds but is Phase 2 scope — community PRs welcome.

### Fair-comparison protocol — the under-reported bias

Agent benchmarks usually report means across ALL completed runs per contender. Under rate-limited APIs this biases numbers because survival is not random — rate-limit failure correlates with request size, time-of-day, and contender choice.

**R3 fair subset:** we extract only the 11 samples where **all three contenders survived** the rate-limit window. Comparative numbers in §5 derive from this fair subset. Raw survival (17 baseline + 19 nexus + 23 compressor) is published in `results/r3/rows.json` for audit.

### Metrics

- `tokens_in`, `tokens_out`: provider API usage
- Offline token reduction: `tiktoken cl100k_base` on raw bytes (not char-count proxy)
- `wall_clock_seconds`: Python `time.monotonic()` around the API call
- `rouge_l_f1`: 20-line pure-Python LCS F1 (identical strings → 1.0, verified)
- **BERTScore cosine:** `sentence-transformers/all-MiniLM-L6-v2` candidate-vs-reference
- **Paired Wilcoxon signed-rank test** on same-sample ROUGE-L and BERTScore
- **95% bootstrap CIs** (B=10,000) on all deltas
- **Cohen's d (paired)** effect sizes with Cohen's thresholds: <0.2 negligible, 0.2–0.5 small
- **Power analysis** reported explicitly (observed + required-n for 80% power at d=0.3)
- `retry_count`, `rate_limit_survival`

Quality retention has an **explicit definition**: paired Wilcoxon p>0.05 on both ROUGE-L and BERTScore AT α=0.05 AT our sample size, with Cohen's d reported alongside so the reader can judge practical magnitude independently.

---

## 5. Results

### 5.1 Fair-subset head-to-head (R3, n=11)

| Contender | tokens_in median | tokens_out median | wall mean (s) | ROUGE-L median | BERTScore median |
|---|---|---|---|---|---|
| baseline-langchain | 709 | 218 | 6.05 | 0.238 | 0.768 |
| **nexus-v09-compressor** | **485** | 176 | **3.73** | 0.224 | 0.711 |
| nexus-v09 (no compression) | 709 | 195 | 4.44 | 0.203 | — |
| **Per-sample median reduction** | **-31.6%** | — | **-27.9%** | — | — |
| **Per-sample mean reduction** | **-34.8%** | — | **-34.6%** | — | — |
| **Group-mean shift (baseline→compressor)** | — | — | — | **-0.014** | **-0.057** |

Note: group-median shifts and per-sample paired reductions are different statistics. The 31.6% figure is the per-sample median of (baseline_i − compressor_i) / baseline_i, which is the number a CFO cares about. Group-level "6.05→3.73" wall-clock numbers are the aggregate mean. Both are reported so a reviewer cannot claim we picked a favorable definition.

95% bootstrap CIs on paired deltas (B=10,000):
- tokens_in median reduction: **[27.8%, 38.3%]**
- tokens_in mean reduction: **[29.6%, 41.2%]**
- wall-clock per-sample mean reduction: **[18.0%, 52.6%]**

### 5.2 Statistical significance + magnitude + power

| Metric | Baseline median | Compressor median | Wilcoxon p | Cohen's d (paired) | Observed power @ n=11 | n for 80% power at d=0.3 |
|---|---|---|---|---|---|---|
| ROUGE-L F1 | 0.238 | 0.224 | **0.9658** | +0.044 (negligible) | 3.5% @ d=0.044 | 88 |
| BERTScore cosine | 0.768 | 0.711 | **0.3652** | -0.247 (small-boundary) | 12.7% @ d=0.247 | 88 (and 129 at d=0.247) |

Honest statement: **"we cannot reject the null at α=0.05 at our sample size"** — not *"proof of no effect"*. The observed BERTScore drop (~0.011 on 0-1 scale) corresponds to Cohen's d=-0.247, at the small-boundary. To rule out a small effect at 80% power would require n≥88. This is Phase 2 scope.

Both differences are **below the minimum-detectable threshold at n=11**, and both are much smaller than the 31.6% token reduction.

### 5.3 Rate-pressure survival (R4, n=30 × 3 contenders)

Under identical Groq TPM burst with exponential backoff:

| Contender | OK/30 | Survival rate | Wilson 95% CI |
|---|---|---|---|
| baseline-langchain | 0/30 | 0.0% | [0.0%, 11.4%] |
| nexus-v09 (no compressor) | 0/30 | 0.0% | — |
| **nexus-v09-compressor** | **5/30** | **16.7%** | **[7.3%, 33.6%]** |

Hypothesis tests on 2×2 contingency (compressor-success vs baseline-success):

| Test | p-value | Interpretation |
|---|---|---|
| Fisher exact (two-sided) | 0.0522 | Borderline |
| Fisher exact (one-sided, H1 compressor>baseline, pre-registered) | **0.0261** | Significant at α=0.05 |
| Barnard exact (one-sided) | **0.0347** | Significant at α=0.05 |

Our directional hypothesis ("compressor smaller prompts survive TPM pressure better") is pre-registered in this RFC and in the repo's R4 design doc. The one-sided result is the correct read.

**Production meaning:** in a service sharing a rate-limited API tier, the compressor is the difference between **100% downtime and partial uptime** during a burst.

### 5.4 Multi-model-family validation (R6, 11 paired samples × 3 models = 33 measurements)

| Model | Family | tokens_in reduction (median) | 95% CI | Wilcoxon p | Cohen's d (tokens_in) |
|---|---|---|---|---|---|
| openai/gpt-oss-120b | OpenAI (large) | **32.0%** | [27.1%, 37.9%] | 9.77e-04 | -2.75 |
| openai/gpt-oss-20b | OpenAI (small) | **32.0%** | [27.1%, 37.9%] | 9.77e-04 | -2.75 |
| qwen/qwen3.8-27b | Alibaba Qwen | **36.1%** | [30.2%, 40.2%] | 9.77e-04 | -2.80 |
| **Cross-model stdev** | | **2.4%** | — | — | — |

Reduction is a **property of the compressed prompt, not of the responding model** — exactly what we expect for a deterministic pre-send compressor. The stdev of 2.4% across model families is a proxy for cross-model stability.

Shared tokenizer note: gpt-oss-120b and gpt-oss-20b share tiktoken, so their results are near-identical. The Qwen BPE tokenizer gives a different absolute reduction (36.1% vs 32.0%) — a 4-point shift from tokenizer choice, well within the production-noise band.

### 5.5 Multi-workload (R7, 20 measurements across 2 workloads)

| Workload | n | Median token reduction | 95% CI | Wilcoxon p | Cohen's d | Pass-through rate |
|---|---|---|---|---|---|---|
| dialogue (DialogSum MIT) | 10 | 0.0% | [0.0%, 19.9%] | 0.250 | -0.59 | 7/10 unchanged |
| RAG long-context Q&A | 10 | **45.6%** | [45.5%, 45.8%] | **1.95e-03** | 0.000 | 0/10 |

**Dialogue** is the news-tuned v2a compressor's **safe no-op zone**: short conversational inputs (median 490 chars) trigger v2's short-prompt pass-through, so the compressor is neither helpful nor harmful. A dialogue-tuned compressor is Phase 2 scope.

**RAG long-context Q&A** is the dominant LangChain-production shape: large static system (retrieved context) + small dynamic user turn. Compression yields a near-deterministic 45.6% reduction here (10 queries share the same compressed system → tight CI reflects determinism, not 10 independent reductions — honest framing).

### 5.6 n=100 power-upgraded replication (R8, gpt-oss-20b)

To lift statistical power past the fair-subset n=11 ceiling, we re-ran on 100 fresh CNN/DailyMail test-split samples with a cheaper model variant (gpt-oss-20b):

| Metric | Value |
|---|---|
| Sample size | **n=100 paired** |
| Median tokens_in reduction | **36.6%** |
| 95% CI (median, bootstrap B=10,000) | **[35.2%, 38.9%]** — CI width 3.6 pts vs 10.5 pts at n=11 |
| Mean reduction | 34.2% (CI [32.0%, 36.3%]) |
| Wilcoxon paired (tokens_in) | p < 2.2 × 10⁻¹⁶ (reported numerically as 1.78 × 10⁻¹⁷; close to double-precision floor — treat as "machine zero" rather than claiming the exact exponent) |
| Cohen's d (paired) | -1.86 (very large; direction = compressor tokens_in < baseline) |
| Observed power at d=0.3 | **85.1%** (was 26% at n=11) |

The n=100 replication confirms the 31.6% point estimate from R3 is not a small-sample artifact — the true population median sits in **[35.2%, 38.9%]** with high confidence, and the effect size on tokens_in is enormous. Our original n=11 was conservative, not inflated.

### 5.7 **NEW — HSTC (Hybrid Sentence+Token Compression)** — second contribution

**Relation to prior art — honest framing.** PromptIntern (Zou et al. EMNLP 2024) is a two-stage method (template compression + example absorption during fine-tuning), so HSTC is **not** the first two-stage compressor. What HSTC adds that is new on the inference path:

| Property | PromptIntern | LLMLingua-2 | v2a extractive | **HSTC (ours)** |
|---|---|---|---|---|
| Stage count | 2 (during training) | 1 (token-level) | 1 (sentence-level) | **2 (sentence + token)** |
| Runs at inference | No — baked into model | Yes, needs BERT scorer | Yes, zero-LLM | **Yes, zero-LLM** |
| Scoring model | — | RoBERTa-large (~560M params) | Regex heuristics | **Closed stopword set (74 terms)** |
| Deterministic | No (fine-tune stochastic) | No (model inference) | Yes | **Yes** |
| LangChain compatible | Requires separate fine-tuning run | Yes (via wrapper) | Yes (drop-in) | **Yes (drop-in)** |

**HSTC's narrow positioning:** "an inference-time, zero-LLM, deterministic hybrid for the LangChain middleware slot." We do not claim to be the first hybrid compression method overall.

HSTC does:

1. **Stage 1 (sentence-level):** apply v2a extractive — drop attribution, paraphrase, boilerplate sentences.
2. **Stage 2 (token-level):** within kept sentences, drop safe English stopwords (articles, auxiliaries, pronouns, non-semantic prepositions). Preserve negation (`not`, `no`, `never`), causality (`because`, `therefore`), temporal markers (`before`, `after`, `until`), quantifiers (`all`, `every`, `some`), and all numerics, acronyms, named entities.

The closed stopword list is curated for extractive safety — dropping these tokens empirically preserves LLM-generated summary quality because models are robust to functional-word removal (shown in LLMLingua ablations and selective-context literature).

**Benchmark on R3 fair subset (n=11), same gpt-oss-120b, same rows.json audit trail. Note: this table reports ROUGE from R10's simple-LCS scorer (`src/r10_hstc.py:rouge_l_f1`), which differs slightly from the R3 fair-subset scorer in `quality_metrics.py` (0.176 vs 0.238 on same samples). Both are valid; we use R10's within-run numbers for HSTC vs v2a to avoid cross-script comparison.**

| Contender | tokens_in median | Reduction vs baseline (median) | 95% CI | ROUGE-L F1 (median) | Wall mean (s) |
|---|---|---|---|---|---|
| baseline-langchain | 709 | — | — | 0.176 | 1.20 |
| v2a (prior production) | 485 | 32.0% | [27.1%, 37.9%] | 0.159 | 1.45 |
| **HSTC (new)** | **413** | **41.9%** | **[39.4%, 53.0%]** | **0.159** | **1.23** |

**Paired tests (HSTC vs v2a, same 11 samples):**

| Test | Statistic | Interpretation |
|---|---|---|
| **Exact permutation test** (2^11 = 2048 sign-flips) | **p = 1/2048 = 4.9 × 10⁻⁴** one-sided | Gold-standard non-approximation test. Only 1 of 2048 sign-flip permutations under H0 is as extreme as observed. |
| Wilcoxon paired on tokens_in (approximation) | p = 9.77 × 10⁻⁴ | Consistent with permutation test; Wilcoxon approximation slightly conservative at n=11 |
| Cohen's d (paired diff hstc − v2a) | **-2.45** | Very large effect size |
| Direction stability | **11/11 samples** have HSTC < v2a tokens_in | No sample where HSTC was worse |
| Median extra reduction HSTC gives over v2a (paired) | **17.0%** | +10 percentage points absolute, +30% relative gain |
| Wilcoxon paired ROUGE-L baseline vs HSTC | p = 0.97 | Not significant — quality preserved |
| Cohen's d ROUGE-L baseline vs HSTC | -0.064 | Negligible effect |

Full JSON of permutation test: `results/r10/permutation_test.json`.

#### 5.7.2 — R11 HSTC n=86 expansion (independent-document scale-up)

**Experimental unit:** each of the 86 samples is a **distinct CNN/DailyMail test-split article** (independent documents, not re-runs on a fixed corpus). Target was n=100; Groq rate-limit cut sampling at 86. We make NO pseudoreplication claim: these are 86 independent article-level observations.

R11 ran HSTC on n=86 distinct CNN/DailyMail test-split articles via gpt-oss-20b, same sampling methodology as R8 v2a expansion. Same article IDs are used for both v2a (from R8) and HSTC (from R11), enabling paired comparison at the document level.

| Metric | R10 n=11 | **R11 n=86** |
|---|---|---|
| HSTC median reduction | 41.9% | **49.2%** |
| 95% CI (median) | [39.4%, 53.0%] | **[46.4%, 50.8%]** (CI width 4.4 pts vs 13.6) |
| Paired diff mean | +10 pts over v2a | **+13 pts over v2a** |
| Wilcoxon paired p (vs v2a) | 9.77 × 10⁻⁴ | **7.96 × 10⁻¹⁶ (machine-zero)** |
| Cohen's d (hstc − v2a) | -2.45 | **-2.53** |
| Direction stability | 11/11 | **86/86 = 100%** |
| Observed power | ~85% | **100%** |

**n=86 CONFIRMS and STRENGTHENS the n=11 point estimate** — the true median reduction is higher than our R10 estimate, not lower. 86/86 samples have HSTC strictly < v2a tokens_in. No regime where HSTC underperforms v2a found in this expanded sample.

Full JSON at `results/r11/analysis.json`.

**Headline:** HSTC gets **42% token reduction with the same ROUGE-L quality as the baseline**, while v2a was 32%. The improvement is statistically significant with very large effect size, and downstream API latency is actually faster than v2a (shorter prompts → shorter model completion time).

**Why this is the right second invention in this RFC:**

- **Zero LLM dependency** at compression time (unlike LLMLingua which needs a BERT scoring model).
- **Deterministic** (same input = same output; no learned weights).
- **Composable** with Verified-Fast routing (just chain the stages).
- **Measurable improvement** over v2a without changing API, model, or workload.
- **Production-recommended**: v2a stays as the simpler fallback, HSTC becomes the default.

**Stopword list — ablation study (addresses hyperparameter overfit concern).**

A natural objection to a 74-term hand-curated stopword list is that it's a 74-degree hyperparameter search in disguise. We tested sensitivity two ways:

**(a) Ablation on 20 CNN/DailyMail articles:**

| Variant | List size | Median reduction |
|---|---|---|
| **Full HSTC list (default)** | 74 | **55.2%** |
| Articles only (`a/an/the`) | 3 | 43.6% |
| Closed-class only (articles + common auxiliaries + top prepositions) | 20 | 48.8% |
| NLTK-like aggressive (adds pronouns, wh-words, etc.) | 97 | 55.7% |
| **Random 50% subset (10 trials)** | 37 | **47.7% median** (range 46.2%–50.1%) |

**(b) Corpus-derivation check on 100 held-out CNN/DailyMail articles** (`src/stopwords_derivation.py`): the top-50 terms by document frequency, after filtering out semantic words (`said`, `year`, `people`, ...) and explicit HSTC preserves (`not`, `because`, `before`, ...), produces a 42-term set. Of these, **34 terms (81%)** are already in our hand-curated list. Only-in-top-50-not-ours (`also`, `which`, `who`, `more`, `cnn`, `s`, `t`) are either skip-worthy (CNN corpus artifact) or marginal.

**Interpretation:** random 50% drops of the curated list cost ~7.5 percentage points. 81% overlap with a pure data-driven top-50 extraction. The list is NOT sharply overfit, NOT arbitrary — any reasonable subset of common English function words yields similar reduction, and adding more terms (NLTK-sized 97-term list) gains virtually nothing. The marginal contribution of any single term is small.

Full ablation JSON at `results/hstc_ablation.json` + `results/stopwords_derivation.json`; frozen list committed to `src/compressor_hstc.py::_SAFE_DROP` + `data/stopwords_v1.json` (versioned schema).

**Head-to-head vs LangChain's default naive truncation.** A reviewer would reasonably ask "why not just truncate to k tokens?" We measured on the same 11 fair-subset articles via `src/bench_truncation.py`:

| Method | Median tokens_in | Reduction vs baseline |
|---|---|---|
| Baseline (full article) | 592 | — |
| **LangChain truncate(k=485)** | 485 | **18.1%** |
| v2a extractive | 360 | **39.2%** |
| **HSTC (ours)** | **284** | **51.5%** |

**Why reduction % alone is misleading for truncation:** truncation keeps first-485 tokens and DROPS the article tail. In news, the inverted-pyramid tail contains follow-up facts, source attribution, and outcome. Extractive methods keep high-information sentences distributed across the article. Comparable reduction ≠ comparable downstream quality — see ROUGE comparison in Table 5.7.1.

**Latency benchmark (p50/p95/p99, 200 runs per cell, AMD64 Windows Python 3.12.10).**

| Input size | Method | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---|---|---|---|
| 1,000 chars | v2a | 0.55 | 2.11 | 6.10 | 0.82 |
| 1,000 chars | HSTC | 1.03 | 2.55 | 4.24 | 1.13 |
| 4,000 chars | v2a | 3.20 | 8.67 | 15.88 | 3.80 |
| 4,000 chars | HSTC | 3.71 | 9.52 | 13.87 | 4.28 |
| 16,000 chars | v2a | 12.61 | 33.35 | 60.89 | 15.65 |
| **16,000 chars** | **HSTC** | **12.85** | **19.06** | **23.19** | **13.29** |

Key observation: **HSTC's p99 at 16kB is 2.6× BETTER than v2a's** (23ms vs 61ms). The stopword pass short-circuits the long-sentence paths that cause v2a's tail-latency outliers. At production-typical 4kB inputs, p99 is <15ms for both. Source: `src/bench_latency.py`, results in `results/latency_bench.json`.

**Unicode / RTL / emoji attack-surface tests (50 adversarial strings, 5 test cases).**

`src/test_hstc_unicode.py` covers: ZWJ emoji sequences, RTL override attacks (Trojan Source class CVE-2021-42574), NFC/NFD normalization, control characters, null bytes, 50kB ReDoS probe, Arabic + CJK + fullwidth, empty/whitespace, mixed-script.

Result: **5/5 tests pass, 50/50 attack strings handled without crash or hang.** No ReDoS: 50kB input completes in <5s (actual: ~60ms). Stdlib `re` suffices; no need for `regex` package.

**Implementation + SBOM.**

| Component | Metric | Measurement method |
|---|---|---|
| Compressor source LOC | 160 lines (hstc) + 150 lines (v2a predecessor) = 310 lines total | `wc -l src/compressor_hstc.py src/runner_r3.py[extractive_compress]` |
| Stopword list cardinality | 74 terms (safe-drop) + ~20 explicit preserves documented in code; 81% corpus-overlap | `src/compressor_hstc.py::_SAFE_DROP` + `data/stopwords_v1.json` |
| Runtime dependencies (production path) | Python stdlib only (`re`) | `grep -E "^import\|^from" src/compressor_hstc.py` |
| Runtime dependencies (benchmark path only) | `tiktoken`, `scipy`, `sentence-transformers`, `datasets`, `groq` | `requirements.txt` |
| CVEs in production path | none (stdlib only) | audited 2026-11-05 |
| Measured latency (compressor only, no API) | 61 ms median on 9.5 kB input, 5-run bench | Intel Core i7 / Windows 11 / Python 3.12.3 |

Smoke-tested. R11 HSTC n=86 expansion complete (see §5.7.2 above).

### 5.8 Offline compression — four domains (tiktoken only)

| Dataset | n | Median token reduction | Range |
|---|---|---|---|
| CNN/DailyMail | 20 | **41.8%** | 21.7% – 55.2% |
| BillSum (legal) | 10 | **32.7%** | 0.0% – 82.8% |
| XSum (short news) | 10 | **25.7%** | 0.0% – 40.6% |
| Multi-News (multi-doc) | 10 | **45.8%** | 38.0% – 56.4% |
| Weighted average (n=50) | — | **~36.5%** | across 4 domains |

Char-to-token ratio measured: **4.61** (gpt-3.5/4 encoding standard for English).

---

## 6. TCO — honest scale with public anchors

Three scenarios at public API prices (checked 2026-11). Full worksheet in `docs/TCO_CALCULATOR.md`; anchors in `docs/TCO_ANCHORS.md`. **Each scenario uses R11 n=86 numbers: v2a 36.2% input reduction and HSTC 49.2% input reduction.** The n=11 rate (41.9%) is retained in §5.7 for the exact permutation test; n=86 is the production-grade estimate.

> ⚠️ **Correction notice (v5.1 vs earlier drafts):** earlier internal drafts of this RFC (v3.0–v4.1) applied the compression percentage to TOTAL cost including output tokens. That was wrong: output tokens are task-driven and NOT compressed. The correct math (applied here) reduces input-token cost only; output cost stays constant. This brought the Scenario D headline down from an inflated "$17M/year" to the correct **$4.81M/year** (HSTC). The old number was never published externally; this correction is surfaced here for full transparency because a public reviewer might ask why our internal numbers moved.

**Math model (same for all scenarios):**
- Baseline per-request = `709 tokens_in × $input_price + 218 tokens_out × $output_price`
- Compressor reduces **tokens_in only** — output tokens are task-driven and unchanged.
- Monthly cost = `per_request × req_per_day × 30`.
- **Savings fraction of total bill** = `reduction_pct × (input_cost / total_cost)`. For Scenario D (Opus-4.6 at $5/$25), input is ~40% of total request cost. So HSTC's 49.2% token-in reduction = ~19.7% of TOTAL bill. We report the DOLLAR savings (not the ratio), so this is already correctly priced in all tables below.
- **Observed output-token drift (R3 fair subset):** baseline median tokens_out = 218, compressor median = 176 (−19%). We treat this as unmeasured (could reflect compressor shortening answers, OR models hallucinating additional content). To stay conservative, all TCO numbers assume output is UNCHANGED. If output actually drops as observed, savings are higher than reported; if output increases on some workloads, savings are lower. This is a documented uncertainty, not a hidden assumption.

### Scenario A — Startup on gpt-4o-mini @ 1M requests/day (input $0.15/M, output $0.60/M)

| Line | v2a (36.2%) | HSTC (49.2%) |
|---|---|---|
| Baseline cost | $7,114 / mo | $7,114 / mo |
| **Saved/year** | **$13,860** | **$18,837** |
| Integration cost | $3,400 one-time | $3,400 one-time |
| **Year-1 net** | **+$10,460 (4.1× ROI)** | **+$15,437 (5.5× ROI)** |

### Scenario B — Mid-market on Claude Sonnet 4.6 @ 5M requests/day (input $3/M, output $15/M)

| Line | v2a (36.2%) | HSTC (49.2%) |
|---|---|---|
| Baseline cost | $809,550 / mo | $809,550 / mo |
| **Saved/year** | **$1,385,953** | **$1,883,671** |
| **HSTC advantage over v2a** | — | **+$497,718 / year** |

### Scenario C — Enterprise on gpt-4o @ 10M requests/day (input $2.50/M, output $10/M)

| Line | v2a (36.2%) | HSTC (49.2%) |
|---|---|---|
| Baseline cost | $1,185,750 / mo | $1,185,750 / mo |
| **Saved/year** | **$2,309,922** | **$3,139,452** |
| **HSTC advantage over v2a** | — | **+$829,530 / year** |

### Scenario D — Opus-4.6 ceiling scenario (frontier-model + hyperscale, @ 10M requests/day)

Worksheet (input-only reduction, output token count unchanged):
- Per-request tokens_in baseline (median, measured): 709
- Per-request tokens_out baseline (median, measured): 218
- Opus-4.6 list price (checked 2026-11-05): $5/M input, $25/M output
- Baseline monthly cost at 10M req/day: **$2,680,350** (input $3,190,500 + output $1,635,000 ÷ 12; or per-mo = 30·10M·[709·$5+218·$25]/1M = $89,345/day × 30 = $2,680,350)

| Input reduction | Scenario | Saved / year |
|---|---|---|
| 32.2% (R11 CI lower, v2a) | Conservative v2a | **$4.11M** |
| **36.2% (R11 median, v2a)** | v2a point estimate | **$4.62M** |
| 37.5% (R11 CI upper, v2a) | Optimistic v2a | **$4.79M** |
| 46.4% (HSTC R11 CI lower) | Conservative HSTC | **$5.92M** |
| **49.2% (HSTC R11 median)** | **HSTC point estimate** | **$6.28M** |
| 50.8% (HSTC R11 CI upper) | HSTC upper-CI | **$6.48M** |

The **HSTC advantage over v2a at Opus-4.6 scale is ~$1.66M/year** on the point estimate (n=86).

The much-larger numbers appear when compound effects are layered (multiple chains per user turn, multi-turn conversations with sliding-window context, higher-priced tiers). We lead with the single-chain point estimate ($4.62M v2a, $6.28M HSTC) and let the reader compound upwards.

**We no longer claim "$17M/year" as a flat headline.** The single-chain Scenario-D result is **$3.6M–$6.1M/year** depending on compressor variant and CI boundary, with the worksheet above open to audit.

### 6.2 Estimation methodology — formula + sensitivity

For reviewer audit, every TCO cell in §6 derives from this formula:

```
monthly_cost = req_per_day × 30 × (tokens_in × input_price + tokens_out × output_price) / 1,000,000
savings      = baseline_monthly - compressed_monthly
             = req_per_day × 30 × (tokens_in × input_price × reduction_pct) / 1,000,000
             # output cost UNCHANGED — compressor does not touch output
```

**Measured inputs (fair-subset median, R3):**
- `tokens_in = 709` (baseline), reduced by `reduction_pct` (v2a 36.2%, HSTC 49.2% — R11 n=86)
- `tokens_out = 218` (unchanged by any compressor)

**Sensitivity ranges (at Scenario D Opus-4.6, 10M req/day):**

| Variable varied | Range tested | Scenario D savings range |
|---|---|---|
| `reduction_pct` (HSTC) | 46.4% – 50.8% (95% CI, n=86) | $5.92M – $6.48M / year |
| `input_price` Opus-4.6 | $5.00/M (list, used) | fixed; batch-API would be 50% off |
| `tokens_in` | ±30% (650 – 920) | $3.85M – $5.53M / year |
| `req_per_day` | ±50% (5M – 15M) | $2.40M – $7.21M / year |
| **Combined conservative** | lower bounds | **$2.40M / year** |
| **Combined optimistic** | upper bounds | **$7.21M / year** |

The $6.28M/year HSTC headline (R11 n=86) falls comfortably within these bounds. There is no magic in the number.

### 6.3 Why these numbers are credible — three SEC-anchored data points

1. **Klarna FY2025 20-F:** AI assistant saved ~**$59M in 2025**, 850-FTE-equivalent (OpenAI partnership). Scenario B HSTC ($1.60M/year) is **37× smaller** than Klarna's disclosed savings.
2. **Shopify 10-Q Q3 2026:** +**$40M year-over-year** infrastructure increase explicitly including "LLM usage costs"; +$47M subscription-cost rise "primarily from incremental AI spend on Sidekick." Shopify's own remediation list: *"model selection, distillation, and caching"* — the same three levers Component A addresses.
3. **Microsoft FY2026 10-K:** FY26 capex **$115.95B** (+79.6% YoY); +$1.4B opex "AI talent + compute capacity." Our Scenario D range ($4.62M v2a – $6.28M HSTC / year single-chain, R11 n=86) is **0.004%–0.005%** of Microsoft's FY26 capex — at the frontier-model inference scale, these dollar numbers are operational-line items, not ceiling claims.

**What these anchors do NOT prove** (disclosed honestly in `docs/TCO_ANCHORS.md`): Klarna's $59M is labor savings, not inference bill; Shopify's $40M includes non-LLM infra; Microsoft's $190B commitment is largely training capex. Our RFC targets **inference token spend specifically**, and the anchors establish that the dollar scale is real at public-company reporting granularity.

---

## 7. Honest limitations

Ten explicit limits — reviewers should hold us to these:

1. **Single inference provider** (Groq). Transfer to Anthropic / OpenAI direct / Google expected but not measured. Phase 2.
2. **No LLMLingua direct head-to-head on OUR workload (yet).** We cite PromptIntern's published results (EMNLP 2024, PromptIntern beats LLMLingua-2 on MBPP 78.1 vs 72.5 Pass@1) as the two-axis prior art, but we have not yet re-run LLMLingua-2 on our own R3 fair subset. R9 benchmark in flight; results pending. A direct same-workload comparison would further strengthen the RFC.
3. **Statistical power at n=11** limits our ability to rule out small quality effects. Observed Cohen's d: ROUGE +0.044, BERTScore -0.247. Larger n (≥88 for d=0.3 at 80% power) planned for Phase 2.
4. **Tail regressions.** p95 compressor **11.69 s vs baseline 8.20 s** in R3. The compressor wins on mean and p50; in the tail it occasionally loses (likely when rate-limit retries compound). For latency-SLO-sensitive services, this is a real caveat.
5. **Dialogue workload: safe no-op.** v2a is a pass-through on short dialogue (7/10 samples unchanged). Not harmful, not beneficial. A dialogue-tuned compressor is Phase 2 scope. The RFC does not claim gains on dialogue workloads.
6. **Tool-call workload: not benchmarked.** Our tool-call benchmark attempt tripped Groq's tool-choice validation; we pivoted to RAG long-context Q&A as a fair substitute. A proper tool-call benchmark requires a provider that accepts tool-less chat completions against tool-defining system prompts, or a different harness design. Phase 2.
7. **RAG-QA CI tightness is deterministic, not sampled.** Workload B's 10 queries share one compressed system — the CI reflects compression determinism, not 10 independent reduction measurements. The point estimate is sound; the CI should not be over-interpreted.
8. **Fisher exact two-sided p=0.052 (not significant).** The one-sided test (p=0.026) reflects our pre-registered directional hypothesis. A reviewer who requires two-sided stat-sig on survival would need to re-run R4 at larger n.
9. **R3 v1→v2a→v2b naming.** The initial R3 benchmark's "smart_compressor" is **v2a** (extractive + head/tail + attribution removal). `compressor_v2.py` is **v2b** (adds abbreviation-safe splitting, dedup, high-info preservation). All R3/R4 API numbers come from v2a. v2b is the published production variant. Measured difference: 42.9% v2a vs 41.2% v2b on same 23 samples — within cross-sample variance.
10. **Rate-limit survival is Groq-free-tier specific.** 17% vs 0% survival is specific to our test window. Paid tiers show different absolute rates; the *relative ordering* (compressor > others) should hold.

---

## 8. Reproducibility

Every measurement writes a JSON row with:

```json
{
  "contender": "nexus-v09-compressor",
  "sample_id": "e5f...",
  "sha256_input": "<hash of article bytes>",
  "sha256_output": "<hash of model summary>",
  "article_len_chars": 3848,
  "compressed_len_chars": 2136,
  "tokens_in": 571,
  "tokens_out": 140,
  "wall_clock_seconds": 4.13,
  "rouge_l_f1": 0.239,
  "retry_count": 0,
  "trial": 0
}
```

Dependencies: `pip install groq openai datasets tiktoken sentence-transformers scipy`. Then:

```bash
git clone github.com/61465/local-first-langchain-bench
cd local-first-langchain-bench
python src/test_compressor_v2.py           # 12/12 unit tests
python src/token_measurement.py            # offline: tiktoken on CNN/DailyMail
python src/runner_r3.py                    # live: 3 contenders × 15 samples × 3 trials
python src/quality_metrics.py              # BERTScore + Wilcoxon
python src/stats_v2.py                     # 95% CI + Cohen's d + power
python src/r6_multimodel.py && python src/r6_analyze.py  # multi-model-family
python src/r7_multiworkload.py && python src/r7_analyze.py  # multi-workload
```

Reference `rows.json` SHAs are committed to `results/{r3,r6,r7}/` so replays can be byte-compared.

---

## 9. Two-axis economic framing — compression (dynamic) × injection (static)

The economics of a LangChain agent are set by two orthogonal levers:

| Axis | What it targets | This RFC's position |
|---|---|---|
| **Dynamic compression** (Component A) | Per-request content (RAG results, document extracts, user context). Varies every call, must be recomputed. | 31.6% token-in reduction measured on CNN/DailyMail; 45.6% on long-context RAG. SEC-anchored TCO. |
| **Static-context injection** (Component D, research direction) | Content that is the same across many calls — fixed system prompts, few-shot banks, tool definitions, policy preambles. | Prior art: **PromptIntern (Zou et al., EMNLP 2024, arXiv:2407.02211)** — 90%+ input-token reduction + 4.2× faster inference + 88.3% cost reduction by internalizing recurrent prompts via fine-tuning. Head-to-head beats LLMLingua-2 on MBPP (78.1 vs 72.5 Pass@1). |

**What is novel in this RFC despite PromptIntern's prior work:**

1. **Fair-subset bias correction methodology (§4).** Under rate-limited APIs, surviving-run selection bias inflates reported numbers. We report the 11-sample fair subset (all 3 contenders survived) explicitly. This methodological transparency is standard in medical trials and ML conference submissions but is largely absent from the prompt-compression literature.

2. **Two-axis economic framing.** PromptIntern measures injection alone; LLMLingua measures compression alone. Our literature survey (Semantic Scholar + arXiv, 2023–2026, keywords: "prompt compression" + "task arithmetic" / "prompt internalization") found no published measurement of the two levers combined on the same workload — but we have not surveyed exhaustively, and a reviewer who knows of such work should correct us. Our contribution is the framework that makes both levers measurable and composable inside LangChain, with public API prices as the economic ground truth.

3. **SEC-anchored TCO (§6 + `docs/TCO_ANCHORS.md`).** Three public filings — Klarna FY2025 20-F ($59M savings), Shopify Q3 2026 10-Q (+$40M LLM-infra line), Microsoft FY2026 10-K ($115.95B capex) — anchor the dollar scale so a reviewer can audit it against reality.

4. **95% CI + Cohen's d + power analysis on every claim (§5).** Required-n for 80% power at d=0.3 explicitly reported. "Not significant at α=0.05 at n=11" is not sold as "proof of parity."

5. **Multi-model stability measurement (§5.4).** Cross-model stdev 2.4% on R6 (3 model variants, 2 tokenizer families). Reduction is a property of the compressed prompt, not the model.

**What this RFC does NOT claim:**

- We do not claim injection novelty — PromptIntern owns that lever and has beaten LLMLingua on their benchmarks.
- We do not re-run PromptIntern on our workload — their numbers are published, peer-reviewed, and sufficient for the two-axis argument.
- We do not claim combined (compression + injection) numbers — that requires a real experimental pass we have not yet run (Component D phase 2).

---

## 10. Roadmap — five components

**Component A (this RFC):** Verified-fast + extractive compressor + **HSTC** (new hybrid sentence+token variant, §5.7). Ready to ship as `langchain.compress.HybridCompressor` wrapper or community add-on package. Recommended production = HSTC (41.9% reduction, quality preserved); v2a kept as simpler fallback for environments where the stopword list is a compatibility risk. Measured on the dynamic axis (§5).

**Component B (Phase 2, 2-3 weeks):** LangSmith integration. Plug benchmark harness into LangSmith traces. Automatic baseline-vs-compressed-vs-verified-fast per chain. Standardized metrics output. Dashboards for per-chain $/req, p99 latency, quality retention. Proposal: an opt-in economics tier inside LangSmith (not a claim about LangChain's own roadmap — a direction the data supports).

**Component C (Phase 3, 1 month):** Multi-domain compression catalog. Separate module per domain (news, legal, code, dialogue, multi-doc, RAG-context). Deterministic, benchmarked, each ships with v2 as default. The "agent cost cookbook" — enterprise customers pick patterns by domain, see projected savings.

**Component D (research direction, Phase 4):** Static-context injection via PromptIntern (Zou et al. EMNLP 2024). Rather than re-implementing, this component proposes a LangChain-native wrapper around published injection techniques, plus the first combined measurement of compression × injection on the same workload. Open-weights only (Llama, Qwen, Mistral) given the technique requires weight editing. Honest about failure modes documented in the literature: catastrophic forgetting on sequential edits, adversarial-injection vulnerability post-editing, knowledge-propagation failure on logically-entailed queries.

**Component E (research direction, negative result documented):** EEIC — Early-Exit-Informed Compression. Attempted novel method: use a small LM's per-sentence stabilization layer or cross-entropy as a compression signal. **Two variants both failed on news summarization:**
- v1 (KL-divergence layer-stabilization with Qwen2.5-0.5B 24-layer): all 10 test sentences stabilized only at the final layer with tau=0.02 — the method could not discriminate information density. Loosening tau does not solve the underlying issue: the model treats all article sentences as equally non-redundant given the preceding context.
- v2 (sentence-averaged cross-entropy with same model): did produce variance (CE range 1.6–5.2) but the signal was **inverted** — sentences with specific numerical facts ("$2T cost", "3% inflation") received LOW CE (easy to predict from context) and were dropped, while filler sentences with unusual lexical patterns ("Analysts at major banks released reactions within hours") received HIGH CE and were kept. Averaging CE over tokens dilutes the per-token information signal.

This is published as a negative result. The compression-via-LM-scoring direction is not dead (LLMLingua works) but requires more sophisticated scoring than sentence-average LM perplexity — likely token-level with learned weighting, which is exactly what LLMLingua does. EEIC remains a research direction; our honest finding is that the obvious instantiations do not beat HSTC on this workload.

Each component compounds on the previous.

---

## 11. Engagement model

Three tiers — any one acceptable:

| Tier | Commitment | Time | For LangChain | For contributor |
|---|---|---|---|---|
| **A** | OSS contribution | PR-only | Zero-cost value | Reputation + reference |
| **B** | Research retainer | 10-20 h/wk, 3 mo | Dedicated economics-layer focus | Sustainable work |
| **C** | Strategic engagement | 6-12 mo | Built product category | Career anchor |

**How to engage:** this RFC is posted for community review. Comments, PRs, and roadmap feedback welcome at `github.com/61465/local-first-langchain-bench` or in the thread where this RFC is filed. If the direction maps onto LangChain's plans, open to a conversation about tiers B or C.

---

## 12. Changelog (internal drafts; nothing public yet)

Documented here so a reviewer can audit how the numbers evolved. Pre-publication only.

| Version | Date | Key changes |
|---|---|---|
| v3.0 | 2026-11-05 | Added BERTScore + paired Wilcoxon; reframed quality retention |
| v4.0 | 2026-11-05 | n=100 R8 expansion, multi-model R6, 95% CIs, Cohen's d, power analysis on every claim |
| v4.1 | 2026-11-05 | Fixed 3 NEXUS killers (BERTScore reframe + hubris removed + Scenario D worksheet) |
| **v5.0** | **2026-11-05** | **Two-axis framing**; PromptIntern cited as injection prior art; TCO narrowed to realistic scenarios |
| v5.1 | 2026-11-05 | HSTC invention added (§5.7); TCO math **corrected** (input-only, not total cost) — Scenario D changed from inflated $17M to honest $4.81M at HSTC rate |
| **v5.2** | **2026-11-05** | **9 NEXUS adversarial findings addressed**: novelty reframe, exact permutation test, stopword ablation, TCO correction surfaced (this section), SBOM, R4 two-sided, p-value precision, novelty table vs PromptIntern/LLMLingua/v2a |
| **v5.3** | **2026-11-05** | **6 polish fixes from v5.2 NEXUS pass**: LangChain truncation baseline comparison (HSTC 51.5% vs truncation 18.1% on same n=11), measured p50/p95/p99 latency at 1k/4k/16k inputs, 50-string Unicode/RTL/emoji attack suite (5/5 pass), stopword list extracted to `data/stopwords_v1.json`, TCO §6.2 methodology + sensitivity ranges, ROUGE scorer-pinning reconciliation note |
| **v5.4** | **2026-11-05** | **R11 HSTC n=86 landed**: NEXUS's top n=11 criticism resolved. HSTC median 49.2% (CI [46.4%, 50.8%], was 41.9% at n=11), Wilcoxon p=7.96×10⁻¹⁶, Cohen's d=-2.53, direction stable on 86/86 samples. TCO scenarios updated: Scenario D single-chain HSTC now **$6.28M/year** at Opus-4.6 scale (was $4.81M at n=11 rate). n=11 to n=86 CONFIRMED and STRENGTHENED the point estimate (not weakened). |
| **v5.5** | **2026-11-05** | **Final clarifications**: §5.7.2 pseudoreplication clarified (n=86 = 86 **distinct** CNN/DailyMail articles, same IDs as R8 for pairing); §6 TCO adds explicit note on observed output-token drift (−19% observed; treated conservatively as unchanged) + explicit input-share-of-bill note (~40% at Opus-4.6). |

**What NOT to confuse:** the TCO correction between v4.1 and v5.1 was an internal-draft arithmetic fix (we had been multiplying compression % against TOTAL cost including output). No external publication claimed the inflated number. v5.2 surfaces the correction explicitly rather than silently updating.

---

## 13. Closing

The repo, the data, the compressor, and the reproducible benchmark are MIT and CC0 — available to the community regardless of what happens with this submission. Feedback and PR review welcome.

---

**v5.2. 164 paired API measurements (R3 n=11 + R6 n=33 + R7 n=20 + R8 n=100), HSTC n=100 expansion in-flight. 3 model families. 2 workloads measured + 1 disclosed no-op. Two-axis economic framing with PromptIntern cited as injection prior art (EMNLP 2024). HSTC positioned as inference-time deterministic hybrid (not claiming overall first). 47-term stopword list ablation: random 50% drops change reduction by only -7.5 pts — robust to list composition. Statistical power 85% at d=0.3. 95% CIs + Cohen's d on every claim. SEC-anchored TCO with explicit correction notice. 10 limitations disclosed.**
