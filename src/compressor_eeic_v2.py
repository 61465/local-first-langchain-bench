"""EEIC v2 — Perplexity-informed compression.

Pivoting from layer-stabilization (v1 showed all sentences stabilize at final
layer — tau too strict) to cross-entropy perplexity per sentence.

**Theory:** given a prefix (prior sentences in the article), the LLM's per-token
cross-entropy over the current sentence measures surprisal. Low CE = redundant
(the model could already predict this content). High CE = information-dense.

This is standard LM eval, applied as a compression signal — still novel in
prompt compression (literature uses supervised scorers like LLMLingua-2's BERT,
not LLM perplexity directly).
"""
from __future__ import annotations

import re
from typing import Optional

_MODEL = None
_TOK = None
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _load():
    global _MODEL, _TOK
    if _MODEL is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    print("[EEIC-v2] Loading Qwen2.5-0.5B...")
    _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype="float32", trust_remote_code=True,
    )
    _MODEL.eval()
    print(f"[EEIC-v2] Model loaded")


def _sentence_ce(sentence: str, prefix: str) -> float:
    """Average per-token cross-entropy of `sentence` given `prefix`.

    Higher = more surprising (info-dense). Lower = more predictable (redundant).
    """
    import torch
    import torch.nn.functional as F
    _load()

    if prefix:
        prefix_ids = _TOK(prefix, return_tensors="pt", add_special_tokens=True).input_ids
        full_ids = _TOK(prefix + " " + sentence, return_tensors="pt", add_special_tokens=True).input_ids
    else:
        prefix_ids = _TOK("", return_tensors="pt", add_special_tokens=True).input_ids
        full_ids = _TOK(sentence, return_tensors="pt", add_special_tokens=True).input_ids

    full_ids = full_ids[:, :512]  # cap
    prefix_len = min(prefix_ids.shape[1], full_ids.shape[1] - 1)
    if full_ids.shape[1] <= prefix_len + 1:
        return 0.0

    with torch.no_grad():
        outputs = _MODEL(full_ids)
    logits = outputs.logits  # [1, seq, vocab]
    # Score tokens AFTER the prefix (the sentence tokens)
    targets = full_ids[:, prefix_len + 1 :]  # [1, S]
    pred_logits = logits[:, prefix_len : -1, :]  # [1, S, vocab]  (shifted by one)
    if targets.shape[1] == 0 or pred_logits.shape[1] == 0:
        return 0.0
    # Align shapes
    L = min(targets.shape[1], pred_logits.shape[1])
    targets = targets[:, :L]
    pred_logits = pred_logits[:, :L, :]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, L]
    ce = -gathered.mean().item()
    return ce


def eeic_v2_compress(article: str, target_ratio: float = 0.65) -> str:
    """Score sentences by LM cross-entropy given prefix; drop lowest (= most predictable)."""
    sentences = [s.strip() for s in _SENTENCE_RE.split(article) if s.strip()]
    if len(sentences) <= 3:
        return article

    scores = []
    for i, s in enumerate(sentences):
        prefix = " ".join(sentences[:i])  # ALL prior sentences as context
        try:
            ce = _sentence_ce(s, prefix)
        except Exception as e:
            ce = 999.0
        scores.append((ce, i, s))

    # Higher CE = keep (info-dense). Sort descending.
    scores_sorted = sorted(scores, key=lambda x: -x[0])
    total_chars = sum(len(s) for s in sentences)
    target_chars = int(total_chars * target_ratio)

    kept = set()
    acc = 0
    for ce, i, s in scores_sorted:
        if acc >= target_chars:
            break
        kept.add(i)
        acc += len(s)

    return " ".join(sentences[i] for i in sorted(kept))


def score_debug(article: str) -> list:
    sentences = [s.strip() for s in _SENTENCE_RE.split(article) if s.strip()]
    out = []
    for i, s in enumerate(sentences):
        prefix = " ".join(sentences[:i])
        try:
            ce = _sentence_ce(s, prefix)
        except Exception:
            ce = -1.0
        out.append((ce, s[:80] + ("..." if len(s) > 80 else "")))
    return out


if __name__ == "__main__":
    test_article = (
        "The President of the United States announced a new economic plan yesterday. "
        "She told reporters at the White House the plan will take effect in 2026. "
        "Officials said the plan will reduce inflation by 3 percent. "
        "The plan will cost $2 trillion over ten years. "
        "Critics argue the plan is too expensive and will worsen the deficit. "
        "Supporters point to projected job growth of 4 million by 2027. "
        "The announcement comes after months of congressional debate. "
        "She emphasized the plan was developed with input from economists. "
        "Analysts at major banks released reactions within hours. "
        "Markets fell by 1 percent on the news."
    )
    print(f"Input: {len(test_article)}ch\n")
    print("Per-sentence cross-entropy (higher = more info-dense):")
    scored = score_debug(test_article)
    scored_sorted = sorted(scored, key=lambda x: -x[0])
    for ce, text in scored_sorted:
        print(f"  CE={ce:5.2f}  '{text}'")

    print()
    c = eeic_v2_compress(test_article, target_ratio=0.5)
    print(f"EEIC-v2 compressed ({(1-len(c)/len(test_article))*100:.1f}% reduction):")
    print(c)
