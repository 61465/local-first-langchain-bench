"""
Token-exact measurement for compression ratio reporting.

Replaces char-based reduction with tiktoken-based reduction so the
RFC's dollar projections are rigorous (not "chars ≈ tokens" assumption).

Uses cl100k_base (OpenAI's gpt-3.5+ encoding) as a widely-applicable
default. For model-specific results, a user can pass encoding="o200k_base"
(gpt-4o) or similar.
"""
from __future__ import annotations

from typing import Optional
import tiktoken


_ENCODER_CACHE: dict[str, "tiktoken.Encoding"] = {}


def get_encoder(encoding_name: str = "cl100k_base") -> "tiktoken.Encoding":
    """Cached encoder — avoid re-init per call."""
    if encoding_name not in _ENCODER_CACHE:
        _ENCODER_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODER_CACHE[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Return exact token count for `text` under `encoding_name`."""
    if not text:
        return 0
    return len(get_encoder(encoding_name).encode(text))


def compression_ratio(
    original: str,
    compressed: str,
    *,
    encoding_name: str = "cl100k_base",
) -> dict:
    """Return detailed compression metrics.

    Returns:
      {
        "orig_chars": N, "comp_chars": M, "char_reduction": 0.X,
        "orig_tokens": N, "comp_tokens": M, "token_reduction": 0.Y,
        "char_to_token_ratio": float,  # chars per token (English ~4)
      }
    """
    orig_tokens = count_tokens(original, encoding_name)
    comp_tokens = count_tokens(compressed, encoding_name)
    orig_chars = len(original)
    comp_chars = len(compressed)

    return {
        "orig_chars": orig_chars,
        "comp_chars": comp_chars,
        "char_reduction": (orig_chars - comp_chars) / orig_chars if orig_chars else 0.0,
        "orig_tokens": orig_tokens,
        "comp_tokens": comp_tokens,
        "token_reduction": (orig_tokens - comp_tokens) / orig_tokens if orig_tokens else 0.0,
        "char_to_token_ratio": orig_chars / orig_tokens if orig_tokens else 0.0,
    }


if __name__ == "__main__":
    import sys, statistics as st, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from compressor_v2 import extractive_compress_v2
    from workload_summarization import load_cnn_dailymail

    print("=== tiktoken-verified v2 compressor on CNN/DailyMail (n=20) ===")
    data = load_cnn_dailymail(n=20)
    stats = []
    for i, d in enumerate(data):
        orig = d["article"]
        comp = extractive_compress_v2(orig)
        m = compression_ratio(orig, comp)
        stats.append(m)
        print(f"  [{i:02d}] {m['orig_tokens']:>5}t -> {m['comp_tokens']:>5}t "
              f"(token={m['token_reduction']*100:>5.1f}%, char={m['char_reduction']*100:>5.1f}%)")

    print()
    char_reds = [s["char_reduction"] * 100 for s in stats]
    token_reds = [s["token_reduction"] * 100 for s in stats]
    print(f"Chars reduction  median: {st.median(char_reds):.1f}% (range {min(char_reds):.1f}-{max(char_reds):.1f}%)")
    print(f"Tokens reduction median: {st.median(token_reds):.1f}% (range {min(token_reds):.1f}-{max(token_reds):.1f}%)")
    ratios = [s["char_to_token_ratio"] for s in stats]
    print(f"Chars per token median: {st.median(ratios):.2f} (for gpt-3.5/4 encoding)")
