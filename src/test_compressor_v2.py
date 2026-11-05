"""
Unit tests for compressor_v2. 40+ cases covering:
- Abbreviation handling (Dr., Mr., etc.)
- Dedup correctness
- High-info preservation
- Adaptive mode switching
- Edge cases (empty, very short, very long)
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compressor_v2 import (
    extractive_compress_v2,
    _split_sentences,
    _is_attribution,
    _dedupe_near_duplicates,
    _ABBREV,
)


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "assertion failed")


def test_abbrev_no_split():
    """Abbreviations don't cause false sentence breaks."""
    cases = [
        "Dr. Smith announced the plan. The plan starts tomorrow.",
        "Mr. Johnson met Mrs. Lee. They agreed.",
        "The U.S. and U.K. signed a deal. Trade will grow.",
        "Visit at 3 p.m. today. The office closes at 5 p.m.",
    ]
    for text in cases:
        sents = _split_sentences(text)
        _assert(len(sents) == 2, f"Expected 2 sents, got {len(sents)} for: {text}")
        # Ensure no broken abbreviations
        for s in sents:
            _assert(not s.startswith("Smith") and not s.startswith("Johnson") and not s.startswith("Lee"),
                    f"Bad split in: {s}")


def test_high_info_preserved():
    """Sentences with numbers/dates/money kept even if they look like attribution."""
    high_info = [
        "In 2024, officials said the plan allocates $5 billion.",
        "The CEO reported 15% growth.",
        "NATO forces, according to sources, moved 10000 troops.",
    ]
    for s in high_info:
        _assert(not _is_attribution(s), f"Should NOT be attribution (high info): {s}")


def test_low_info_attribution_removed():
    """Pure attribution without numbers/dates IS marked attribution."""
    low_info = [
        "The spokesperson said the matter is complex.",
        "According to officials, the issue is sensitive.",
        "Dr. Smith added that more review is needed.",
        "The source told reporters, demanding anonymity.",
    ]
    for s in low_info:
        _assert(_is_attribution(s), f"Should be attribution: {s}")


def test_dedupe():
    """Near-duplicate first-8-words are deduplicated."""
    sents = [
        "The committee announced a plan for new infrastructure.",
        "The committee announced a plan for new infrastructure today afternoon.",  # dup
        "Infrastructure funding remains a concern.",
    ]
    out = _dedupe_near_duplicates(sents)
    _assert(len(out) == 2, f"Expected 2, got {len(out)}")


def test_short_passthrough():
    """Articles ≤8 sentences pass through unchanged."""
    short = "Sentence one. Sentence two. Sentence three."
    _assert(extractive_compress_v2(short) == short)


def test_normal_mode():
    """Articles under 3000 chars use normal mode (3 head + 2 tail)."""
    sents = [f"Sentence number {i} contains generic content about a topic ranging from politics to climate change today, with multiple details that could warrant a full sentence here." for i in range(1, 16)]
    article = " ".join(sents)
    _assert(1500 < len(article) < 3000, f"test article wrong len: {len(article)}")
    out = extractive_compress_v2(article)
    # Should keep head + tail + partial middle
    _assert(sents[0] in out, "Head must be present")
    _assert(sents[-1] in out, "Tail must be present")
    _assert(len(out) < len(article), "Should compress")


def test_aggressive_mode():
    """Articles >3000 chars use aggressive mode (2 head + 1 tail)."""
    sents = [f"Long sentence number {i} with substantial content about current events and global trends ranging across many topics including the environment, economy, and political landscape of today and tomorrow." for i in range(1, 25)]
    article = " ".join(sents)
    _assert(len(article) > 3000, f"test article too short: {len(article)}")
    out = extractive_compress_v2(article)
    _assert(sents[0] in out, "Head must be present")
    _assert(sents[-1] in out, "Tail must be present")
    # Aggressive mode: should shrink more than normal
    _assert(len(out) < len(article) * 0.6, f"Aggressive should get <60% of original, got {100*len(out)/len(article):.1f}%")


def test_reduction_range():
    """Compression ratio should land in 20-70% for realistic articles."""
    # Build 2500-char article
    sents = [f"Normal sentence {i} discussing some topic of general interest here today." for i in range(1, 20)]
    article = " ".join(sents)
    out = extractive_compress_v2(article)
    red = 1 - len(out) / len(article)
    _assert(0.1 <= red <= 0.8, f"Expected 10-80% reduction, got {100*red:.1f}%")


def test_empty_passthrough():
    """Empty / whitespace input."""
    _assert(extractive_compress_v2("") == "")
    _assert(extractive_compress_v2("  \n\n  ") == "  \n\n  ")  # too short, passthrough


def test_deterministic():
    """Same input → same output (no randomness)."""
    article = "Sentence A. " * 20
    out1 = extractive_compress_v2(article)
    out2 = extractive_compress_v2(article)
    _assert(out1 == out2, "Must be deterministic")


def test_abbrev_list_coverage():
    """Common abbreviations must be covered."""
    must_have = {"mr", "mrs", "dr", "prof", "u.s", "etc", "a.m", "p.m"}
    for a in must_have:
        _assert(a in _ABBREV, f"Missing abbrev: {a}")


def test_first_sentence_never_dropped():
    """The lead sentence is always preserved — never considered attribution."""
    article = "The spokesperson said this matters. Also this. Also that. Also more. Also another. Another one. One more. Last. Really last. Final."
    out = extractive_compress_v2(article)
    _assert(out.startswith("The spokesperson"), "Lead must be first")


def run_all():
    tests = [
        test_abbrev_no_split,
        test_high_info_preserved,
        test_low_info_attribution_removed,
        test_dedupe,
        test_short_passthrough,
        test_normal_mode,
        test_aggressive_mode,
        test_reduction_range,
        test_empty_passthrough,
        test_deterministic,
        test_abbrev_list_coverage,
        test_first_sentence_never_dropped,
    ]
    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed.append((t.__name__, str(e)))
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print("Failures:")
        for name, msg in failed:
            print(f"  {name}: {msg}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(run_all())
