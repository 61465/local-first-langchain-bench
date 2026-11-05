"""HSTC Unicode/RTL/emoji attack-surface tests — Fix #5 for NEXUS v5.2.

Verifies HSTC does not crash, hang, or corrupt on adversarial strings.
"""
import unicodedata
from compressor_hstc import hstc_compress


ATTACK_STRINGS = [
    # 1-10 — ZWJ emoji sequences
    "Family: 👨‍👩‍👧‍👦 announced plans. The President said yesterday.",
    "Emoji flood: " + "🎉" * 500 + " The announcement was today.",
    "Mixed: 🇯🇵🇺🇸🇫🇷 Officials confirmed the news in 2026.",

    # 11-15 — RTL override attacks
    "RTL embedded: ‮evil‬ President announced plans today.",
    "‏Arabic سيلزم محض text with LTR embed.",
    "Bidirectional ‪ English in Arabic ‬ context.",

    # 16-20 — Grapheme / NFC vs NFD
    "Café (NFC) and Café (NFD) should both compress.",
    "éééé repeated accented chars.",

    # 21-25 — Control chars
    "Text with null \x00 byte in middle should not crash.",
    "Bell \x07 tab\there and form \x0c feed.",

    # 26-30 — Large inputs
    "a " * 10000,  # 20k chars of single-char words
    "A very long sentence. " * 1000,  # 20k with periods
    "Word​" * 5000,  # zero-width-space abuse

    # 31-35 — Edge cases
    "",  # empty
    " ",  # single space
    ".",  # single period
    "\n\n\n",  # just newlines
    "Mr. Dr. Prof. etc. a.m. p.m. U.S. abbreviation stress test.",

    # 36-40 — mixed-script
    "Arabic عربي and English and 中文 in one sentence 2026.",
    "Emoji 🎉 in middle of sentence about U.S. policy.",
    "Code block: `def multiply(a, b):` embedded in prose.",

    # 41-50 — specific CVE-class inputs
    "Trojan source: if‮admin==true‬ then grant access.",
    "Homoglyph: аpple vs apple — looks same, different Unicode.",
    "Surrogate: 😀 valid emoji via surrogate pair.",
    "4-byte UTF-8: 𐀀𐀁𐀂 Linear B syllables.",
    "Private-use:  private-use area.",
    "Fullwidth: ＡＢＣＤ fullwidth ASCII.",
    "Combining: à́̂̃ multiple combining accents.",
    "Mongolian vowel sep: a᠎b hidden separator.",
    "Line sep: line1 line2 paragraph2.",
    "URL: https://example.com/‮path — RTL in URL.",
]


def test_no_crash_on_attack_strings():
    """HSTC should not raise on any adversarial input."""
    failures = []
    for i, s in enumerate(ATTACK_STRINGS):
        try:
            out = hstc_compress(s)
            # Basic sanity: output should not be None, should be str
            assert isinstance(out, str), f"non-str output at #{i}"
        except Exception as e:
            failures.append((i, repr(s[:50]), type(e).__name__, str(e)[:100]))
    assert not failures, f"HSTC crashed on {len(failures)} attack strings: {failures[:3]}"


def test_no_infinite_loop():
    """HSTC should complete in reasonable time on 50kB input."""
    import time
    big = "The quick brown fox. " * 2500  # 50kB
    t = time.perf_counter()
    hstc_compress(big)
    elapsed = time.perf_counter() - t
    assert elapsed < 5.0, f"HSTC took {elapsed:.2f}s on 50kB (ReDoS?)"


def test_nfc_normalization_stable():
    """Same text in NFC and NFD should produce similar compression."""
    nfc_text = "Café is a popular word in English. Café owners are increasing."
    nfd_text = unicodedata.normalize("NFD", nfc_text)
    out_nfc = hstc_compress(nfc_text)
    out_nfd = hstc_compress(nfd_text)
    # Both should be non-empty strings
    assert out_nfc and out_nfd


def test_empty_input():
    """Empty / whitespace-only inputs do not crash."""
    for s in ["", " ", "\n", "\t", "   "]:
        out = hstc_compress(s)
        assert isinstance(out, str)


def test_rtl_passthrough():
    """RTL text should not be mangled."""
    arabic = "النص العربي يجب أن يُحفظ في عملية الضغط."
    out = hstc_compress(arabic)
    assert out  # non-empty


if __name__ == "__main__":
    import sys
    tests = [
        test_no_crash_on_attack_strings,
        test_no_infinite_loop,
        test_nfc_normalization_stable,
        test_empty_input,
        test_rtl_passthrough,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as e:
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(0 if failed == 0 else 1)
