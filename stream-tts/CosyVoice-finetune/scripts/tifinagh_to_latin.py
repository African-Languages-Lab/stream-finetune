"""Transliterate Neo-Tifinagh (IRCAM) into the Latin Berber orthography the model expects.

The Berber checkpoint was fine-tuned on Latin-script Tamazight -- its working reference
prompt reads "umrar yellan di lbir. Uccen yuder, inisi yuli-d. Yenṭeq wuccen ɣer yinisi".
Several Berber corpora are transcribed in Tifinagh instead, which the model has never seen;
feeding it one produces a transcript that does not match its audio, and CosyVoice3 collapses
when that happens.

The mapping below is the standard IRCAM Tifinagh <-> Latin Berber correspondence, including
the labialisation mark (ⵯ) which attaches to the preceding consonant.
"""
import re
import unicodedata

# IRCAM Neo-Tifinagh -> Latin Berber (Kabyle/Tachelhit convention, matching the
# emphatic and pharyngeal letters used in the model's training transcripts)
TIFINAGH_LATIN = {
    "ⴰ": "a", "ⴱ": "b", "ⴳ": "g", "ⴷ": "d", "ⴹ": "ḍ", "ⴻ": "e",
    "ⴼ": "f", "ⴽ": "k", "ⵀ": "h", "ⵃ": "ḥ", "ⵄ": "ɛ", "ⵅ": "x",
    "ⵇ": "q", "ⵉ": "i", "ⵊ": "j", "ⵍ": "l", "ⵎ": "m", "ⵏ": "n",
    "ⵓ": "u", "ⵔ": "r", "ⵕ": "ṛ", "ⵖ": "ɣ", "ⵙ": "s", "ⵚ": "ṣ",
    "ⵛ": "c", "ⵜ": "t", "ⵟ": "ṭ", "ⵡ": "w", "ⵢ": "y", "ⵣ": "z",
    "ⵥ": "ẓ", "ⵯ": "ʷ",
    # less common / regional letters
    "ⴲ": "b", "ⴴ": "g", "ⴵ": "j", "ⴶ": "g", "ⴸ": "d", "ⴺ": "ḍ",
    "ⴽⵯ": "kʷ", "ⴳⵯ": "gʷ", "ⵁ": "h", "ⵂ": "h", "ⵈ": "q",
    "ⵎⵯ": "mʷ", "ⵗ": "ɣ", "ⵆ": "x", "ⵌ": "j", "ⵐ": "n", "ⵑ": "n",
    "ⵞ": "č", "ⵝ": "t", "ⵟⵯ": "ṭʷ", "ⵠ": "v", "ⵡⵯ": "wʷ",
}

_TIFI_RANGE = re.compile(r"[ⴰ-⵿]")


def is_tifinagh(text, threshold=0.5):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    n = sum(1 for c in letters if "TIFINAGH" in unicodedata.name(c, ""))
    return n / len(letters) >= threshold


def transliterate(text):
    """Tifinagh -> Latin. Non-Tifinagh characters (punctuation, spaces, Latin) pass through."""
    out = []
    i = 0
    while i < len(text):
        # try the two-character sequences first (labialised consonants)
        if i + 1 < len(text) and text[i:i + 2] in TIFINAGH_LATIN:
            out.append(TIFINAGH_LATIN[text[i:i + 2]])
            i += 2
            continue
        ch = text[i]
        if ch in TIFINAGH_LATIN:
            out.append(TIFINAGH_LATIN[ch])
        elif _TIFI_RANGE.match(ch):
            out.append("")          # unmapped Tifinagh letter: drop rather than emit a stray glyph
        else:
            out.append(ch)
        i += 1
    # tidy the spacing that Tifinagh punctuation conventions leave behind
    s = "".join(out)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def residual_tifinagh(text):
    return sum(1 for c in text if _TIFI_RANGE.match(c))


if __name__ == "__main__":
    samples = [
        "ⵢⵉⵡⵉ ⴷ ⵢⵉⵣⵎ; ⴰⵍⵍⵉⵖ ⵜⵎⵎⵓⴷⴷⴰ ⵜⵎⵖⴰⵔⵜ ⵏⵏⵙ ⵙ ⴷⴰⵔ ⴰⵢⵜ ⴷⴰⵔⵙ",
        "ⵉⵎⵎⵓⵜ ⵡⴰⵡⵜⵉⵍ ⵙ ⵜⴰⴹⵚⴰ",
    ]
    for s in samples:
        print(f"{s}\n  -> {transliterate(s)}\n")
