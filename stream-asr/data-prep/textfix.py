"""Deterministic transcript repairs that recover the intended text exactly (no guessing).

MOJIBAKE. Some sources stored UTF-8 bytes that were later decoded as Windows-1252, so Kikuyu
"ĩ" (bytes C4 A9) became "Ä©" and "ũ" (C5 A9) became "Å©". Re-encoding
such a run as cp1252 and decoding it as UTF-8 returns the original character exactly. A run is
only replaced when that round trip succeeds and yields a single character, so correct text --
including a genuine "Ä" or "©" -- is left alone.

COMBINING PERISPOMENI. U+0342 (a Greek accent) was typed as a tilde over i/u; it is replaced by
U+0303 COMBINING TILDE, and NFC then composes i + U+0303 into the single letter U+0129.

SOFT HYPHEN (U+00AD) is invisible formatting and is removed.
"""
import re
import unicodedata

# Characters that cp1252 produces for UTF-8 continuation bytes 0x80-0xBF.
_CONT = "".join(bytes([b]).decode("cp1252", errors="ignore") for b in range(0x80, 0xC0))
# Characters that cp1252 produces for UTF-8 lead bytes 0xC2-0xF4.
_LEAD = "".join(bytes([b]).decode("cp1252", errors="ignore") for b in range(0xC2, 0xF5))
_MOJIBAKE = re.compile(f"[{re.escape(_LEAD)}][{re.escape(_CONT)}]{{1,3}}")

_PERISPOMENI = "͂"
_TILDE = "̃"
_SOFT_HYPHEN = "­"


def _undo(m):
    s = m.group(0)
    try:
        fixed = s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
    return fixed if len(fixed) == 1 else s


def repair(text):
    """Return (repaired_text, list of repair kinds applied)."""
    kinds = []
    t = _MOJIBAKE.sub(_undo, text)
    if t != text:
        kinds.append("mojibake")
    if _PERISPOMENI in t:
        t = t.replace(_PERISPOMENI, _TILDE)
        kinds.append("perispomeni")
    if _SOFT_HYPHEN in t:
        t = t.replace(_SOFT_HYPHEN, "")
        kinds.append("soft_hyphen")
    n = unicodedata.normalize("NFC", t)
    if n != t:
        kinds.append("nfc")
    return n, kinds


if __name__ == "__main__":
    tests = [
        "MaÅ©ndÅ© marÄ©a mangenagia haha nÄ©kuona",
        "Å¨ndÅ© Å©yÅ©",
        "Café © 2020 Ä real",
        "tondũ nĩ",
        "ki͂ndu",
        "abc­def",
        "Ọ mọ náà ṣiṣẹ́",
    ]
    for s in tests:
        print(ascii(s), "->", repair(s))
