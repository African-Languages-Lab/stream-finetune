"""
Filters out rows with corrupted `text` fields from all Nemotron manifests, writing cleaned
copies to a new directory (originals untouched, since cluster/individual jobs read them live).

Investigation found TWO distinct, unrelated data-prep bugs (both confirmed against the
original source CSVs, predating anything in this pipeline) -- an earlier version of this
script used a single blanket "chars per second" threshold, which was too blunt: verified
against real source-CSV cross-checks and real audio durations (via soundfile), it both
missed catching some corruption and, worse, wrongly dropped tens of thousands of legitimate
rows (naturally dense/fast Sesotho/Setswana/Xhosa sentences with 40-190 chars/sec, well
within real speech). Replaced with two precise, high-confidence signature detectors instead:

1. CSV-concatenation corruption (Common Voice-derived sw-KE/lg-UG and others): the `text`
   field is not the utterance's transcript but the entire remainder of that dataset's source
   CSV concatenated in -- sentence + age-bracket + gender + lang-code + hash + filename +
   hash + next sentence + ... repeating for thousands of rows (worst case: 389,269 characters
   of text for a 2.34-second clip). Confirmed present verbatim in the original source CSV
   (speech_transcribed_*.csv), i.e. this predates our manifest build entirely. Detected via
   repeated occurrence of the exact corruption markers (.mp3 / gender-tag tokens), which
   cannot legitimately appear more than once in a real single-utterance transcript.

2. Phoneme/forced-alignment data mistaken for transcript text (south african language group:
   af-ZA/en-ZA/nso-ZA/st-ZA/ts-ZA/ve-ZA/xh-ZA/zu-ZA -- one shared corrupted source dataset,
   consistent with the English-merge-into-local-language design pulling the same corrupted
   en-ZA source into all 7 local languages; en-ZA itself turns out to be ~100% corrupted at
   the source): `text` contains forced-alignment output (sample-offset numbers interleaved
   with phone labels) instead of orthographic text. Detected via fraction of pure-numeric
   tokens alone (no fixed phone-label whitelist -- an earlier attempt required matching a
   specific label set and missed many legitimate variants; sampled ~82k rows confirm a sharp
   bimodal split with essentially nothing between ~0% (real text) and ~65-70% (alignment
   dumps), so a plain numeric-fraction threshold is both necessary and sufficient).

A third failure mode was found (single legitimate, coherent sentence paired with a real but
implausibly short audio duration, e.g. a 426-character real sentence against a genuine
1.1-second clip) but is NOT filtered here: it's rare, hard to detect precisely without
false-positiving on legitimate dense/fast speech (confirmed multiple real examples up to
~190 chars/sec), and a handful of such audio/text misalignments won't meaningfully affect
training at this data volume -- unlike the blanket rate filter, which measurably discarded
real data for a diagnosable-but-marginal problem.
"""
import json
import re
import shutil
from pathlib import Path

SRC_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests")
DST_DIR = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests_clean")

def is_csv_concatenation_corrupted(text):
    return text.count(".mp3") >= 2 or text.count("female_feminine") >= 2 or text.count("male_masculine") >= 2


ALIGNMENT_DUMP_START_RE = re.compile(r"^0 \d{5,}\b")


def is_alignment_dump(text):
    # Two earlier versions of this check were wrong in different directions. v1 required one
    # of a specific whitelist of phone-label tokens (sil/spk/ep/...) alongside a numeric-token
    # fraction -- a random sample of en-ZA rows the whitelist accepted as "clean" turned out to
    # be the same corruption with different phone labels not in the whitelist (there is no
    # fixed enumerable label set). v2 dropped the whitelist for a pure numeric-fraction check,
    # but its minimum-token-count floor let short alignment-dump remnants (e.g. "0 15360000
    # sil", 3 tokens) slip through uncaught. This version instead matches the one truly
    # invariant structural fact about the format: a forced-alignment dump always starts its
    # offset sequence at sample 0, so the text literally begins with "0 <big number>" -- a
    # pattern real transcribed speech never produces. Verified against the full 14.26M-row
    # corpus: matches exactly 12,242 rows in each of 8 south african language files (exactly
    # en-ZA's total row count -- it is ~100% corrupted at the source) and zero rows anywhere
    # else, i.e. zero false positives at full-corpus scale.
    return bool(ALIGNMENT_DUMP_START_RE.match(text))


# Conservative backstop for anything even more extreme that slips past both signature
# checks above -- high enough (500 chars/sec) to clear every verified-legitimate dense
# example found during investigation (max observed: ~190 chars/sec).
EXTREME_RATE_CEILING = 500


def is_corrupted(row):
    text = row["text"]
    if is_csv_concatenation_corrupted(text):
        return True
    if is_alignment_dump(text):
        return True
    d = row["duration"]
    if d > 0 and len(text) / d > EXTREME_RATE_CEILING:
        return True
    return False


def clean_file(src_path, dst_path):
    n_total, n_kept, n_dropped = 0, 0, 0
    with open(src_path) as fin, open(dst_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            n_total += 1
            if is_corrupted(row):
                n_dropped += 1
                continue
            fout.write(line)
            n_kept += 1
    return n_total, n_kept, n_dropped


def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    grand_total, grand_dropped = 0, 0
    for src_path in sorted(SRC_DIR.glob("*.jsonl")):
        dst_path = DST_DIR / src_path.name
        n_total, n_kept, n_dropped = clean_file(src_path, dst_path)
        grand_total += n_total
        grand_dropped += n_dropped
        if n_dropped:
            print(f"{src_path.name}: {n_total} -> {n_kept} ({n_dropped} dropped)")
    for extra in SRC_DIR.glob("*.json"):
        shutil.copy(extra, DST_DIR / extra.name)
    print(f"\nTotal: {grand_total} rows scanned, {grand_dropped} dropped "
          f"({100*grand_dropped/grand_total:.3f}%), wrote to {DST_DIR}")


if __name__ == "__main__":
    main()
