"""Convert the existing NeMo manifests into the fairseq2 tsv/wrd form Omnilingual reads.

The corpus was already walked once when the NeMo manifests were built, and those JSONL
files carry everything needed (audio path, duration, transcript, language). Re-walking
speech_out would risk the two pipelines disagreeing about what is in the training set, so
this converts rather than rebuilds -- the NeMo manifests stay the single source of truth.

OUTPUT, per split, in the layout ManifestStorage expects:

    <out>/train.tsv     line 1 is the audio root, then "<relative path>\\t<frames>"
    <out>/train.wrd     one transcript per line, aligned by line number
    <out>/train.lang    one language token per line, same alignment

The .lang file is not part of the stock recipe. Omnilingual conditions its decoder on a
language token, and one combined model over 38 languages is the entire point here, so the
token has to travel with each example. Emitting it as a parallel file keeps the tsv/wrd
pair exactly as the stock reader expects while making the conditioning available.

LENGTH IS IN FRAMES, not seconds. The reader treats column two as a sample count; writing
seconds there would make every utterance look like a fraction of a second and the batcher
would mis-bucket everything. duration * 16000 is the conversion.

Rows whose language has no Omnilingual token are skipped and counted, not guessed at --
see lang_map.UNSUPPORTED.

ENGLISH IS FILTERED OUT OF THE OTHER LANGUAGES. Whole English datasets were filed under
non-English languages (ghana-english-asr-2700hrs inside ewe and twi, afrispeech-200 inside
ten of them) and English utterances are scattered through the NCHLT aux sets. lid_filter.py
classifies every transcript and writes the English rows to <dir>/<lang>.english.tsv; pass
that directory as --english-drops and they are skipped here. Two scripts rather than one so
the deletion is auditable -- you can read exactly what was removed without re-running the
model. Without the flag nothing is filtered, which is why the flag is loud rather than a
silent default.

    python build_manifests.py --out /path/to/manifests            # all languages
    python build_manifests.py --out ... --langs ha-NG,ig-NG,yo-NG # a subset
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lang_map import LANG_MAP, NEW_LANG_CODES, UNSUPPORTED, token_for

NEMO = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests"
SAMPLE_RATE = 16000


def load_english_drops(drops_dir, code):
    """Audio paths lid_filter.py judged English for this language."""
    if not drops_dir:
        return set()
    f = os.path.join(drops_dir, f"{code}.english.tsv")
    if not os.path.exists(f):
        return set()
    paths = set()
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if parts and parts[0]:
                paths.add(parts[0])
    return paths


def convert(split, langs, out_dir, audio_root, min_sec, max_sec, drops_dir=None):
    rows, skipped = [], Counter()
    for code in langs:
        src = f"{NEMO}/{code}_{split}.jsonl"
        if not os.path.exists(src):
            skipped["no manifest"] += 1
            continue
        tok = token_for(code)
        if tok is None:
            skipped[f"unsupported language {code}"] += 1
            continue
        english = load_english_drops(drops_dir, code)
        with open(src, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    skipped["bad json"] += 1
                    continue
                path, text = r.get("audio_filepath"), (r.get("text") or "").strip()
                dur = float(r.get("duration") or 0)
                if not path or not text:
                    skipped["missing path or text"] += 1
                    continue
                if path in english:
                    skipped["English inside a non-English language"] += 1
                    continue
                if not (min_sec <= dur <= max_sec):
                    skipped["duration out of range"] += 1
                    continue
                rel = os.path.relpath(path, audio_root)
                if rel.startswith(".."):
                    skipped["outside audio root"] += 1
                    continue
                rows.append((rel, int(dur * SAMPLE_RATE), text, tok))
    return rows, skipped


def write(rows, out_dir, split, audio_root):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{split}.tsv", "w", encoding="utf-8") as tsv, \
         open(f"{out_dir}/{split}.wrd", "w", encoding="utf-8") as wrd, \
         open(f"{out_dir}/{split}.lang", "w", encoding="utf-8") as lng:
        tsv.write(audio_root + "\n")
        for rel, frames, text, tok in rows:
            tsv.write(f"{rel}\t{frames}\n")
            wrd.write(text.replace("\n", " ") + "\n")
            lng.write(tok + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio-root",
                    default="/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out")
    ap.add_argument("--langs", default="", help="comma-separated corpus codes; default all mapped")
    ap.add_argument("--splits", default="train,dev,test")
    ap.add_argument("--min-sec", type=float, default=0.5)
    ap.add_argument("--max-sec", type=float, default=30.0,
                    help="the limited-length models cap around 30 s of audio")
    ap.add_argument("--english-drops", default="",
                    help="directory of <lang>.english.tsv from lid_filter.py; rows listed "
                         "there are dropped. Omit and no English filtering happens.")
    a = ap.parse_args()

    langs = [c.strip() for c in a.langs.split(",") if c.strip()] or sorted({**LANG_MAP, **NEW_LANG_CODES})
    unmapped = [c for c in langs if c in UNSUPPORTED]
    if unmapped:
        print("skipping, no language token yet: " +
              ", ".join(f"{c} ({UNSUPPORTED[c][0]})" for c in unmapped))

    if not a.english_drops:
        print("WARNING: --english-drops not given, so English planted in the other "
              "languages stays in (ghana-english-asr-2700hrs, afrispeech-200, the NCHLT "
              "aux sets). Run lid_filter.py first.")

    total = 0
    for split in [s.strip() for s in a.splits.split(",") if s.strip()]:
        rows, skipped = convert(split, langs, a.out, a.audio_root, a.min_sec, a.max_sec,
                                a.english_drops or None)
        write(rows, a.out, split, a.audio_root)
        hours = sum(r[1] for r in rows) / SAMPLE_RATE / 3600
        langs_seen = len({r[3] for r in rows})
        print(f"  {split:6}{len(rows):9,} utterances  {hours:8.1f} h  {langs_seen} languages"
              f"  -> {a.out}/{split}.tsv")
        for reason, n in skipped.most_common(5):
            print(f"          skipped {n:>8,}  {reason}")
        total += len(rows)
    print(f"\n{total:,} utterances written to {a.out}")


if __name__ == "__main__":
    main()
