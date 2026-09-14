"""Re-pair transcripts with their own audio where filenames collide across datasets.

The August manifest builder indexed audio by filename stem and kept the first file it met, so
when two datasets in one language both contain e.g. `adamawa_001216.wav`, every transcript with
that id was paired with whichever file happened to be indexed first. For fula that put 196k
transcripts (315 h) on the wrong recording.

Each source CSV row carries `dataset_name` and `duration_sec`. A row is resolved to exactly one
file, in this order:
  1. the only file with that stem;
  2. among same-stem files, the one whose folder matches the dataset name;
  3. among same-stem files, the one whose actual WAV duration equals duration_sec (+-0.02 s).
A row that none of these settles is reported as unresolved -- never guessed.

Rows describing the same recording (same resolved file, same text) are merged into one: several
Hugging Face mirrors of one corpus list identical recordings, and keeping each copy would train
on the same utterance several times.

    python resolve_audio.py --lang fula --code ff-SN --out <dir>          # analysis + manifests
"""
import argparse
import collections
import csv
import json
import os
import re
import sys
import wave

csv.field_size_limit(sys.maxsize)

SPEECH_OUT = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
DUR_TOL = 0.02


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def wav_duration(path, cache):
    if path not in cache:
        try:
            with wave.open(path, "rb") as w:
                cache[path] = w.getnframes() / float(w.getframerate())
        except Exception:
            cache[path] = None
    return cache[path]


def index_audio(lang):
    index = collections.defaultdict(list)
    root = os.path.join(SPEECH_OUT, lang, "audio")
    for dirpath, _dirs, files in os.walk(root):
        folder = os.path.relpath(dirpath, root).split(os.sep)[0]
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in AUDIO_EXTS:
                index[stem].append((folder, os.path.join(dirpath, f)))
    return index


def folder_match(dataset, candidates):
    d = norm(dataset)
    exact = [c for c in candidates if norm(c[0]) == d]
    if len(exact) == 1:
        return exact[0][1]
    return None


def resolve(row, index, dur_cache, stats):
    cands = index.get(row["audio_id"], [])
    if not cands:
        stats["no_audio"] += 1
        return None
    if len(cands) == 1:
        stats["unique_stem"] += 1
        return cands[0][1]
    path = folder_match(row.get("dataset_name") or "", cands)
    if path:
        stats["folder_match"] += 1
        return path
    try:
        want = float(row.get("duration_sec") or 0)
    except ValueError:
        want = 0.0
    hits = [p for _, p in cands if (d := wav_duration(p, dur_cache)) is not None and abs(d - want) <= DUR_TOL]
    if len(hits) == 1:
        stats["duration_match"] += 1
        return hits[0]
    stats["unresolved_ambiguous" if len(hits) != 1 else "unresolved"] += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, help="speech_out folder, e.g. fula")
    ap.add_argument("--code", required=True, help="corpus code, e.g. ff-SN")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    index = index_audio(a.lang)
    colliding = {s for s, c in index.items() if len(c) > 1}
    stats = collections.Counter()
    dur_cache = {}
    seen = {}
    rows_out = collections.defaultdict(list)
    unresolved_examples = []

    with open(os.path.join(SPEECH_OUT, a.lang, f"speech_transcribed_{a.lang}.csv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            text = (row.get("transcript") or "").strip()
            if not text:
                stats["empty_text"] += 1
                continue
            stats["rows_with_text"] += 1
            path = resolve(row, index, dur_cache, stats)
            if path is None:
                if len(unresolved_examples) < 5 and row["audio_id"] in index:
                    unresolved_examples.append((row["audio_id"], row.get("dataset_name"), row.get("duration_sec"),
                                                [(f, wav_duration(p, dur_cache)) for f, p in index[row["audio_id"]]]))
                continue
            k = (path, text)
            if k in seen:
                stats["same_recording_same_text_merged"] += 1
                continue
            seen[k] = True
            try:
                dur = float(row.get("duration_sec") or 0)
            except ValueError:
                dur = 0.0
            split = row.get("split") if row.get("split") in ("train", "dev", "test") else "train"
            rows_out[split].append({"audio_filepath": path, "duration": dur, "text": text,
                                    "target_lang": a.code, "dataset_name": row.get("dataset_name"),
                                    "speaker_id": row.get("speaker_id"), "resolved_from_collision": row["audio_id"] in colliding})

    # A file that still carries two different texts is a genuine conflict in the source data.
    by_path = collections.defaultdict(set)
    for split_rows in rows_out.values():
        for r in split_rows:
            by_path[r["audio_filepath"]].add(r["text"])
    stats["files_with_conflicting_texts"] = sum(1 for t in by_path.values() if len(t) > 1)

    os.makedirs(a.out, exist_ok=True)
    for split, split_rows in rows_out.items():
        with open(os.path.join(a.out, f"{a.code}_{split}.jsonl"), "w", encoding="utf-8") as fh:
            for r in split_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = {s: round(sum(r["duration"] for r in v) / 3600, 1) for s, v in rows_out.items()}
    report = {"lang": a.lang, "code": a.code, "stats": dict(stats), "hours": hours,
              "colliding_stems": len(colliding), "unresolved_examples": unresolved_examples}
    with open(os.path.join(a.out, f"{a.code}_resolve_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
