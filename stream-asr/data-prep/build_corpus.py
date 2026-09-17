"""Build the versioned ASR corpus that both training tracks read.

This is data preparation only. Omnilingual and Nemotron each convert it to their own format
(parquet / Lhotse manifests) and train independently.

Per non-English language, in order:
  1. read nemotron_ft/manifests_clean/<code>_{train,dev,test}.jsonl
     (CSV-concatenation and forced-alignment-dump rows are already gone there)
  2. drop rows with empty text
  3. drop excluded sources (EXCLUDED_SOURCES)
  4. move the English rows listed in omni_ft/english_drops/<code>.english.tsv to English
  5. repair text deterministically (textfix.repair: mojibake, perispomeni, soft hyphen, NFC)
  6. remove audio files that appear in more than one split (kept in test > dev > train)
  7. carve dev/test where a language has < MIN_SPLIT_H of it (see carve())
  8. write fixed evaluation subsets: dev_eval (<= DEV_EVAL_UTTS) and test_eval (<= TEST_EVAL_H)

ENGLISH IS ONE LANGUAGE, code "en". en-GH, en-NG, en-UG and en-ZA are merged, and the English
removed from the other languages is added to it instead of being discarded (afrispeech-200 and
the NCHLT English phrases exist nowhere else in the corpus). Each English row records where it
came from in "origin" (the corpus it was filed under), so accent-level analysis stays possible.
Rescued rows that duplicate an English-corpus audio file are dropped as duplicates.

No row is removed for being long. Cutting long audio is a later, alignment-based step.

Carved splits are sampled per source by a hash of the audio path, so they are reproducible, but
the manifests carry no speaker ids: carved dev/test are NOT guaranteed speaker-disjoint from
train. The report marks every carved split so that is never forgotten.

    python build_corpus.py --out /leonardo_scratch/large/userexternal/atsado00/asr_corpus/v1
"""
import argparse
import collections
import glob
import hashlib
import json
import os
from multiprocessing import Pool

from textfix import repair

SRC = "/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/manifests_clean"
DROPS = "/leonardo_scratch/large/userexternal/atsado00/omni_ft/english_drops"

ENGLISH_CODES = ("en-GH", "en-NG", "en-UG", "en-ZA")
ENGLISH = "en"

# Arabic translations of Tamazight speech, not Berber transcripts.
EXCLUDED_SOURCES = {"ber-MA": {"Tamazight-Speech-to-Arabic-Text"}}

MIN_SPLIT_H = 0.5          # below this a dev/test split is topped up
DEV_FROM_DEV_MIN_H = 4.0   # a missing test is taken from dev only if dev has at least this
DEV_EVAL_UTTS = 600
TEST_EVAL_H = 2.0
SPLITS = ("train", "dev", "test")
RESCUED_DIR = "_english_rescued"


def source_of(path):
    parts = path.split("/speech_out/")[-1].split("/")
    return parts[2] if len(parts) > 2 else "?"


def key(path):
    return hashlib.sha1(path.encode("utf-8")).hexdigest()


def hours(rows):
    return sum(r["duration"] for r in rows) / 3600


def new_report(code):
    return {"language": code, "dropped": collections.Counter(), "repairs": collections.Counter(),
            "repaired_rows": 0, "carved": {}, "input": {}, "output": {}}


def load_drops(code):
    f = os.path.join(DROPS, f"{code}.english.tsv")
    if not os.path.exists(f):
        return None
    with open(f, encoding="utf-8") as fh:
        return {l.split("\t", 1)[0] for l in fh if l.strip() and not l.startswith("#")}


def read_rows(code, split, report, lang, drops=frozenset(), excluded=frozenset()):
    """Returns (kept rows, english rows moved out)."""
    kept, english = [], []
    f = os.path.join(SRC, f"{code}_{split}.jsonl")
    n_in = 0
    if os.path.exists(f):
        with open(f, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    report["dropped"]["bad_json"] += 1
                    continue
                n_in += 1
                path, text = r.get("audio_filepath"), (r.get("text") or "").strip()
                if not path or not text:
                    report["dropped"]["empty_text"] += 1
                    continue
                src = source_of(path)
                if src in excluded:
                    report["dropped"][f"excluded_source:{src}"] += 1
                    continue
                fixed, kinds = repair(text)
                row = {"audio_filepath": path, "duration": float(r.get("duration") or 0),
                       "text": fixed, "lang": lang, "source": src, "origin": code}
                if kinds:
                    row["text_raw"] = text
                    report["repaired_rows"] += 1
                    report["repairs"].update(kinds)
                if path in drops:
                    row["lang"] = ENGLISH
                    english.append(row)
                    report["dropped"]["english_moved_to_en"] += 1
                    continue
                kept.append(row)
    report["input"][split] = report["input"].get(split, 0) + n_in
    return kept, english


def take_hours(pool_rows, target_h):
    """Deterministic, source-stratified sample of about target_h from pool_rows."""
    by_src = collections.defaultdict(list)
    for r in pool_rows:
        by_src[r["source"]].append(r)
    total = hours(pool_rows)
    chosen = []
    for rows in by_src.values():
        quota = target_h * hours(rows) / total if total else 0
        got = 0.0
        for r in sorted(rows, key=lambda r: key(r["audio_filepath"])):
            if got >= quota:
                break
            chosen.append(r)
            got += r["duration"] / 3600
    return chosen


def carve(split_rows, report):
    train_h = hours(split_rows["train"])
    target = min(2.0, max(MIN_SPLIT_H, 0.05 * train_h))
    for split in ("dev", "test"):
        have = hours(split_rows[split])
        if have >= MIN_SPLIT_H:
            continue
        if split == "test" and hours(split_rows["dev"]) >= DEV_FROM_DEV_MIN_H:
            donor = "dev"
            moved = take_hours(split_rows["dev"], hours(split_rows["dev"]) / 2)
        else:
            donor = "train"
            moved = take_hours(split_rows["train"], target - have)
        ids = {r["audio_filepath"] for r in moved}
        split_rows[donor] = [r for r in split_rows[donor] if r["audio_filepath"] not in ids]
        split_rows[split].extend(moved)
        report["carved"][split] = {"from": donor, "utts": len(moved), "hours": round(hours(moved), 3),
                                   "speaker_disjoint": False}


def finalize(code, split_rows, report, out_root):
    """Dedupe across splits, carve missing splits, write splits + eval subsets + report."""
    seen = set()
    for split in ("test", "dev", "train"):
        kept = []
        for r in split_rows[split]:
            if r["audio_filepath"] in seen:
                report["dropped"][f"duplicate_audio:{split}"] += 1
                continue
            seen.add(r["audio_filepath"])
            kept.append(r)
        split_rows[split] = kept

    if split_rows["train"]:
        carve(split_rows, report)

    out = os.path.join(out_root, code)
    os.makedirs(out, exist_ok=True)
    for split in SPLITS:
        with open(os.path.join(out, f"{split}.jsonl"), "w", encoding="utf-8") as fh:
            for r in split_rows[split]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["output"][split] = {"utts": len(split_rows[split]), "hours": round(hours(split_rows[split]), 3)}

    dev_eval = sorted(split_rows["dev"], key=lambda r: key(r["audio_filepath"]))[:DEV_EVAL_UTTS]
    test_eval, got = [], 0.0
    for r in sorted(split_rows["test"], key=lambda r: key(r["audio_filepath"])):
        if got >= TEST_EVAL_H:
            break
        test_eval.append(r)
        got += r["duration"] / 3600
    for name, rows in (("dev_eval", dev_eval), ("test_eval", test_eval)):
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["output"][name] = {"utts": len(rows), "hours": round(hours(rows), 3)}

    report["dropped"] = dict(report["dropped"])
    report["repairs"] = dict(report["repairs"])
    with open(os.path.join(out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    return report


def build_language(args):
    code, out_root = args
    report = new_report(code)
    drops = load_drops(code)
    report["english_filter_applied"] = drops is not None
    excluded = EXCLUDED_SOURCES.get(code, set())

    split_rows = {}
    rescued_dir = os.path.join(out_root, RESCUED_DIR)
    os.makedirs(rescued_dir, exist_ok=True)
    for split in SPLITS:
        kept, english = read_rows(code, split, report, lang=code, drops=drops or set(), excluded=excluded)
        split_rows[split] = kept
        with open(os.path.join(rescued_dir, f"{code}_{split}.jsonl"), "w", encoding="utf-8") as fh:
            for r in english:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return finalize(code, split_rows, report, out_root)


def build_english(out_root):
    report = new_report(ENGLISH)
    report["english_filter_applied"] = True
    split_rows = {s: [] for s in SPLITS}
    seen = set()
    origin_h = collections.Counter()

    def add(rows, split):
        for r in rows:
            if r["audio_filepath"] in seen:
                report["dropped"][f"duplicate_audio:{split}"] += 1
                continue
            seen.add(r["audio_filepath"])
            split_rows[split].append(r)
            origin_h[r["origin"]] += r["duration"] / 3600

    # English corpora first, so a rescued row that repeats one of their files is the duplicate.
    for code in ENGLISH_CODES:
        for split in SPLITS:
            kept, _ = read_rows(code, split, report, lang=ENGLISH)
            add(kept, split)
    for f in sorted(glob.glob(os.path.join(out_root, RESCUED_DIR, "*.jsonl"))):
        split = os.path.basename(f)[:-6].rsplit("_", 1)[1]
        with open(f, encoding="utf-8") as fh:
            add((json.loads(l) for l in fh), split)

    report["hours_by_origin"] = {k: round(v, 2) for k, v in origin_h.most_common()}
    return finalize(ENGLISH, split_rows, report, out_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    codes = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(f"{SRC}/*.jsonl")})
    codes = [c for c in codes if c not in ENGLISH_CODES]
    os.makedirs(a.out, exist_ok=True)
    with Pool(a.workers) as pool:
        reports = pool.map(build_language, [(c, a.out) for c in codes])
    reports.append(build_english(a.out))

    print(f"{'lang':8}{'train h':>9}{'dev h':>7}{'test h':>8}{'eng moved':>10}{'repaired':>10}  carved")
    for r in sorted(reports, key=lambda r: r["language"]):
        o = r["output"]
        carved = "; ".join(f"{s}<-{c['from']} {c['hours']}h" for s, c in r["carved"].items())
        print(f"{r['language']:8}{o['train']['hours']:>9.1f}{o['dev']['hours']:>7.1f}{o['test']['hours']:>8.1f}"
              f"{r['dropped'].get('english_moved_to_en', 0):>10,}{r['repaired_rows']:>10,}  {carved}"
              f"{'' if r['english_filter_applied'] else '  [no english filter run]'}")
    en = next(r for r in reports if r["language"] == ENGLISH)
    print("\nenglish hours by origin:", en["hours_by_origin"])
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=1, ensure_ascii=False, default=dict)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
