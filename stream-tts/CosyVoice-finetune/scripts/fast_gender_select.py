"""Pick a male and a female reference clip per language, quickly.

The earlier sweep analysed clips blindly and took hours: pitch tracking and MFCCs over
thousands of files, most of which were the wrong gender or had an unusable transcript.

This inverts the order. The corpus CSV already carries a gender column and the transcript,
and audio_id + dataset_name resolve straight to a path, so candidates can be filtered down
to "right gender, usable transcript, sensible duration" before any audio is opened. Only
those few are analysed acoustically.

Languages whose corpus has no gender labels (hausa, ewe, fula, malagasy, chichewa) fall back
to classifying by median F0.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")
import librosa  # noqa: E402
import numpy as np  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
SO = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"

CORPUS = {
    "hausa": "hausa", "twi": "twi", "igbo": "igbo", "ewe": "ewe", "berber": "berber",
    "umbundu": "umbundu", "amharic": "amharic", "arabic": "arabic", "fula": "fula",
    "luganda": "luganda", "lingala": "lingala", "malagasy": "malagasy", "sepedi": "sepedi",
    "chichewa": "chichewa", "oromo": "oromo", "somali": "somali", "sesotho": "sesotho",
    "swahili": "swahili", "tigrinya": "tigrinya", "tswana": "tswana", "tsonga": "tsonga",
    "venda": "venda", "xhosa": "xhosa", "zulu": "zulu",
}
SCRIPT = {"amharic": "ETHIOPIC", "tigrinya": "ETHIOPIC", "arabic": "ARABIC"}

BAD_DS = re.compile(r"synthetic|xtts|yourtts|vits|l2_|learner|_l2", re.I)
GOOD_DS = re.compile(r"bible|fleurs|lwazi|anv[-_]|afrivoice|african_voices|naija|ugspeech|[-_]tts", re.I)
FIELD_DS = re.compile(r"waxal|common.?voice|radio|broadcast|conversation|speech_dataset", re.I)

MAX_ANALYSE = 45          # per gender
LO, HI = 4.5, 15.0


def script_ok(t, want):
    L = [c for c in t if c.isalpha()]
    return bool(L) and sum(1 for c in L if want in unicodedata.name(c, "")) / len(L) >= 0.65


def transcript_ok(t, lang):
    if not t:
        return False
    t = t.strip()
    if len(t) < 20 or len(t.split()) < 4:
        return False
    if re.fullmatch(r"[0-9a-fA-F\s]+", t):
        return False
    if any(len(w) >= 24 and re.fullmatch(r"[0-9a-fA-F]+", w) for w in t.split()):
        return False
    if sum(c.isdigit() for c in t) > len(t) * 0.3:
        return False
    if not script_ok(t, SCRIPT.get(lang, "LATIN")):
        return False
    toks = [w.strip(".,;:()") for w in t.split()]
    if t.count(",") >= len(toks) * 0.35:
        return False
    stems = {}
    for w in toks:
        if len(w) >= 4:
            stems[w[:4]] = stems.get(w[:4], 0) + 1
    return max(stems.values(), default=0) < max(3, len(toks) * 0.4)


def analyse(path):
    try:
        y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception:
        return None
    n = len(y)
    dur = n / sr
    if not (LO <= dur <= HI):
        return None
    peak = float(np.max(np.abs(y)))
    if not (0.12 <= peak <= 0.999):
        return None
    if float(np.mean(np.abs(y) > 0.99)) > 0.0008:
        return None

    r = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    db = 20 * np.log10(np.maximum(r, 1e-9))
    noise = float(np.percentile(db, 10))
    if noise > -52:
        return None
    silence = float(np.mean(r < np.percentile(r, 95) * 0.02))
    if silence > 0.32:
        return None

    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256)
    e = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][:len(f0)]
    v = (e > np.percentile(e, 55)) & (f0 > 63) & (f0 < 380)
    if v.sum() < 20:
        return None
    med = float(np.median(f0[v]))
    iqr = float(np.percentile(f0[v], 75) - np.percentile(f0[v], 25))
    if iqr > 80:
        return None

    # timbre consistency -> catches two speakers / background conversation
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    per = max(1, int(sr / 256))
    segs = np.array([mf[:, i:i + per].mean(axis=1)
                     for i in range(0, mf.shape[1] - per // 2, per)])
    segs = segs[~np.isnan(segs).any(axis=1)]
    if len(segs) < 4:
        return None
    segs = (segs - segs.mean(0)) / (segs.std(0) + 1e-9)
    c = [segs[0], segs[-1]]
    lab = None
    for _ in range(10):
        d = np.stack([np.linalg.norm(segs - ci, axis=1) for ci in c])
        lab = d.argmin(0)
        if lab.min() == lab.max():
            break
        c = [segs[lab == k].mean(0) for k in (0, 1)]
    if lab is not None and lab.min() != lab.max():
        between = np.linalg.norm(c[0] - c[1])
        within = np.mean([np.linalg.norm(segs[lab == k] - c[k], axis=1).mean() for k in (0, 1)])
        split = float(between / (within + 1e-9))
    else:
        split = 0.0
    if split > 2.6:
        return None

    return dict(sr=int(sr), dur=round(dur, 2), peak=round(peak, 3),
                noise_dbfs=round(noise, 1), silence=round(silence, 3),
                f0_med=round(med, 1), f0_iqr=round(iqr, 1), speaker_split=round(split, 2),
                detected_gender="male" if med < 165 else "female")


def score(m, pen):
    s = -m["noise_dbfs"] * 0.7
    s -= abs(m["dur"] - 8.5) * 3.0
    s -= m["silence"] * 110
    s -= m["speaker_split"] * 22
    s -= pen
    return s


def gather(lang):
    """Rows from the CSV that are worth opening, grouped by labelled gender."""
    corpus = CORPUS[lang]
    p = os.path.join(SO, corpus, f"speech_transcribed_{corpus}.csv")
    by = {"male": [], "female": [], "unknown": []}
    if not os.path.exists(p):
        return by
    with open(p, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ds = row.get("dataset_name", "")
            if not ds or BAD_DS.search(ds):
                continue
            t = (row.get("transcript") or "").strip()
            if not transcript_ok(t, lang):
                continue
            try:
                d = float(row.get("duration_sec") or 0)
            except ValueError:
                d = 0
            if d and not (LO <= d <= HI):
                continue
            g = (row.get("gender") or "").strip().lower()
            key = "male" if g.startswith("m") else "female" if g.startswith("f") else "unknown"
            path = os.path.join(SO, corpus, "audio", ds, row["audio_id"] + ".wav")
            pen = 45.0 if FIELD_DS.search(ds) else (0.0 if GOOD_DS.search(ds) else 20.0)
            by[key].append({"path": path, "transcript": t, "dataset": ds, "pen": pen})
    return by


def pick(lang):
    by = gather(lang)
    random.seed(5)
    out = {}
    labelled = len(by["male"]) > 20 and len(by["female"]) > 20

    for g in ("male", "female"):
        pool = by[g] if labelled else by["unknown"] + by["male"] + by["female"]
        random.shuffle(pool)
        pool.sort(key=lambda c: c["pen"])
        best, tried = None, 0
        for c in pool:
            if tried >= MAX_ANALYSE:
                break
            if not os.path.exists(c["path"]):
                continue
            tried += 1
            m = analyse(c["path"])
            if not m:
                continue
            if not labelled and m["detected_gender"] != g:
                continue
            q = score(m, c["pen"])
            if best is None or q > best["q"]:
                best = {"q": q, "path": c["path"], "transcript": c["transcript"],
                        "dataset": c["dataset"], "metrics": m}
        out[g] = best
    out["_labelled"] = labelled
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*")
    args = ap.parse_args()
    langs = args.langs or list(CORPUS)

    res = {}
    if os.path.exists(args.out):
        try:
            res = json.load(open(args.out))
        except Exception:
            res = {}

    for lang in langs:
        if lang in res and res[lang].get("male") and res[lang].get("female"):
            print(f"{lang:10} already complete, skipping", flush=True)
            continue
        r = pick(lang)
        res[lang] = r
        bits = []
        for g in ("male", "female"):
            c = r.get(g)
            bits.append(f"{g[0].upper()}: {c['dataset'][:18]} f0={c['metrics']['f0_med']:.0f}"
                        if c else f"{g[0].upper()}: none")
        print(f"{lang:10} {'labelled' if r['_labelled'] else 'F0-detected':12} " + " | ".join(bits), flush=True)
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)

    both = [l for l, v in res.items() if v.get("male") and v.get("female")]
    print(f"\nboth genders: {len(both)}/{len(res)}", flush=True)
    print("incomplete:", [l for l, v in res.items() if not (v.get("male") and v.get("female"))], flush=True)


if __name__ == "__main__":
    main()
