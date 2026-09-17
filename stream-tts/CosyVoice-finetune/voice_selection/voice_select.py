"""Select male and female reference clips per language, rejecting multi-speaker recordings.

The earlier passes scored only stationary signal quality (SNR, noise floor, clipping, dead
air). That missed exactly the faults a listener catches immediately:
  * two people talking (chichewa),
  * background conversation under the speaker (luganda),
  * a voice that does not sound like a native reading (igbo).

So this adds:
  * speaker_split  -- 2-means over per-second MFCC means; a high between/within ratio means
                      the timbre changes partway through, i.e. more than one voice;
  * f0_spread      -- interquartile range of the pitch track; one speaker reading a sentence
                      has a fairly tight range, overlapping voices widen it;
  * voiced_ratio   -- fraction of frames with detectable pitch; background chatter under a
                      pause shows up as "voiced" material in otherwise silent regions.

Gender comes from median F0 over voiced frames (below ~165 Hz reads male, above female),
which also gives the male/female preset pair.
"""
import csv
import glob
import json
import os
import random
import re
import sys
import unicodedata
import warnings

import librosa
import numpy as np

warnings.filterwarnings("ignore")
csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

SO = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"
SR = 16000
F0_MIN, F0_MAX = 60, 400
GENDER_SPLIT = 165.0

CORPUS = {
    "hausa": "hausa", "twi": "twi", "igbo": "igbo", "ewe": "ewe", "berber": "berber",
    "umbundu": "umbundu", "amharic": "amharic", "arabic": "arabic", "fula": "fula",
    "luganda": "luganda", "lingala": "lingala", "malagasy": "malagasy", "sepedi": "sepedi",
    "chichewa": "chichewa", "oromo": "oromo", "somali": "somali", "sesotho": "sesotho",
    "swahili": "swahili", "tigrinya": "tigrinya", "tswana": "tswana", "tsonga": "tsonga",
    "venda": "venda", "xhosa": "xhosa", "zulu": "zulu",
}
SCRIPT = {"amharic": "ETHIOPIC", "tigrinya": "ETHIOPIC", "arabic": "ARABIC"}

# never clone a synthetic voice, and never clone a second-language LEARNER reading -- the
# previous zulu pick came from l2_learner_zulu, which is exactly the "does not sound native"
# fault to avoid.
BAD_DIR = re.compile(r"synthetic|xtts|yourtts|_tts_synth|vits|l2_|learner|_l2", re.I)
# read/studio corpora first; conversational + crowdsourced field sets last
GOOD_DIR = re.compile(r"bible|fleurs|lwazi|anv[-_]|anv_|afrivoice|african_voices|ugspeech|naija|[-_]tts", re.I)
FIELD_DIR = re.compile(r"waxal|common.?voice|radio|broadcast|interview|conversation|speech_dataset", re.I)


def analyse(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
    except Exception:
        return None
    n = len(y)
    if n < SR * 4:
        return None
    dur = n / sr
    if not (4.5 <= dur <= 14.0):
        return None

    peak = float(np.max(np.abs(y)))
    if not (0.12 <= peak <= 0.999):
        return None
    if float(np.mean(np.abs(y) > 0.99)) > 0.0005:
        return None

    fl = 512
    rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=fl // 2)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    noise = float(np.percentile(db, 10))
    speech = float(np.percentile(db, 90))
    silence = float(np.mean(rms < np.percentile(rms, 95) * 0.02))
    if noise > -55 or silence > 0.32:
        return None

    # pitch track over voiced frames
    f0 = librosa.yin(y, fmin=F0_MIN, fmax=F0_MAX, sr=sr, frame_length=1024, hop_length=256)
    e = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][:len(f0)]
    voiced = (e > np.percentile(e, 55)) & (f0 > F0_MIN * 1.05) & (f0 < F0_MAX * 0.95)
    if voiced.sum() < 20:
        return None
    f0v = f0[voiced]
    f0_med = float(np.median(f0v))
    f0_iqr = float(np.percentile(f0v, 75) - np.percentile(f0v, 25))
    voiced_ratio = float(voiced.mean())

    # timbre consistency: 2-means over per-second MFCC means
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    per_sec = max(1, int(sr / 256))
    segs = [mf[:, i:i + per_sec].mean(axis=1)
            for i in range(0, mf.shape[1] - per_sec // 2, per_sec)]
    segs = np.array([s for s in segs if not np.isnan(s).any()])
    if len(segs) < 4:
        return None
    segs = (segs - segs.mean(0)) / (segs.std(0) + 1e-9)
    c = [segs[0], segs[-1]]
    for _ in range(12):
        d = np.stack([np.linalg.norm(segs - ci, axis=1) for ci in c])
        lab = d.argmin(0)
        if lab.min() == lab.max():
            break
        c = [segs[lab == k].mean(0) for k in (0, 1)]
    if lab.min() == lab.max():
        split = 0.0
    else:
        between = np.linalg.norm(c[0] - c[1])
        within = np.mean([np.linalg.norm(segs[lab == k] - c[k], axis=1).mean() for k in (0, 1)])
        split = float(between / (within + 1e-9))

    return dict(sr=int(sr), dur=round(dur, 2), peak=round(peak, 3),
                rms=round(float(np.sqrt(np.mean(y ** 2))), 4),
                snr=round(speech - noise, 1), noise_dbfs=round(noise, 1),
                silence=round(silence, 3), f0_med=round(f0_med, 1),
                f0_iqr=round(f0_iqr, 1), voiced_ratio=round(voiced_ratio, 3),
                speaker_split=round(split, 2),
                gender="male" if f0_med < GENDER_SPLIT else "female")


def single_speaker(m):
    """Reject clips that look like more than one voice."""
    # calibrated against known cases: the vetted hausa clip scores 1.59, while the
    # two-speaker chichewa scores 3.93 and the background-chatter luganda 3.49.
    return (m["speaker_split"] <= 2.6        # timbre stays put across the clip
            and m["f0_iqr"] <= 75            # pitch range consistent with one reader
            and m["voiced_ratio"] <= 0.80)   # not wall-to-wall voicing (chatter under pauses)


def score(m, prefer_field_penalty=0.0):
    s = -m["noise_dbfs"] * 0.7
    s -= abs(m["dur"] - 8.5) * 3.0
    s -= m["silence"] * 110
    s -= m["speaker_split"] * 22            # strongly favour one clear voice
    s -= max(0.0, m["f0_iqr"] - 40) * 0.8
    s -= abs(np.log10(max(m["rms"], 1e-4) / 0.06)) * 28
    s -= prefer_field_penalty
    return s


def script_ok(t, want):
    L = [c for c in t if c.isalpha()]
    return bool(L) and sum(1 for c in L if want in unicodedata.name(c, "")) / len(L) >= 0.65


def transcript_ok(t, lang):
    if not t:
        return False
    t = t.strip()
    if len(t) < 12 or " " not in t or len(t.split()) < 3:
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
    if t.count(",") >= len(toks) * 0.35:          # comma-separated word list
        return False
    stems = {}
    for w in toks:
        if len(w) >= 4:
            stems[w[:4]] = stems.get(w[:4], 0) + 1
    return max(stems.values(), default=0) < max(3, len(toks) * 0.4)


def rank_dirs(parent):
    ds = [d for d in sorted(os.listdir(parent))
          if os.path.isdir(os.path.join(parent, d)) and not BAD_DIR.search(d)]
    return sorted(ds, key=lambda d: (0 if GOOD_DIR.search(d) else (2 if FIELD_DIR.search(d) else 1), d))


def collect(lang, per_dir, want_n=40):
    corpus = CORPUS[lang]
    parent = os.path.join(SO, corpus, "audio")
    if not os.path.isdir(parent):
        return []
    random.seed(hash(lang) & 0xffff)
    pool = []
    for d in rank_dirs(parent):
        files = glob.glob(os.path.join(parent, d, "*.wav"))
        if not files:
            continue
        random.shuffle(files)
        pen = 45.0 if FIELD_DIR.search(d) else 0.0
        for f in files[:per_dir]:
            m = analyse(f)
            if m and single_speaker(m):
                pool.append({"path": f, "metrics": m, "dataset": d,
                             "q": score(m, pen), "gender": m["gender"]})
        if sum(1 for c in pool if c["gender"] == "male") >= want_n // 2 and \
           sum(1 for c in pool if c["gender"] == "female") >= want_n // 2:
            break
    return pool


def resolve_transcripts(lang, cands):
    corpus = CORPUS[lang]
    idmap = {}
    for c in cands:
        idmap.setdefault(os.path.splitext(os.path.basename(c["path"]))[0], []).append(c)
    p = os.path.join(SO, corpus, f"speech_transcribed_{corpus}.csv")
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("audio_id") in idmap:
                for c in idmap[row["audio_id"]]:
                    c["transcript"] = (row.get("transcript") or "").strip()
                    c["corpus_gender"] = (row.get("gender") or "").strip()


def pick_for(lang, per_dir=110):
    pool = collect(lang, per_dir)
    out = {}
    for g in ("male", "female"):
        cands = sorted([c for c in pool if c["gender"] == g], key=lambda c: -c["q"])[:25]
        if not cands:
            out[g] = None
            continue
        resolve_transcripts(lang, cands)
        out[g] = next((c for c in cands if transcript_ok(c.get("transcript", ""), lang)), None)
    return out, len(pool)


if __name__ == "__main__":
    out_path = sys.argv[1]
    langs = sys.argv[2:] or list(CORPUS)
    # write after every language: these sweeps get killed (scratch resets, OOM) and losing
    # 40 minutes of corpus scanning to a truncated run is not worth the tidier code.
    result = {}
    if os.path.exists(out_path):
        try:
            result = json.load(open(out_path))
            print(f"resuming, {len(result)} already done", flush=True)
        except Exception:
            result = {}
    for lang in langs:
        if lang in result and any(result[lang].get(g) for g in ("male", "female")):
            print(f"{lang:10} already done, skipping", flush=True)
            continue
        try:
            picks, n = pick_for(lang)
        except Exception as exc:
            print(f"{lang:10} FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        result[lang] = picks
        bits = []
        for g in ("male", "female"):
            c = picks.get(g)
            if c:
                m = c["metrics"]
                bits.append(f"{g[0].upper()}: {c['dataset'][:20]} f0={m['f0_med']:.0f} "
                            f"split={m['speaker_split']} noise={m['noise_dbfs']}")
            else:
                bits.append(f"{g[0].upper()}: none")
        print(f"{lang:10} pool={n:>4}  " + " | ".join(bits), flush=True)
        json.dump(result, open(out_path, "w"), ensure_ascii=False, indent=2)
    print("\nwrote", out_path, f"({len(result)} languages)")
