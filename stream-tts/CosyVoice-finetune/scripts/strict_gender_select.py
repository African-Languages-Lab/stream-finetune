"""Pick loud, clear, quiet-background male/female reference clips per language.

The previous pass let through 24 of 38 clips that were too quiet, too noisy, or both --
its floor was peak >= 0.12 and noise <= -52 dBFS, which is nowhere near "loud and clean".

Bar here:
    peak      >= 0.50      (not a faint recording)
    rms       >= 0.050     (actually carries level, not just one loud transient)
    noise     <= -65 dBFS  (silences are genuinely silent)
    snr       >= 50 dB     (speech sits well clear of the floor)
    split     <= 2.2       (one speaker, no background conversation)

Selected clips are then peak-normalised to a common level so every reference sounds equally
loud -- normalisation alone cannot rescue a noisy clip (it lifts the noise too), which is why
the noise floor is screened before, not after.
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import unicodedata
import warnings

warnings.filterwarnings("ignore")
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
SO = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"

CORPUS = {
    "hausa": "hausa", "twi": "twi", "igbo": "igbo", "ewe": "ewe", "berber": "berber",
    "umbundu": "umbundu", "amharic": "amharic", "arabic": "arabic", "fula": "fula",
    "luganda": "luganda", "lingala": "lingala", "malagasy": "malagasy", "sepedi": "sepedi",
    "chichewa": "chichewa", "oromo": "oromo", "somali": "somali", "sesotho": "sesotho",
    "swahili": "swahili", "tigrinya": "tigrinya", "tswana": "tswana", "tsonga": "tsonga",
    "venda": "venda", "xhosa": "xhosa", "zulu": "zulu",
    # trained but not yet served -- under evaluation
    "afrikaans": "afrikaans",
    "bemba": "bemba",
    "english": "english",
    "kikuyu": "kikuyu",
    "ndebele": "ndebele",
    "kinyarwanda": "kinyarwanda",
    "shona": "shona",
    "wolof": "wolof",
    "yoruba": "yoruba",
    # newly finished trainings, not yet served
    "swati": "swati", "bambara": "bambara", "kanuri": "kanuri",
    "fon": "fon", "krio": "krio",
}
SCRIPT = {"amharic": "ETHIOPIC", "tigrinya": "ETHIOPIC", "arabic": "ARABIC"}
# berber must be Latin Tamazight; its Tifinagh/Arabic corpora are unusable as-is
BERBER_LATIN = {"moroccan_amazigh_asr", "Beni-Mellal-Tamazight", "8dretna_tamazight",
                "hdrtna_tamazight", "Tamazight-Arabic-Translation"}

BAD_DS = re.compile(r"synthetic|xtts|yourtts|vits|l2_|learner|_l2", re.I)
GOOD_DS = re.compile(r"bible|fleurs|lwazi|anv[-_]|afrivoice|african_voices|naija|ugspeech|[-_]tts", re.I)
FIELD_DS = re.compile(r"waxal|common.?voice|radio|broadcast|conversation|speech_dataset", re.I)

# Loudness is fixed by normalising at the end, so gating on raw peak only discarded good
# recordings -- the "Anya" hausa clip has peak 0.41 but noise -72 dBFS and 52 dB SNR,
# which is exactly what we want. What normalising CANNOT fix is a high noise floor or
# poor SNR (turning those up just makes the noise louder), so the bar sits there.
MIN_PEAK, MIN_RMS, MAX_NOISE, MIN_SNR, MAX_SPLIT = 0.10, 0.012, -66.0, 48.0, 2.3
TARGET_PEAK = 0.85
LO, HI = 4.0, 15.0
MAX_ANALYSE = 240


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
    dur = len(y) / sr
    if not (LO <= dur <= HI):
        return None
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    if peak < MIN_PEAK or peak > 0.999 or rms < MIN_RMS:
        return None
    if float(np.mean(np.abs(y) > 0.99)) > 0.001:
        return None

    r = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    db = 20 * np.log10(np.maximum(r, 1e-9))
    noise, speech = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    if noise > MAX_NOISE or (speech - noise) < MIN_SNR:
        return None
    silence = float(np.mean(r < np.percentile(r, 95) * 0.02))
    if silence > 0.30:
        return None

    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256)
    e = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][:len(f0)]
    v = (e > np.percentile(e, 55)) & (f0 > 63) & (f0 < 380)
    if v.sum() < 20:
        return None
    med = float(np.median(f0[v]))
    if float(np.percentile(f0[v], 75) - np.percentile(f0[v], 25)) > 80:
        return None

    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    per = max(1, int(sr / 256))
    segs = np.array([mf[:, i:i + per].mean(axis=1)
                     for i in range(0, mf.shape[1] - per // 2, per)])
    segs = segs[~np.isnan(segs).any(axis=1)]
    if len(segs) < 4:
        return None
    segs = (segs - segs.mean(0)) / (segs.std(0) + 1e-9)
    c, lab = [segs[0], segs[-1]], None
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
    if split > MAX_SPLIT:
        return None

    return dict(dur=round(dur, 2), peak=round(peak, 3), rms=round(rms, 4),
                noise_dbfs=round(noise, 1), snr=round(speech - noise, 1),
                silence=round(silence, 3), f0_med=round(med, 1),
                speaker_split=round(split, 2),
                detected_gender="male" if med < 165 else "female")


def score(m, pen):
    # reward loudness and a quiet floor above all else
    return (min(m["snr"], 90) * 1.2
            - m["noise_dbfs"] * 0.5
            + min(m["rms"], 0.15) * 220
            - abs(m["dur"] - 8.5) * 2.5
            - m["speaker_split"] * 18
            - m["silence"] * 90
            - pen)


def gather(lang):
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
            if lang == "berber" and ds not in BERBER_LATIN:
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
            by[key].append({
                "path": os.path.join(SO, corpus, "audio", ds, row["audio_id"] + ".wav"),
                "transcript": t, "dataset": ds,
                "pen": 40.0 if FIELD_DS.search(ds) else (0.0 if GOOD_DS.search(ds) else 15.0)})
    return by


def pick(lang):
    by = gather(lang)
    random.seed(9)
    labelled = len(by["male"]) > 20 and len(by["female"]) > 20
    out = {}
    for g in ("male", "female"):
        pool = list(by[g]) if labelled else by["unknown"] + by["male"] + by["female"]
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
            # trust measured pitch over the corpus label -- some labels are simply wrong
            if m["detected_gender"] != g:
                continue
            q = score(m, c["pen"])
            if best is None or q > best["q"]:
                best = {"q": round(q, 1), "path": c["path"], "transcript": c["transcript"],
                        "dataset": c["dataset"], "metrics": m}
        out[g] = best
    return out


def normalise(src, dst, target=TARGET_PEAK):
    y, sr = librosa.load(src, sr=None, mono=True)
    pk = float(np.max(np.abs(y)))
    if pk > 0:
        y = y * (target / pk)
    sf.write(dst, np.clip(y, -1, 1), sr)
    return sr, len(y) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wavdir", required=True)
    ap.add_argument("--langs", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.wavdir, exist_ok=True)

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for lang in (args.langs or list(CORPUS)):
        if lang in res and res[lang].get("male") and res[lang].get("female"):
            print(f"{lang:10} already complete", flush=True)
            continue
        r = pick(lang)
        for g in ("male", "female"):
            if r.get(g):
                dst = os.path.join(args.wavdir, f"{lang}__{g}.wav")
                sr, dur = normalise(r[g]["path"], dst)
                r[g]["normalised_wav"] = dst
        res[lang] = r
        bits = []
        for g in ("male", "female"):
            c = r.get(g)
            bits.append(
                f"{g[0].upper()}: {c['dataset'][:16]} f0={c['metrics']['f0_med']:.0f} "
                f"snr={c['metrics']['snr']:.0f} nz={c['metrics']['noise_dbfs']:.0f}"
                if c else f"{g[0].upper()}: NONE")
        print(f"{lang:10} " + " | ".join(bits), flush=True)
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)

    both = [l for l, v in res.items() if v.get("male") and v.get("female")]
    print(f"\nboth genders: {len(both)}/{len(res)}", flush=True)
    print("incomplete:", [l for l, v in res.items() if not (v.get("male") and v.get("female"))], flush=True)


if __name__ == "__main__":
    main()
