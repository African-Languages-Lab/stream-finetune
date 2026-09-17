"""Shortlist reference-voice candidates, relaxing the quality bar only as far as needed.

Every language gets candidates. The bar starts where we want it and steps down only when a
corpus cannot meet it, and each clip records the tier it came from -- so "this is the best
that exists for Tigrinya female" is visible rather than hidden behind a single pass/fail.

Tier 1 is broadcast-clean; tier 4 is "this is all the corpus has". Loudness is not screened
at any tier because every survivor is peak-normalised to a common level afterwards; what
normalising cannot repair is a high noise floor, which is what the tiers actually gate on.
"""
import argparse, csv, json, os, random, re, sys, unicodedata, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf, librosa
from strict_gender_select import (CORPUS, SCRIPT, BERBER_LATIN, BAD_DS, GOOD_DS, FIELD_DS,
                                  transcript_ok, TARGET_PEAK)
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
SO = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"

#        noise   snr   split  dur_lo dur_hi  label
TIERS = [(-66.0, 48.0, 2.3,   4.0, 15.0, "1 (broadcast-clean)"),
         (-58.0, 40.0, 2.6,   4.0, 16.0, "2 (clean)"),
         (-50.0, 32.0, 3.0,   3.5, 18.0, "3 (usable)"),
         (-42.0, 24.0, 3.6,   3.0, 22.0, "4 (best available)")]
BUDGET = 900
ROW_CAP = 3000        # per gender; enough to choose from without holding a whole corpus in memory

def measure(path, lo, hi):
    try: y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception: return None
    dur = len(y)/sr
    if not (lo <= dur <= hi): return None
    peak = float(np.max(np.abs(y))); rms = float(np.sqrt(np.mean(y**2)))
    if peak < 0.05 or peak > 0.999 or rms < 0.008: return None
    if float(np.mean(np.abs(y) > 0.99)) > 0.002: return None
    r = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    db = 20*np.log10(np.maximum(r, 1e-9))
    noise, speech = float(np.percentile(db,10)), float(np.percentile(db,90))
    silence = float(np.mean(r < np.percentile(r,95)*0.02))
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256)
    e = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][:len(f0)]
    v = (e > np.percentile(e,55)) & (f0 > 63) & (f0 < 380)
    if v.sum() < 20: return None
    med = float(np.median(f0[v]))
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=256)
    per = max(1, int(sr/256))
    segs = np.array([mf[:, i:i+per].mean(axis=1) for i in range(0, mf.shape[1]-per//2, per)])
    segs = segs[~np.isnan(segs).any(axis=1)]
    if len(segs) < 4: return None
    segs = (segs-segs.mean(0))/(segs.std(0)+1e-9)
    c, lab = [segs[0], segs[-1]], None
    for _ in range(10):
        d = np.stack([np.linalg.norm(segs-ci, axis=1) for ci in c]); lab = d.argmin(0)
        if lab.min() == lab.max(): break
        c = [segs[lab==k].mean(0) for k in (0,1)]
    if lab is not None and lab.min() != lab.max():
        between = np.linalg.norm(c[0]-c[1])
        within = np.mean([np.linalg.norm(segs[lab==k]-c[k], axis=1).mean() for k in (0,1)])
        split = float(between/(within+1e-9))
    else: split = 0.0
    return dict(dur=round(dur,2), peak=round(peak,3), rms=round(rms,4),
                noise_dbfs=round(noise,1), snr=round(speech-noise,1),
                silence=round(silence,3), f0_med=round(med,1), speaker_split=round(split,2),
                detected_gender="male" if med < 165 else "female")

def passes(m, t):
    noise, snr, split, lo, hi, _ = t
    return (m["noise_dbfs"] <= noise and m["snr"] >= snr
            and m["speaker_split"] <= split and m["silence"] <= 0.35)

def score(m, pen):
    return (min(m["snr"],90)*1.2 - m["noise_dbfs"]*0.5 + min(m["rms"],0.15)*220
            - abs(m["dur"]-8.5)*2.0 - m["speaker_split"]*18 - m["silence"]*90 - pen)

def _audio_index(corpus):
    """basename -> real path on disk.

    The CSV's dataset_name does not reliably match the directory name
    ("Chichewa-Speech-Dataset" vs chichewa_speech_dataset, "ANV-Tswana" vs anv-tsn), so
    composing paths from it silently produced files that do not exist -- which is why six
    languages returned zero candidates even at the most permissive tier. Index the real
    files once and look the audio_id up instead.
    """
    import glob as _glob
    idx = {}
    for f in _glob.glob(os.path.join(SO, corpus, "audio", "*", "*.wav")):
        idx.setdefault(os.path.splitext(os.path.basename(f))[0], f)
    return idx


def rows_for(lang):
    corpus = CORPUS[lang]
    p = os.path.join(SO, corpus, f"speech_transcribed_{corpus}.csv")
    by = {"male": [], "female": [], "unknown": []}
    if not os.path.exists(p): return by
    index = _audio_index(corpus)
    with open(p, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ds = row.get("dataset_name","")
            if not ds or BAD_DS.search(ds): continue
            if lang == "berber" and ds not in BERBER_LATIN: continue
            t = (row.get("transcript") or "").strip()
            if not transcript_ok(t, lang): continue
            g = (row.get("gender") or "").strip().lower()
            key = "male" if g.startswith("m") else "female" if g.startswith("f") else "unknown"
            if len(by[key]) >= ROW_CAP: continue      # bound memory on 200k-row corpora
            path = index.get(row["audio_id"])
            if not path: continue
            by[key].append({"path": path,
                            "transcript": t, "dataset": os.path.basename(os.path.dirname(path)),
                            "pen": 40.0 if FIELD_DS.search(ds) else (0.0 if GOOD_DS.search(ds) else 15.0)})
    return by

def normalise(src, dst):
    y, sr = librosa.load(src, sr=None, mono=True)
    pk = float(np.max(np.abs(y)))
    if pk > 0: y = y*(TARGET_PEAK/pk)
    sf.write(dst, np.clip(y,-1,1), sr)

def for_language(lang, want=3):
    by = rows_for(lang); random.seed(21)
    labelled = len(by["male"]) > 20 and len(by["female"]) > 20
    out = {}
    for g in ("male","female"):
        pool = list(by[g]) if labelled else by["unknown"]+by["male"]+by["female"]
        random.shuffle(pool); pool.sort(key=lambda c: c["pen"])
        measured, tried = [], 0
        for c in pool:
            if tried >= BUDGET: break
            if not os.path.exists(c["path"]): continue
            tried += 1
            m = measure(c["path"], 3.0, 22.0)
            if not m or m["detected_gender"] != g: continue
            measured.append((c, m))
        chosen, seen = [], set()
        for t in TIERS:                       # step down only until we have enough
            for c, m in measured:
                if len(chosen) >= want: break
                if not passes(m, t): continue
                k = c["transcript"][:40]
                if k in seen: continue
                seen.add(k)
                chosen.append({"path": c["path"], "transcript": c["transcript"],
                               "dataset": c["dataset"], "metrics": m, "tier": t[5],
                               "q": round(score(m, c["pen"]),1)})
            if len(chosen) >= want: break
        chosen.sort(key=lambda e: -e["q"])
        out[g] = chosen
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--wavdir", required=True)
    ap.add_argument("--langs", nargs="+", required=True); ap.add_argument("--want", type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.wavdir, exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for lang in a.langs:
        if lang in res and res[lang].get("male") and res[lang].get("female"):
            print(f"{lang:10} done", flush=True); continue
        c = for_language(lang, a.want)
        for g, lst in c.items():
            for i, e in enumerate(lst, 1):
                f = f"{lang}__{g}_{i}.wav"; normalise(e["path"], os.path.join(a.wavdir, f)); e["file"] = f
        res[lang] = c
        def d(g): return f"{g[0].upper()}:{len(c[g])}" + (f"[t{c[g][0]['tier'][0]}]" if c[g] else "")
        print(f"{lang:10} {d('male')} {d('female')}", flush=True)
        json.dump(res, open(a.out,"w"), ensure_ascii=False, indent=2)
