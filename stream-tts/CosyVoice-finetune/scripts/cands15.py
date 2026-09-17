"""Shortlist male/female reference candidates, memory-light and tiered.

Earlier versions built paths from the CSV's dataset_name (which often does not match the
directory on disk, so files silently did not exist) and then tried to index every wav in a
corpus at once (which the login node OOM-killed).

This walks the audio directories directly, measures a bounded sample, and only then does a
single CSV pass to fetch transcripts for the handful of clips that survived. The quality bar
steps down through tiers so every language yields something, and each clip carries the tier
it qualified under.
"""
import argparse, csv, json, os, random, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf, librosa

# CosyVoice3's load_wav rejects anything under 16 kHz. librosa.load(sr=16000)
# resamples on read and hides the original rate, so check the file directly --
# an 8 kHz swati corpus slipped through and failed at synthesis time.
SOURCE_SR_MIN = 16000
# A too-short reference makes CosyVoice3 read the reference aloud before the
# requested text: hausa female at 3.9s leaked, hausa male at 11.4s did not, at
# the same words-per-second. CosyVoice's own code assumes a minimum prompt size.
REF_MIN_SEC = 15.0
from strict_gender_select import (CORPUS, BERBER_LATIN, BAD_DS, GOOD_DS, FIELD_DS,
                                  transcript_ok, TARGET_PEAK)
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
SO = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"

TIERS = [(-66.0, 48.0, 2.3, "1 broadcast-clean"),
         (-58.0, 40.0, 2.6, "2 clean"),
         (-50.0, 32.0, 3.0, "3 usable"),
         (-40.0, 22.0, 3.8, "4 best available")]
PER_DIR = 130          # files sampled per dataset directory
LO, HI = REF_MIN_SEC, 28.0

def measure(path):
    try:
        if sf.info(path).samplerate < SOURCE_SR_MIN: return None
    except Exception: return None
    try: y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception: return None
    dur = len(y)/sr
    if not (LO <= dur <= HI): return None
    peak = float(np.max(np.abs(y))); rms = float(np.sqrt(np.mean(y**2)))
    if peak < 0.05 or peak > 0.999 or rms < 0.006: return None
    if float(np.mean(np.abs(y) > 0.99)) > 0.003: return None
    r = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    db = 20*np.log10(np.maximum(r, 1e-9))
    noise, speech = float(np.percentile(db,10)), float(np.percentile(db,90))
    silence = float(np.mean(r < np.percentile(r,95)*0.02))
    if silence > 0.38: return None
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
        b = np.linalg.norm(c[0]-c[1])
        w = np.mean([np.linalg.norm(segs[lab==k]-c[k], axis=1).mean() for k in (0,1)])
        split = float(b/(w+1e-9))
    else: split = 0.0
    del y, mf, segs
    return dict(dur=round(dur,2), peak=round(peak,3), rms=round(rms,4),
                noise_dbfs=round(noise,1), snr=round(speech-noise,1), silence=round(silence,3),
                f0_med=round(med,1), speaker_split=round(split,2),
                gender="male" if med < 165 else "female")

def score(m, pen):
    return (min(m["snr"],90)*1.2 - m["noise_dbfs"]*0.5 + min(m["rms"],0.15)*220
            - abs(m["dur"]-8.5)*2.0 - m["speaker_split"]*18 - m["silence"]*90 - pen)

def dirs_for(lang):
    parent = os.path.join(SO, CORPUS[lang], "audio")
    if not os.path.isdir(parent): return []
    out = []
    for d in sorted(os.listdir(parent)):
        if not os.path.isdir(os.path.join(parent, d)) or BAD_DS.search(d): continue
        if lang == "berber" and d not in BERBER_LATIN: continue
        pen = 40.0 if FIELD_DS.search(d) else (0.0 if GOOD_DS.search(d) else 15.0)
        out.append((d, pen))
    return sorted(out, key=lambda x: x[1])

def normalise(src, dst):
    y, sr = librosa.load(src, sr=None, mono=True)
    pk = float(np.max(np.abs(y)))
    if pk > 0: y = y*(TARGET_PEAK/pk)
    sf.write(dst, np.clip(y,-1,1), sr)

def run(lang, want=3):
    random.seed(29)
    scored = []
    for d, pen in dirs_for(lang):
        p = os.path.join(SO, CORPUS[lang], "audio", d)
        try: files = [f for f in os.listdir(p) if f.endswith(".wav")]
        except OSError: continue
        random.shuffle(files)
        for fn in files[:PER_DIR]:
            m = measure(os.path.join(p, fn))
            if m: scored.append({"id": os.path.splitext(fn)[0], "path": os.path.join(p, fn),
                                 "dataset": d, "pen": pen, "m": m})
        if sum(1 for s in scored if s["m"]["gender"]=="male") >= 25 and \
           sum(1 for s in scored if s["m"]["gender"]=="female") >= 25:
            break
    # transcripts only for what survived
    ids = {s["id"]: s for s in scored}
    cp = os.path.join(SO, CORPUS[lang], f"speech_transcribed_{CORPUS[lang]}.csv")
    if os.path.exists(cp):
        with open(cp, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                s = ids.get(row.get("audio_id"))
                if s is not None: s["transcript"] = (row.get("transcript") or "").strip()
    out = {}
    for g in ("male","female"):
        pool = [s for s in scored if s["m"]["gender"]==g
                and transcript_ok(s.get("transcript",""), lang)]
        pool.sort(key=lambda s: -score(s["m"], s["pen"]))
        chosen, seen = [], set()
        for noise, snr, split, label in TIERS:
            for s in pool:
                if len(chosen) >= want: break
                m = s["m"]
                if m["noise_dbfs"] > noise or m["snr"] < snr or m["speaker_split"] > split: continue
                k = s["transcript"][:40]
                if k in seen: continue
                seen.add(k)
                chosen.append({"path": s["path"], "transcript": s["transcript"],
                               "dataset": s["dataset"], "metrics": m, "tier": label,
                               "q": round(score(m, s["pen"]),1)})
            if len(chosen) >= want: break
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
        c = run(lang, a.want)
        for g, lst in c.items():
            for i, e in enumerate(lst,1):
                f = f"{lang}__{g}_{i}.wav"; normalise(e["path"], os.path.join(a.wavdir,f)); e["file"]=f
        res[lang]=c
        f=lambda g: f"{g[0].upper()}:{len(c[g])}"+(f"[t{c[g][0]['tier'][0]}]" if c[g] else "")
        print(f"{lang:10} {f('male')} {f('female')}", flush=True)
        json.dump(res, open(a.out,"w"), ensure_ascii=False, indent=2)
