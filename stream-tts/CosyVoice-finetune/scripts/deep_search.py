"""Deeper, gender-strict search for the languages whose candidates were rejected by ear.

Three of the rejections were gender errors, not quality ones: lingala "none are male",
twi "cannot hear a female", chichewa female_2 "is actually a male's voice". A single
165 Hz split is too loose -- male and female ranges overlap either side of it. So this
uses a guard band (male < 145 Hz, female > 190 Hz) and simply discards anything in
between rather than guessing.

It also samples far more of each corpus, since igbo and swahili were rejected on quality
and the earlier pass only looked at a small slice.
"""
import argparse, csv, json, os, random, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf, librosa
from strict_gender_select import (CORPUS, BAD_DS, GOOD_DS, FIELD_DS, transcript_ok, TARGET_PEAK)
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
SO = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"

MALE_MAX, FEMALE_MIN = 145.0, 190.0     # guard band: ambiguous pitches are dropped
PER_DIR = 420
TIERS = [(-66.0, 48.0, 2.2, "1 broadcast-clean"),
         (-60.0, 42.0, 2.5, "2 clean"),
         (-52.0, 34.0, 2.9, "3 usable")]

def measure(path):
    try: y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception: return None
    dur = len(y)/sr
    if not (3.5 <= dur <= 18.0): return None
    peak = float(np.max(np.abs(y))); rms = float(np.sqrt(np.mean(y**2)))
    if peak < 0.06 or peak > 0.999 or rms < 0.008: return None
    if float(np.mean(np.abs(y) > 0.99)) > 0.002: return None
    r = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    db = 20*np.log10(np.maximum(r, 1e-9))
    noise, speech = float(np.percentile(db,10)), float(np.percentile(db,90))
    silence = float(np.mean(r < np.percentile(r,95)*0.02))
    if silence > 0.35: return None
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256)
    e = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][:len(f0)]
    v = (e > np.percentile(e,60)) & (f0 > 63) & (f0 < 380)
    if v.sum() < 25: return None
    med = float(np.median(f0[v]))
    p25, p75 = float(np.percentile(f0[v],25)), float(np.percentile(f0[v],75))
    if p75 - p25 > 85: return None
    g = "male" if med < MALE_MAX else "female" if med > FEMALE_MIN else None
    if g is None: return None                      # inside the guard band -> ambiguous
    # "snr" above is really dynamic range (P90-P10 of frame energy). A clip of continuous
    # speech has no quiet frames, so that number collapses even when the recording is clean --
    # which is exactly why 59 of 61 clean lingala males were being thrown away. Measure
    # cleanliness a second way that does not need pauses: spectral flatness over the loud
    # frames, where broadband hiss shows up directly.
    loud = np.argsort(e)[-max(1, int(len(e)*0.4)):]
    fl = []
    for i in loud:
        seg = y[i*256:i*256+1024]
        if len(seg) < 256: continue
        P = np.abs(np.fft.rfft(seg*np.hanning(len(seg))))**2 + 1e-12
        fl.append(float(np.exp(np.mean(np.log(P)))/np.mean(P)))
    speech_flat = float(np.mean(fl)) if fl else 1.0
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
                speech_flat=round(speech_flat,5),
                f0_med=round(med,1), speaker_split=round(split,2), gender=g)

def score(m, pen):
    return (min(m["snr"],90)*1.3 - m["noise_dbfs"]*0.5 + min(m["rms"],0.15)*200
            - abs(m["dur"]-8.5)*2.0 - m["speaker_split"]*20 - m["silence"]*90 - pen)

def run(lang, genders, want=4):
    parent = os.path.join(SO, CORPUS[lang], "audio")
    dirs = []
    for d in sorted(os.listdir(parent)):
        if not os.path.isdir(os.path.join(parent,d)) or BAD_DS.search(d): continue
        dirs.append((d, 40.0 if FIELD_DS.search(d) else (0.0 if GOOD_DS.search(d) else 15.0)))
    dirs.sort(key=lambda x: x[1])
    random.seed(37); scored=[]
    for d, pen in dirs:
        p = os.path.join(parent, d)
        try: files=[f for f in os.listdir(p) if f.endswith(".wav")]
        except OSError: continue
        random.shuffle(files)
        for fn in files[:PER_DIR]:
            m = measure(os.path.join(p,fn))
            if m and m["gender"] in genders:
                scored.append({"id":os.path.splitext(fn)[0],"path":os.path.join(p,fn),
                               "dataset":d,"pen":pen,"m":m})
        if all(sum(1 for s in scored if s["m"]["gender"]==g) >= 30 for g in genders): break
    ids={s["id"]:s for s in scored}
    cp=os.path.join(SO, CORPUS[lang], f"speech_transcribed_{CORPUS[lang]}.csv")
    if os.path.exists(cp):
        with open(cp,encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                s=ids.get(row.get("audio_id"))
                if s is not None: s["transcript"]=(row.get("transcript") or "").strip()
    out={}
    for g in genders:
        pool=[s for s in scored if s["m"]["gender"]==g and transcript_ok(s.get("transcript",""),lang)]
        pool.sort(key=lambda s:-score(s["m"], s["pen"]))
        chosen, seen = [], set()
        for noise,snr,split,label in TIERS:
            for s in pool:
                if len(chosen)>=want: break
                m=s["m"]
                if m["speaker_split"]>split: continue
                if m["silence"] >= 0.04:
                    # there are real pauses, so the floor and the dynamic range both mean
                    # something -- judge the recording on them
                    if m["noise_dbfs"]>noise or m["snr"]<snr: continue
                else:
                    # continuous speech: P10 of frame energy is still speech, so BOTH the
                    # "noise floor" and the "SNR" are meaningless here. Judge by broadband
                    # hiss in the speech itself instead.
                    if m["speech_flat"] > (0.006 if noise <= -66 else 0.010 if noise <= -60 else 0.016):
                        continue
                k=s["transcript"][:40]
                if k in seen: continue
                seen.add(k)
                chosen.append({"path":s["path"],"transcript":s["transcript"],"dataset":s["dataset"],
                               "metrics":m,"tier":label,"q":round(score(m,s["pen"]),1)})
            if len(chosen)>=want: break
        out[g]=chosen
    return out

def normalise(src,dst):
    y,sr=librosa.load(src,sr=None,mono=True)
    pk=float(np.max(np.abs(y)))
    if pk>0: y=y*(TARGET_PEAK/pk)
    sf.write(dst,np.clip(y,-1,1),sr)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--lang",required=True); ap.add_argument("--genders",nargs="+",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--wavdir",required=True)
    ap.add_argument("--want",type=int,default=4)
    a=ap.parse_args(); os.makedirs(a.wavdir,exist_ok=True)
    c=run(a.lang,a.genders,a.want)
    for g,lst in c.items():
        for i,e in enumerate(lst,1):
            f=f"{a.lang}__{g}_R{i}.wav"; normalise(e["path"],os.path.join(a.wavdir,f)); e["file"]=f
    json.dump({a.lang:c},open(a.out,"w"),ensure_ascii=False,indent=2)
    for g,lst in c.items():
        print(f"{a.lang} {g}: {len(lst)}" + (f" tier{lst[0]['tier'][0]} f0={lst[0]['metrics']['f0_med']}" if lst else ""), flush=True)
