"""Find unambiguously female (or male) reference clips using pyin rather than yin.

yin was reading roughly 15-20 Hz high on these corpora, which pushed borderline voices over
the female threshold -- the twi "female" candidates measured 192-213 Hz by yin but 177-195 Hz
by pyin, i.e. inside the overlap where a listener hears them as male. pyin tracks the true
fundamental far more reliably, so gender is decided on it, with a wide guard band.
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
REF_MIN_SEC = 9.0
from strict_gender_select import CORPUS, BAD_DS, GOOD_DS, FIELD_DS, transcript_ok, TARGET_PEAK
csv.field_size_limit(min(sys.maxsize, 2**31-1))
SO = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"

def analyse(path, fmin_gate, gender):
    try:
        if sf.info(path).samplerate < SOURCE_SR_MIN: return None
    except Exception: return None
    try: y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception: return None
    dur=len(y)/sr
    if not (REF_MIN_SEC <= dur <= 18.0): return None
    peak=float(np.max(np.abs(y))); rms=float(np.sqrt(np.mean(y**2)))
    if peak<0.06 or peak>0.999 or rms<0.008: return None
    # cheap yin pre-filter so pyin only runs on plausible clips
    yv=librosa.yin(y,fmin=60,fmax=400,sr=sr,frame_length=1024,hop_length=256)
    e=librosa.feature.rms(y=y,frame_length=1024,hop_length=256)[0][:len(yv)]
    v=(e>np.percentile(e,60))&(yv>63)&(yv<380)
    if v.sum()<25: return None
    rough=float(np.median(yv[v]))
    if gender=="female" and rough < fmin_gate-30: return None
    if gender=="male"   and rough > fmin_gate+30: return None
    f0,_,_=librosa.pyin(y,fmin=60,fmax=400,sr=sr,frame_length=1024,hop_length=256)
    ok=~np.isnan(f0)
    if ok.sum()<25: return None
    med=float(np.median(f0[ok]))
    if gender=="female" and med < fmin_gate: return None
    if gender=="male"   and med > fmin_gate: return None
    r=librosa.feature.rms(y=y,frame_length=512,hop_length=256)[0]
    db=20*np.log10(np.maximum(r,1e-9))
    noise,speech=float(np.percentile(db,10)),float(np.percentile(db,90))
    silence=float(np.mean(r<np.percentile(r,95)*0.02))
    loud=np.argsort(e)[-max(1,int(len(e)*0.4)):]
    fl=[]
    for i in loud:
        seg=y[i*256:i*256+1024]
        if len(seg)<256: continue
        P=np.abs(np.fft.rfft(seg*np.hanning(len(seg))))**2+1e-12
        fl.append(float(np.exp(np.mean(np.log(P)))/np.mean(P)))
    flat=float(np.mean(fl)) if fl else 1.0
    if silence>=0.04:
        if noise>-58 or (speech-noise)<40: return None
    elif flat>0.010: return None
    return dict(dur=round(dur,2), peak=round(peak,3), rms=round(rms,4),
                noise_dbfs=round(noise,1), snr=round(speech-noise,1), silence=round(silence,3),
                speech_flat=round(flat,5), f0_pyin=round(med,1), f0_yin=round(rough,1))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--lang",required=True); ap.add_argument("--gender",required=True)
    ap.add_argument("--gate",type=float,required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--wavdir",required=True)
    ap.add_argument("--want",type=int,default=5); ap.add_argument("--per-dir",type=int,default=500)
    a=ap.parse_args(); os.makedirs(a.wavdir,exist_ok=True)
    parent=os.path.join(SO,CORPUS[a.lang],"audio")
    dirs=[]
    for d in sorted(os.listdir(parent)):
        if not os.path.isdir(os.path.join(parent,d)) or BAD_DS.search(d): continue
        dirs.append((d,40.0 if FIELD_DS.search(d) else (0.0 if GOOD_DS.search(d) else 15.0)))
    dirs.sort(key=lambda x:x[1])
    random.seed(53); found=[]
    for d,pen in dirs:
        p=os.path.join(parent,d)
        try: files=[f for f in os.listdir(p) if f.endswith(".wav")]
        except OSError: continue
        random.shuffle(files)
        for fn in files[:a.per_dir]:
            m=analyse(os.path.join(p,fn), a.gate, a.gender)
            if m: found.append({"id":os.path.splitext(fn)[0],"path":os.path.join(p,fn),
                                "dataset":d,"pen":pen,"m":m})
        print(f"  {d}: {len(found)} so far", flush=True)
        if len(found)>=a.want*4: break
    ids={f["id"]:f for f in found}
    cp=os.path.join(SO,CORPUS[a.lang],f"speech_transcribed_{CORPUS[a.lang]}.csv")
    if os.path.exists(cp):
        with open(cp,encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                s=ids.get(row.get("audio_id"))
                if s is not None: s["transcript"]=(row.get("transcript") or "").strip()
    pool=[s for s in found if transcript_ok(s.get("transcript",""),a.lang)]
    # most clearly-gendered first, then cleanest
    pool.sort(key=lambda s: (-(abs(s["m"]["f0_pyin"]-a.gate)), s["m"]["noise_dbfs"]))
    out=[]
    for s in pool[:a.want]:
        f=f"{a.lang}__{a.gender}_P{len(out)+1}.wav"
        y,sr=librosa.load(s["path"],sr=None,mono=True)
        pk=float(np.max(np.abs(y)))
        sf.write(os.path.join(a.wavdir,f), np.clip(y*(TARGET_PEAK/pk),-1,1), sr)
        out.append({**{k:s[k] for k in ("path","transcript","dataset")}, "metrics":s["m"], "file":f})
        print(f"  -> {f} pyin={s['m']['f0_pyin']} Hz (yin said {s['m']['f0_yin']}) [{s['dataset']}]", flush=True)
    json.dump({a.lang:{a.gender:out}}, open(a.out,"w"), ensure_ascii=False, indent=2)
    print(f"{a.lang} {a.gender}: {len(out)}", flush=True)
