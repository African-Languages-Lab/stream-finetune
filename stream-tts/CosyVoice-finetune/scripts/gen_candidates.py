"""Emit several loud/clean candidates per gender per language for human selection.

Deliberately does not choose: it normalises each survivor to a common level, writes the
transcript and measurements beside it, and leaves the judgement to a listener. My automated
picks have repeatedly passed measurement while failing the ear (two speakers, a learner
reading, a transcript in the wrong script), so the model here is "shortlist, don't decide".
"""
import argparse, json, os, random, shutil, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf, librosa
from strict_gender_select import (CORPUS, analyse, gather, score, TARGET_PEAK)

def normalise(src, dst, target=TARGET_PEAK):
    y, sr = librosa.load(src, sr=None, mono=True)
    pk = float(np.max(np.abs(y)))
    if pk > 0: y = y * (target / pk)
    sf.write(dst, np.clip(y, -1, 1), sr)

def candidates(lang, n=4, budget=320):
    by = gather(lang)
    random.seed(13)
    labelled = len(by["male"]) > 20 and len(by["female"]) > 20
    out = {}
    for g in ("male", "female"):
        pool = list(by[g]) if labelled else by["unknown"] + by["male"] + by["female"]
        random.shuffle(pool); pool.sort(key=lambda c: c["pen"])
        found, tried, seen = [], 0, set()
        for c in pool:
            if tried >= budget or len(found) >= n * 3: break
            if not os.path.exists(c["path"]): continue
            tried += 1
            m = analyse(c["path"])
            if not m or m["detected_gender"] != g: continue
            if c["transcript"][:40] in seen: continue
            seen.add(c["transcript"][:40])
            found.append({"path": c["path"], "transcript": c["transcript"],
                          "dataset": c["dataset"], "metrics": m,
                          "q": round(score(m, c["pen"]), 1)})
        found.sort(key=lambda c: -c["q"])
        out[g] = found[:n]
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--wavdir", required=True)
    ap.add_argument("--langs", nargs="+", required=True); ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.wavdir, exist_ok=True)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for lang in a.langs:
        if lang in res and any(res[lang].values()):
            print(f"{lang:10} done already", flush=True); continue
        c = candidates(lang, a.n)
        for g, lst in c.items():
            for i, e in enumerate(lst, 1):
                f = f"{lang}__{g}_{i}.wav"
                normalise(e["path"], os.path.join(a.wavdir, f)); e["file"] = f
        res[lang] = c
        print(f"{lang:10} male={len(c['male'])} female={len(c['female'])}", flush=True)
        json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=2)
