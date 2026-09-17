"""Which of the untested languages actually have a usable model?

The standing verdicts -- "collapses", "hiss", "too quiet", "generation errors" -- were each
reached from ONE automatically chosen checkpoint. Yoruba was condemned that way and turned out
to work fine on an earlier epoch, so none of the others can be trusted either.

For each language this tries three LLM epochs spread across the run (early, middle, late)
against the best male and female reference candidate, and reports whether anything speaks.
Flow is fixed: output length is decided by the LLM's token count and flow only renders those
tokens, so varying it cannot move this measure.

Both zero-shot and cross-lingual are tried. That matters because chichewa/male turned out to
have no working zero-shot cell at all while working perfectly with no transcript -- these
corpora's transcripts cannot be assumed to match their audio.
"""
import argparse
import json
import os
import re
import shutil
import sys

REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))
FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(FT, "scripts"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from try_new_models import PROBES  # noqa: E402

PRE = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"
CAND = "/leonardo_scratch/large/userexternal/atsado00/unserved_refs"
OUT = "/leonardo_scratch/large/userexternal/atsado00/unserved_sweep"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"
ISO = {"afrikaans": "af-ZA", "bemba": "bem-ZM", "english": "en-UG", "english_combined": "english", "kikuyu": "ki-KE",
       "ndebele": "nd-ZW", "shona": "sn-ZW", "wolof": "wo-SN", "swati": "ss-SZ",
       "bambara": "bm-ML", "kanuri": "kr-NG", "fon": "fon-BJ", "krio": "kri-SL", "kinyarwanda": "rw-RW"}
REPS = 2


def epochs(iso, stage):
    d = f"{ROOT}/individual_{iso}/{stage}"
    fs = [f for f in os.listdir(d) if re.fullmatch(r"epoch_\d+(_step_\d+)?_whole\.pt|epoch_\d+_step_\d+\.pt|epoch_\d+_whole\.pt", f)]
    def key(f):
        n = [int(x) for x in re.findall(r"\d+", f)]
        return (n[0], n[1] if len(n) > 1 else 10**9)   # whole sorts after its steps
    return sorted(fs, key=key)


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build(iso, llm, flow, d):
    os.makedirs(d, exist_ok=True)
    clean(f"{ROOT}/individual_{iso}/llm/{llm}", f"{d}/llm.pt")
    clean(f"{ROOT}/individual_{iso}/flow/{flow}", f"{d}/flow.pt")
    for a in SHARED:
        shutil.copy2(f"{PRE}/{a}", f"{d}/{a}")
    if not os.path.exists(f"{d}/CosyVoice-BlankEN"):
        shutil.copytree(f"{PRE}/CosyVoice-BlankEN", f"{d}/CosyVoice-BlankEN")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3

    results = []
    for lang in a.langs:
        iso = ISO[lang]
        try:
            le, fe = epochs(iso, "llm"), epochs(iso, "flow")
        except Exception as e:
            print(f"{lang}: no checkpoints ({e})", flush=True); continue
        if not le or not fe:
            print(f"{lang}: no whole checkpoints (llm={len(le)} flow={len(fe)})", flush=True); continue
        pick = [le[0], le[len(le) // 2], le[-1]] if len(le) >= 3 else le
        flow = fe[-1]
        cj = f"{CAND}/json/{'english' if lang == 'english_combined' else lang}.json"
        if not os.path.exists(cj):
            print(f"{lang}: no reference candidates", flush=True); continue
        cd = json.load(open(cj))["english" if lang == "english_combined" else lang]
        refs = []
        for g in ("male", "female"):
            lst = cd.get(g) or []
            if lst:
                refs.append((g, os.path.join(CAND, "wavs", lst[0]["file"]), lst[0]["transcript"]))
        target = PROBES.get("english" if lang == "english_combined" else lang)
        if not target:
            print(f"{lang}: no probe sentence", flush=True); continue

        print(f"\n=== {lang} ({iso}) -- llm {[p.replace('_whole.pt','') for p in pick]} "
              f"+ flow {flow.replace('_whole.pt','')} ===", flush=True)
        print(f"{'llm':10}{'ref':9}{'zero-shot':>22}{'cross-lingual':>22}", flush=True)
        for llm in pick:
            d = os.path.join(OUT, "work", f"{lang}_{llm}")
            try:
                model = CosyVoice3(build(iso, llm, flow, d), fp16=False)
            except Exception as e:
                print(f"{llm[:9]:10}  BUILD FAILED {str(e)[:50]}", flush=True); continue
            for g, wav, ptext in refs:
                rd = sf.info(wav).duration
                exp = len(target) / (len(ptext) / rd)
                zs, xl = [], []
                for i in range(REPS):
                    for mode, out in (("zs", zs), ("xl", xl)):
                        try:
                            r = (model.inference_zero_shot(target, EOP + ptext, wav, stream=False)
                                 if mode == "zs" else
                                 model.inference_cross_lingual(EOP + target, wav, stream=False))
                            au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in list(r)])
                            out.append(round(len(au) / 24000, 2))
                            sf.write(os.path.join(OUT, f"{lang}__{llm[:9]}__{g}__{mode}_{i}.wav"), au, 24000)
                        except Exception:
                            out.append(None)
                def verdict(v):
                    good = [x for x in v if x]
                    if not good: return "ERR"
                    if all(x < exp * 0.4 for x in good): return "collapse"
                    if all(x > exp * 2.2 for x in good): return "overlong"
                    return "ok"
                results.append({"lang": lang, "llm": llm, "flow": flow, "ref": g,
                                "expected": round(exp, 2), "zero_shot": zs, "cross_lingual": xl,
                                "zs_verdict": verdict(zs), "xl_verdict": verdict(xl)})
                print(f"{llm.replace('_whole.pt',''):10}{g:9}"
                      f"{str(zs) + ' ' + verdict(zs):>22}{str(xl) + ' ' + verdict(xl):>22}", flush=True)
            del model
            torch.cuda.empty_cache()
            shutil.rmtree(d, ignore_errors=True)
            json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)

    print("\n=== languages with at least one usable configuration ===", flush=True)
    ok = {}
    for r in results:
        if r["zs_verdict"] == "ok" or r["xl_verdict"] == "ok":
            ok.setdefault(r["lang"], []).append(
                f"{r['llm'].replace('_whole.pt','')}/{r['ref']}"
                f"{'/zs' if r['zs_verdict']=='ok' else ''}{'/xl' if r['xl_verdict']=='ok' else ''}")
    for l in sorted(ok): print(f"  {l:13} {', '.join(ok[l][:6])}", flush=True)
    dead = sorted({r["lang"] for r in results} - set(ok))
    print("  NONE WORKING:", dead or "-", flush=True)


if __name__ == "__main__":
    main()
