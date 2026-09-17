"""Sweep LLM x flow checkpoints for one language, against its deployed reference clips.

Written after two languages were judged on a single automatic checkpoint pick and both
verdicts turned out to be wrong in opposite directions:

  yoruba   -- condemned as "collapses to under a second". There is no tensorboard history for
              yo-NG so the selector fell back to "newest epoch", which was a diverged one.
              Earlier epochs work: 19 of 24 generations usable.
  chichewa -- the selector picked llm epoch_0 (CV/loss 3.5468, the FIRST epoch). Its reference
              leaks on every run and three tier-1 replacement clips collapse, so the clip was
              never the problem.

A later epoch is not automatically a better epoch and neither is the lowest logged CV loss.
The only reliable check is to synthesize and listen, so this makes that cheap to do.

Scored by unexplained duration in reference-lengths -- the same leak metric used across all
48 deployed voices, with expected length taken from each reference's own speaking rate.
"""
import argparse
import json
import os
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

PRETRAINED = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CKPT_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"
FINAL = os.path.join(FT, "voice_selection", "final")
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"
ISO = {
    "hausa": "ha-NG", "twi": "tw-GH", "igbo": "ig-NG", "ewe": "ee-GH", "berber": "ber-MA",
    "umbundu": "umb-AO", "amharic": "am-ET", "arabic": "ar-AR", "fula": "ff-SN",
    "luganda": "lg-UG", "lingala": "ln-CD", "malagasy": "mg-MG", "sepedi": "nso-ZA",
    "chichewa": "ny-MW", "oromo": "or-KE", "somali": "so-SO", "sesotho": "st-ZA",
    "swahili": "sw-KE", "tigrinya": "ti-ER", "tswana": "tn-BW", "tsonga": "ts-ZA",
    "venda": "ve-ZA", "xhosa": "xh-ZA", "zulu": "zu-ZA", "yoruba": "yo-NG",
    "afrikaans": "af-ZA", "bemba": "bem-ZM", "kikuyu": "ki-KE", "ndebele": "nd-ZW",
    "kinyarwanda": "rw-RW", "shona": "sn-ZW", "wolof": "wo-SN",
}


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build(iso, llm, flow, d):
    os.makedirs(d, exist_ok=True)
    clean(f"{CKPT_ROOT}/individual_{iso}/llm/{llm}", f"{d}/llm.pt")
    clean(f"{CKPT_ROOT}/individual_{iso}/flow/{flow}", f"{d}/flow.pt")
    for a in SHARED:                    # hift.pt = the ORIGINAL pretrained vocoder
        shutil.copy2(f"{PRETRAINED}/{a}", f"{d}/{a}")
    if not os.path.exists(f"{d}/CosyVoice-BlankEN"):
        shutil.copytree(f"{PRETRAINED}/CosyVoice-BlankEN", f"{d}/CosyVoice-BlankEN")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--llms", required=True, help="comma-separated checkpoint filenames")
    ap.add_argument("--flows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "work")
    os.makedirs(work, exist_ok=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3

    iso = ISO[a.lang]
    target = PROBES[a.lang]
    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))[a.lang]
    refs = {}
    for g in ("male", "female"):
        e = prompts.get(g)
        if e:
            wav = os.path.join(FINAL, e["prompt_wav"])
            refs[g] = (wav, e["prompt_text"], sf.info(wav).duration)

    results = []
    print(f"=== {a.lang}: {len(a.llms.split(','))} llm x {len(a.flows.split(','))} flow, "
          f"original vocoder ===", flush=True)
    print(f"{'llm':18}{'flow':18}{'ref':8}{'runs (dur / excess)':>30}", flush=True)
    for llm in a.llms.split(","):
        for flow in a.flows.split(","):
            tag = f"{llm.replace('.pt','')}__{flow.replace('.pt','')}"
            d = os.path.join(work, tag)
            try:
                model = CosyVoice3(build(iso, llm, flow, d), fp16=False)
            except Exception as e:
                print(f"{llm[:17]:18}{flow[:17]:18}{'-':8}  BUILD FAILED {str(e)[:50]}", flush=True)
                continue
            for g, (wav, ptext, ref_dur) in refs.items():
                expected = len(target) / (len(ptext) / ref_dur)
                cells, runs = [], []
                for i in range(a.reps):
                    try:
                        res = list(model.inference_zero_shot(target, EOP + ptext, wav, stream=False))
                        au = np.concatenate([r["tts_speech"].squeeze(0).cpu().numpy() for r in res])
                        dur = len(au) / 24000
                        ex = (dur - expected) / ref_dur
                        sf.write(os.path.join(a.out, f"{tag}__{g}_{i}.wav"), au, 24000)
                        cells.append(f"{dur:.1f}s/{ex:+.2f}")
                        runs.append({"dur": round(dur, 2), "excess": round(ex, 2)})
                    except Exception as e:
                        cells.append(f"ERR({type(e).__name__})")
                        runs.append({"error": str(e)[:120]})
                good = [r for r in runs if "dur" in r]
                verdict = ("ALL_FAILED" if not good else
                           "COLLAPSE" if all(r["dur"] < expected * 0.4 for r in good) else
                           "LEAK" if all(r["excess"] > 0.35 for r in good) else
                           "MIXED" if any(r["dur"] < expected * 0.4 or r["excess"] > 0.35
                                          for r in good) else "OK")
                results.append({"llm": llm, "flow": flow, "ref": g, "verdict": verdict,
                                "expected": round(expected, 2), "runs": runs})
                print(f"{llm[:17]:18}{flow[:17]:18}{g:8}{'  '.join(cells):>30}  {verdict}", flush=True)
            del model
            torch.cuda.empty_cache()
            shutil.rmtree(d, ignore_errors=True)
            json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=2)

    ok = [r for r in results if r["verdict"] == "OK"]
    print(f"\n=== {len(ok)} of {len(results)} clean ===", flush=True)
    for r in ok:
        print(f"  {r['llm']} + {r['flow']} [{r['ref']}]", flush=True)


if __name__ == "__main__":
    main()
