"""Find a chichewa/male combination that does not speak its reference aloud.

Two partial sweeps have been run and neither settled it:
  - the deployed 5.3s clip against llm epochs 1/3/5/7 -- leaked on every one
  - three tier-1 replacement clips against llm epoch_0 -- collapsed on every one

Those share no cell, so the useful combination has never been tried: the NEW clips against
the BETTER epochs. This crosses every LLM epoch with every male reference to settle it.

Flow is fixed. The earlier sweep produced byte-identical numbers across flow epochs 2, 5 and
8, which makes sense -- output length is decided by how many speech tokens the LLM emits, and
flow only turns those tokens into audio. So flow cannot move this metric, and varying it here
would trade real coverage for none. Flow quality still needs ears; this measures the leak.
"""
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

PRE = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CK = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_ny-MW"
FINAL = os.path.join(FT, "voice_selection", "final")
RELEN = os.path.join(FT, "voice_selection", "relen_wavs")
OUT = "/leonardo_scratch/large/userexternal/atsado00/chichewa_male"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"
FLOW = "epoch_8_whole.pt"
REPS = 3


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build(llm, d):
    os.makedirs(d, exist_ok=True)
    clean(f"{CK}/llm/{llm}", f"{d}/llm.pt")
    clean(f"{CK}/flow/{FLOW}", f"{d}/flow.pt")
    for a in SHARED:
        shutil.copy2(f"{PRE}/{a}", f"{d}/{a}")
    if not os.path.exists(f"{d}/CosyVoice-BlankEN"):
        shutil.copytree(f"{PRE}/CosyVoice-BlankEN", f"{d}/CosyVoice-BlankEN")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    work = os.path.join(OUT, "work")
    from cosyvoice.cli.cosyvoice import CosyVoice3

    dep = json.load(open(os.path.join(FINAL, "prompts.json")))["chichewa"]["male"]
    cands = json.load(open("/leonardo_scratch/large/userexternal/atsado00/fixrefs/chichewa_male_cands.json"))
    refs = [("deployed_5.3s", os.path.join(FINAL, dep["prompt_wav"]), dep["prompt_text"])]
    for f, e in sorted(cands.items()):
        refs.append((f.replace(".wav", ""), os.path.join(RELEN, f), e["transcript"]))

    llms = [f"epoch_{i}_whole.pt" for i in range(9)]
    target = PROBES["chichewa"]
    results = []
    print(f"chichewa/male -- {len(llms)} llm epochs x {len(refs)} references, flow fixed at {FLOW}")
    print(f"target ({len(target)} chars): {target}\n")
    hdr = f"{'llm':10}" + "".join(f"{r[0][:17]:>19}" for r in refs)
    print(hdr); print("-" * len(hdr), flush=True)

    for llm in llms:
        d = os.path.join(work, llm.replace(".pt", ""))
        try:
            model = CosyVoice3(build(llm, d), fp16=False)
        except Exception as e:
            print(f"{llm[:9]:10}  BUILD FAILED {str(e)[:60]}", flush=True)
            continue
        cells = []
        for name, wav, ptext in refs:
            ref_dur = sf.info(wav).duration
            expected = len(target) / (len(ptext) / ref_dur)
            ex = []
            for i in range(REPS):
                try:
                    res = list(model.inference_zero_shot(target, EOP + ptext, wav, stream=False))
                    au = np.concatenate([r["tts_speech"].squeeze(0).cpu().numpy() for r in res])
                    dur = len(au) / 24000
                    ex.append(round((dur - expected) / ref_dur, 2))
                    sf.write(os.path.join(OUT, f"{llm.replace('.pt','')}__{name}_{i}.wav"), au, 24000)
                except Exception:
                    ex.append(None)
            good = [e for e in ex if e is not None]
            verdict = ("ERR" if not good else
                       "collapse" if all(e < -0.4 for e in good) else
                       "LEAK" if any(e > 0.35 for e in good) else "ok")
            results.append({"llm": llm, "ref": name, "excess": ex, "verdict": verdict,
                            "expected": round(expected, 2)})
            cells.append(f"{verdict}{max(good) if good else 0:+.2f}")
        print(f"{llm.replace('_whole.pt',''):10}" + "".join(f"{c:>19}" for c in cells), flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)
        json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)

    ok = [r for r in results if r["verdict"] == "ok"]
    print(f"\n=== {len(ok)} of {len(results)} combinations clean ===")
    for r in ok:
        print(f"   {r['llm']} + {r['ref']}   excess {r['excess']}", flush=True)


if __name__ == "__main__":
    main()
