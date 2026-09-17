"""Does chichewa/male work with NO reference transcript?

The 9x4 zero-shot matrix produced one usable cell out of 36, and it was unstable. The pattern
was entirely reference-driven -- the deployed clip leaked at every epoch, the replacement
clips collapsed at every epoch -- which is the signature of a transcript that does not match
its audio, the same fault that once collapsed twi to 0.04s.

Cross-lingual passes NO transcript, so if the transcripts are the fault these same clips
should suddenly behave. That also happens to be a shippable fix: the endpoint already routes
to cross-lingual whenever prompt_text is empty, so a bad transcript can be dropped rather than
corrected.

Zero-shot is included alongside as the control, on the same clips in the same run.
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
OUT = "/leonardo_scratch/large/userexternal/atsado00/chichewa_xl"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"
REPS = 3


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build(llm, flow, d):
    os.makedirs(d, exist_ok=True)
    clean(f"{CK}/llm/{llm}", f"{d}/llm.pt")
    clean(f"{CK}/flow/{flow}", f"{d}/flow.pt")
    for a in SHARED:
        shutil.copy2(f"{PRE}/{a}", f"{d}/{a}")
    if not os.path.exists(f"{d}/CosyVoice-BlankEN"):
        shutil.copytree(f"{PRE}/CosyVoice-BlankEN", f"{d}/CosyVoice-BlankEN")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3

    pr = json.load(open(os.path.join(FINAL, "prompts.json")))["chichewa"]
    cands = json.load(open("/leonardo_scratch/large/userexternal/atsado00/fixrefs/chichewa_male_cands.json"))
    refs = [("deployed_5.3s", os.path.join(FINAL, pr["male"]["prompt_wav"]), pr["male"]["prompt_text"])]
    for f, e in sorted(cands.items()):
        refs.append((f.replace(".wav", ""), os.path.join(RELEN, f), e["transcript"]))
    # the female clip is the control -- it already works in zero-shot
    refs.append(("female_CONTROL", os.path.join(FINAL, pr["female"]["prompt_wav"]),
                 pr["female"]["prompt_text"]))

    target = PROBES["chichewa"]
    results = []
    for llm, flow in (("epoch_0_whole.pt", "epoch_8_whole.pt"), ("epoch_4_whole.pt", "epoch_8_whole.pt")):
        model = CosyVoice3(build(llm, flow, os.path.join(OUT, "work")), fp16=False)
        print(f"\n=== {llm} + {flow} ===", flush=True)
        print(f"{'reference':20}{'zero-shot (excess)':>28}{'cross-lingual (dur)':>28}", flush=True)
        for name, wav, ptext in refs:
            ref_dur = sf.info(wav).duration
            expected = len(target) / (len(ptext) / ref_dur)
            zs, xl = [], []
            for i in range(REPS):
                try:
                    r = list(model.inference_zero_shot(target, EOP + ptext, wav, stream=False))
                    a = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in r])
                    zs.append(round((len(a) / 24000 - expected) / ref_dur, 2))
                except Exception:
                    zs.append(None)
                try:
                    r = list(model.inference_cross_lingual(EOP + target, wav, stream=False))
                    a = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in r])
                    xl.append(round(len(a) / 24000, 2))
                    sf.write(os.path.join(OUT, f"{llm[:7]}__{name}__xl_{i}.wav"), a, 24000)
                except Exception:
                    xl.append(None)
            good = [v for v in xl if v]
            verdict = ("ERR" if not good else
                       "collapse" if all(v < expected * 0.4 for v in good) else "ok")
            results.append({"llm": llm, "ref": name, "zero_shot": zs, "cross_lingual": xl,
                            "expected": round(expected, 2), "xl_verdict": verdict})
            print(f"{name:20}{str(zs):>28}{str(xl) + ' ' + verdict:>28}", flush=True)
        del model
        torch.cuda.empty_cache()
        json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    print(f"\n(expected duration for this text is about "
          f"{results[0]['expected']:.1f}s against the deployed clip)", flush=True)


if __name__ == "__main__":
    main()
