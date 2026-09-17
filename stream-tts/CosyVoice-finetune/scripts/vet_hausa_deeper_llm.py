"""Hausa with deeper LLM checkpoints, to test vocabulary coverage.

CV loss is dominated by frequent tokens, so a checkpoint can keep essentially the same CV
while having seen far more of the vocabulary. Hausa never diverged -- its whole 28-checkpoint
range spans 3.3441 to 3.4642, under 4% -- so much deeper checkpoints are available cheaply:

  INDIVIDUAL (best flow epoch_8_step_220000 held constant)
     49,999  cv 3.3441  acc 0.2745   epoch_1_step_50000    <- the current pick
     97,674  cv 3.3611  acc 0.2724   epoch_3_whole          +0.51%,  2.0x the steps
    180,165  cv 3.4069  acc 0.2699   epoch_6_whole          +1.88%,  3.6x the steps
    207,695  cv 3.4642  acc 0.2658   epoch_7_whole          +3.59%,  4.2x the steps

  COMBINED (best flow epoch_15_step_320000 held constant)
    252,778  cv 3.5902  acc 0.2366   epoch_11_whole         <- the current pick
    269,999  cv 3.5915  acc 0.2365   epoch_12_step_270000   +0.04%, the newest

TWO TEXTS, because Hausa has an orthography question of its own. The corpus is written with
the hooked letters (ɓ 4,167 / ɗ 6,967 / ƙ 10,160 occurrences) but is ALSO entirely lowercase
with zero punctuation across 200,000 sampled lines -- while every request so far has sent
capitalised, punctuated text.

  vocab  hooked letters, normal capitalisation and punctuation
  norm   the same words lowercased with punctuation stripped, matching training exactly

Every content word was verified present in the training corpus before use: buƙatar 4,876,
ɓata 368, tuƙuru 231, ɗalibai 181, ƙauye 79, ƙasarmu 84, ƙarfin 196, matasa 2,079.

Cross-lingual only, so no reference transcript confounds the comparison. nts20 throughout.
"""
import json
import os
import shutil
import sys

REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))
FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

PRE = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CK = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints"
FINAL = os.path.join(FT, "voice_selection", "final")
OUT = "/leonardo_scratch/large/userexternal/atsado00/vet_en_ha_ig"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"
STEPS = [int(s) for s in os.environ.get("VET_STEPS", "20").split(",")]

IND = f"{CK}/individual/individual_ha-NG"
COMB = f"{CK}/combined/combined"

# label -> (base, llm, flow)
BUILDS = {
    "individual-llm50k":  (IND,  "epoch_1_step_50000.pt", "epoch_8_step_220000.pt"),
    "individual-llm98k":  (IND,  "epoch_3_whole.pt",      "epoch_8_step_220000.pt"),
    "individual-llm180k": (IND,  "epoch_6_whole.pt",      "epoch_8_step_220000.pt"),
    "individual-llm208k": (IND,  "epoch_7_whole.pt",      "epoch_8_step_220000.pt"),
    "combined-llm253k":   (COMB, "epoch_11_whole.pt",     "epoch_15_step_320000.pt"),
    "combined-llm270k":   (COMB, "epoch_12_step_270000.pt", "epoch_15_step_320000.pt"),
}

VOCAB = ("Ɗalibai da yawa suna zuwa makarantar ƙauye don neman ilimi. Ƙasarmu tana buƙatar "
         "ƙarfin matasa domin gina gobe mai kyau. Kada mu ɓata lokaci, sai mu yi aiki tuƙuru.")
# same words, lowercased and unpunctuated -- exactly how the training text is written
NORM = ("ɗalibai da yawa suna zuwa makarantar ƙauye don neman ilimi ƙasarmu tana buƙatar "
        "ƙarfin matasa domin gina gobe mai kyau kada mu ɓata lokaci sai mu yi aiki tuƙuru")
TEXTS = {"vocab": VOCAB, "norm": NORM}


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build(base, llm, flow, d):
    os.makedirs(d, exist_ok=True)
    clean(f"{base}/llm/{llm}", f"{d}/llm.pt")
    clean(f"{base}/flow/{flow}", f"{d}/flow.pt")
    for a in SHARED:
        shutil.copy2(f"{PRE}/{a}", f"{d}/{a}")
    if not os.path.exists(f"{d}/CosyVoice-BlankEN"):
        shutil.copytree(f"{PRE}/CosyVoice-BlankEN", f"{d}/CosyVoice-BlankEN")
    return d


def set_steps(model, n):
    dec = None
    for path in ("model.flow.decoder", "flow.decoder"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            dec = obj
            break
        except AttributeError:
            continue
    if dec is None:
        raise RuntimeError("could not locate the flow decoder")
    if not hasattr(dec, "_orig_forward"):
        dec._orig_forward = dec.forward
    def forward(*args, **kw):
        kw["n_timesteps"] = n
        return dec._orig_forward(*args, **kw)
    dec.forward = forward


def main():
    os.makedirs(OUT, exist_ok=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3

    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))["hausa"]
    manifest = []
    for k, v in TEXTS.items():
        print(f"[{k}] {v}", flush=True)
    print(f"\n{'build':22}{'text':7}{'gender':8}{'steps':>6}{'dur':>8}  file", flush=True)

    for label, (base, llm, flow) in BUILDS.items():
        d = os.path.join(OUT, "work_ha", label)
        try:
            model = CosyVoice3(build(base, llm, flow, d), fp16=False)
        except Exception as e:
            print(f"{label}: BUILD FAILED {str(e)[:90]}", flush=True)
            continue
        for gender in ("male", "female"):
            wav = os.path.join(FINAL, prompts[gender]["prompt_wav"])
            for tkey, text in TEXTS.items():
                for n in STEPS:
                    set_steps(model, n)
                    name = f"{label}__hausa-{tkey}__{gender}__nts{n}.wav"
                    try:
                        r = list(model.inference_cross_lingual(EOP + text, wav, stream=False))
                        au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in r])
                        sf.write(os.path.join(OUT, name), au, 24000)
                        dur = len(au) / 24000
                        print(f"{label:22}{tkey:7}{gender:8}{n:>6}{dur:7.2f}s  {name}", flush=True)
                        manifest.append({"build": label, "llm": llm, "flow": flow,
                                         "text_variant": tkey, "text": text, "gender": gender,
                                         "n_timesteps": n, "duration_sec": round(dur, 2),
                                         "file": name})
                    except Exception as ex:
                        print(f"{label:22}{tkey:7}{gender:8}{n:>6}   ERROR {str(ex)[:50]}", flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)

    json.dump(manifest, open(os.path.join(OUT, "manifest_hausa_deeper_llm.json"), "w"), indent=1)
    print(f"\nwrote {len(manifest)} hausa clips to {OUT}", flush=True)


if __name__ == "__main__":
    main()
