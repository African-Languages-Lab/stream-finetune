"""Igbo with proper diacritics, against the ASCII set already generated.

The igbo training corpus is overwhelmingly written in proper Igbo orthography, but every
igbo clip generated so far used a plain-ASCII target text. Counts over the first 300k
training lines:

    akwụkwọ 22,546 vs akwukwo   585      39:1
    ụmụaka   3,054 vs umuaka     87      35:1
    ọganihu    128 vs oganihu     4      32:1
    ọbụla    1,839 vs obula     175      10:1
    kwesịrị  1,416 vs kwesiri   204       7:1
    nwanyị   3,393 vs nwanyi    553       6:1

So the ASCII spellings are a rare minority form as far as this model is concerned, which is
the classic setup for words coming out nearly-but-not-quite right. This generates the same
grid with the diacritics restored so the two can be compared directly.

Everything else is held constant: same checkpoints, same reference voices, same solver step
counts, both inference modes. Only the spelling of the target text changes.

Note for zero-shot specifically: the igbo reference transcript already carries diacritics, so
these clips are the first where the transcript and the target text share one orthography.
Previously the model saw two conventions in a single sequence.

Files are labelled igbo-dia so they sit beside their igbo counterparts and collide with
nothing already uploaded.
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
STEPS = [int(s) for s in os.environ.get("VET_STEPS", "10,20,50").split(",")]

BUILDS = {
    "individual_igbo": (f"{CK}/individual/individual_ig-NG",
                        "epoch_2_whole.pt", "epoch_31_whole.pt"),
    "combined":        (f"{CK}/combined/combined",
                        "epoch_11_whole.pt", "epoch_15_step_320000.pt"),
}

# the ASCII text used everywhere so far, for reference:
#   "Ndewo, kedu ka i mere? Enwere m olileanya na i di mma taa. Agumakwukwo bu isi ihe
#    na-eweta oganihu na obodo obula. Umuaka nwoke na nwanyi kwesiri iga akwukwo."
TEXT_DIA = ("Ndewo, kedu ka ị mere? Enwere m olileanya na ị dị mma taa. Agụmakwụkwọ bụ isi "
            "ihe na-eweta ọganihu na obodo ọbụla. Ụmụaka nwoke na nwanyị kwesịrị ịga akwụkwọ.")


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

    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))["igbo"]
    manifest = []
    print(f"text: {TEXT_DIA}\n", flush=True)
    print(f"{'arm':20}{'gender':8}{'mode':14}{'steps':>6}{'dur':>8}  file", flush=True)

    for arm, (base, llm, flow) in BUILDS.items():
        d = os.path.join(OUT, "work_dia", arm)
        try:
            model = CosyVoice3(build(base, llm, flow, d), fp16=False)
        except Exception as e:
            print(f"{arm}: BUILD FAILED {str(e)[:90]}", flush=True)
            continue
        for gender in ("male", "female"):
            wav = os.path.join(FINAL, prompts[gender]["prompt_wav"])
            transcript = (prompts[gender].get("prompt_text") or "").strip()
            for n in STEPS:
                set_steps(model, n)
                for mode in ("cross_lingual", "zero_shot"):
                    if mode == "zero_shot" and not transcript:
                        continue
                    tag = "" if mode == "cross_lingual" else "__zeroshot"
                    name = f"{arm}__igbo-dia__{gender}{tag}__nts{n}.wav"
                    try:
                        if mode == "cross_lingual":
                            r = model.inference_cross_lingual(EOP + TEXT_DIA, wav, stream=False)
                        else:
                            r = model.inference_zero_shot(TEXT_DIA, EOP + transcript, wav,
                                                          stream=False)
                        au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy()
                                             for x in list(r)])
                        sf.write(os.path.join(OUT, name), au, 24000)
                        dur = len(au) / 24000
                        print(f"{arm:20}{gender:8}{mode:14}{n:>6}{dur:7.2f}s  {name}", flush=True)
                        manifest.append({"arm": arm, "language": "igbo", "orthography": "diacritics",
                                         "gender": gender, "mode": mode, "n_timesteps": n,
                                         "duration_sec": round(dur, 2), "file": name,
                                         "llm": llm, "flow": flow, "text": TEXT_DIA})
                    except Exception as ex:
                        print(f"{arm:20}{gender:8}{mode:14}{n:>6}   ERROR {str(ex)[:55]}", flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)

    json.dump(manifest, open(os.path.join(OUT, "manifest_igbo_diacritics.json"), "w"), indent=1)
    print(f"\nwrote {len(manifest)} diacritic clips to {OUT}", flush=True)


if __name__ == "__main__":
    main()
