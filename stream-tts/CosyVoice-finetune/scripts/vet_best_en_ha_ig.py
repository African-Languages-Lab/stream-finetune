"""Vetting set: individual vs combined, on the BEST llm+flow of each, for en/ha/ig.

Two things are being compared at once, so every clip differs from its partner in exactly
one variable:

  arm         individual (that language's own fine-tune) vs combined (one multilingual model)
  n_timesteps 10 (what production runs today) vs 50 (the proposed fix)

n_timesteps is the flow-matching ODE solver's step count. It is hardcoded to 10 at
cosyvoice/flow/flow.py:140,276,409 -- a latency default, not a quality one. It is passed as a
keyword into decoder(...), so it can be overridden at runtime by wrapping the decoder's
forward rather than editing the vendored source.

Checkpoints are the lowest-CV of each stage, not the newest -- for igbo the newest llm is 2.1x
its own minimum. The vocoder is the ORIGINAL pretrained hift.pt in every case, matching how
serving works.

Generation is cross-lingual (no reference transcript). That is the path the endpoint uses for
a transcript-free voice, and it avoids the instruct-slot mismatch that collapses zero-shot.

English has no reference voice of its own in voice_selection/final, so English text is spoken
with the hausa reference clip -- the voice carries, the language does not.
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
STEPS = [int(s) for s in os.environ.get("VET_STEPS", "10,50").split(",")]

# lowest-CV checkpoint for each stage, chosen over the newest deliberately
BUILDS = {
    "individual_english": (f"{CK}/individual/individual_english",
                           "epoch_1_step_70000.pt", "epoch_7_step_270000.pt", ["english"]),
    "individual_hausa":   (f"{CK}/individual/individual_ha-NG",
                           "epoch_1_step_50000.pt", "epoch_8_step_220000.pt", ["hausa"]),
    "individual_igbo":    (f"{CK}/individual/individual_ig-NG",
                           "epoch_2_whole.pt", "epoch_31_whole.pt", ["igbo"]),
    "combined":           (f"{CK}/combined/combined",
                           "epoch_11_whole.pt", "epoch_15_step_320000.pt",
                           ["english", "hausa", "igbo"]),
}

TEXT = {
    "english": ("Good morning everyone, and thank you for joining us today. Before we begin, I "
                "would like to remind you that the meeting is being recorded, and the slides "
                "will be shared with everyone afterwards."),
    "hausa":   ("Sannu da safe, yaya kake? Ina fatan kana lafiya a yau. Ilimi shine mabudin ci "
                "gaba a kowace kasa ta duniya. Yara maza da mata su tafi makaranta domin su koyi "
                "karatu da rubutu."),
    "igbo":    ("Ndewo, kedu ka i mere? Enwere m olileanya na i di mma taa. Agumakwukwo bu isi "
                "ihe na-eweta oganihu na obodo obula. Umuaka nwoke na nwanyi kwesiri iga akwukwo."),
}
# whose voice speaks each language; english borrows the hausa clip
REF = {"english": "hausa", "hausa": "hausa", "igbo": "igbo"}


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
    """Force the flow-matching solver step count, overriding the hardcoded 10."""
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
        raise RuntimeError("could not locate the flow decoder to set n_timesteps")
    if not hasattr(dec, "_orig_forward"):
        dec._orig_forward = dec.forward
    def forward(*args, **kw):
        kw["n_timesteps"] = n
        return dec._orig_forward(*args, **kw)
    dec.forward = forward
    return dec


def main():
    os.makedirs(OUT, exist_ok=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3

    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))
    manifest = []
    print(f"{'arm':20}{'lang':9}{'gender':8}{'steps':>6}{'dur':>8}  file", flush=True)

    for arm, (base, llm, flow, langs) in BUILDS.items():
        d = os.path.join(OUT, "work", arm)
        try:
            model = CosyVoice3(build(base, llm, flow, d), fp16=False)
        except Exception as e:
            print(f"{arm}: BUILD FAILED {str(e)[:90]}", flush=True)
            continue
        for lang in langs:
            entry = prompts[REF[lang]]
            for gender in ("male", "female"):
                wav = os.path.join(FINAL, entry[gender]["prompt_wav"])
                for n in STEPS:
                    set_steps(model, n)
                    name = f"{arm}__{lang}__{gender}__nts{n}.wav"
                    try:
                        r = list(model.inference_cross_lingual(EOP + TEXT[lang], wav, stream=False))
                        au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in r])
                        sf.write(os.path.join(OUT, name), au, 24000)
                        dur = len(au) / 24000
                        print(f"{arm:20}{lang:9}{gender:8}{n:>6}{dur:7.2f}s  {name}", flush=True)
                        manifest.append({"arm": arm, "language": lang, "gender": gender,
                                         "n_timesteps": n, "duration_sec": round(dur, 2),
                                         "file": name, "llm": llm, "flow": flow})
                    except Exception as ex:
                        print(f"{arm:20}{lang:9}{gender:8}{n:>6}   ERROR {str(ex)[:60]}", flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)

    json.dump(manifest, open(os.path.join(OUT, f"manifest_nts{'_'.join(map(str,STEPS))}.json"), "w"), indent=1)
    print(f"\nwrote {len(manifest)} clips to {OUT}", flush=True)


if __name__ == "__main__":
    main()
