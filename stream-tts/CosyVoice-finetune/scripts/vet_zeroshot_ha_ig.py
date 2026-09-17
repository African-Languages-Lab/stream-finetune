"""Zero-shot counterparts to the cross-lingual vetting set, for hausa and igbo.

Same checkpoints, same reference voices, same text, same solver steps as
vet_best_en_ha_ig.py -- the ONLY thing that changes is the inference mode:

  cross-lingual  inference_cross_lingual(EOP + text, wav)          no transcript
  zero-shot      inference_zero_shot(text, EOP + transcript, wav)  transcript passed

That isolates the one difference between these clips and production. On the live endpoint
47 of 48 preset voices carry a transcript, so serving runs zero-shot for almost everything,
while the earlier vetting clips were all cross-lingual.

Zero-shot is expected to be worse: fine-tuning built its sequence as
    [sos] [instruct] [text] [task_id] [speech_token]
with a CONSTANT instruct string, whereas inference builds
    [sos] [prompt_text] [text] [task_id] [prompt_speech_token]
so a real sentence lands in a slot that only ever held one fixed string during training.
The point of these clips is to hear how much that costs, on the same models.

English is excluded: it has no reference voice of its own, so its "transcript" would be a
hausa sentence against english text -- a confound rather than a comparison.

Filenames carry the zeroshot marker so nothing collides with the cross-lingual set already
uploaded to Drive.
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
OUT = "/leonardo_scratch/large/userexternal/atsado00/vet_en_ha_ig"   # same dir, new files
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"
STEPS = [int(s) for s in os.environ.get("VET_STEPS", "10,50").split(",")]

BUILDS = {
    "individual_hausa": (f"{CK}/individual/individual_ha-NG",
                         "epoch_1_step_50000.pt", "epoch_8_step_220000.pt", ["hausa"]),
    "individual_igbo":  (f"{CK}/individual/individual_ig-NG",
                         "epoch_2_whole.pt", "epoch_31_whole.pt", ["igbo"]),
    "combined":         (f"{CK}/combined/combined",
                         "epoch_11_whole.pt", "epoch_15_step_320000.pt", ["hausa", "igbo"]),
}

TEXT = {
    "hausa": ("Sannu da safe, yaya kake? Ina fatan kana lafiya a yau. Ilimi shine mabudin ci "
              "gaba a kowace kasa ta duniya. Yara maza da mata su tafi makaranta domin su koyi "
              "karatu da rubutu."),
    "igbo":  ("Ndewo, kedu ka i mere? Enwere m olileanya na i di mma taa. Agumakwukwo bu isi "
              "ihe na-eweta oganihu na obodo obula. Umuaka nwoke na nwanyi kwesiri iga akwukwo."),
}


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

    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))
    manifest = []
    print(f"{'arm':20}{'lang':8}{'gender':8}{'steps':>6}{'dur':>8}  file", flush=True)

    for arm, (base, llm, flow, langs) in BUILDS.items():
        d = os.path.join(OUT, "work_zs", arm)
        try:
            model = CosyVoice3(build(base, llm, flow, d), fp16=False)
        except Exception as e:
            print(f"{arm}: BUILD FAILED {str(e)[:90]}", flush=True)
            continue
        for lang in langs:
            entry = prompts[lang]
            for gender in ("male", "female"):
                wav = os.path.join(FINAL, entry[gender]["prompt_wav"])
                transcript = (entry[gender].get("prompt_text") or "").strip()
                if not transcript:
                    print(f"{arm:20}{lang:8}{gender:8}  SKIP (no transcript)", flush=True)
                    continue
                for n in STEPS:
                    set_steps(model, n)
                    name = f"{arm}__{lang}__{gender}__zeroshot__nts{n}.wav"
                    try:
                        r = list(model.inference_zero_shot(TEXT[lang], EOP + transcript, wav,
                                                           stream=False))
                        au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in r])
                        sf.write(os.path.join(OUT, name), au, 24000)
                        dur = len(au) / 24000
                        print(f"{arm:20}{lang:8}{gender:8}{n:>6}{dur:7.2f}s  {name}", flush=True)
                        manifest.append({"arm": arm, "language": lang, "gender": gender,
                                         "mode": "zero_shot", "n_timesteps": n,
                                         "duration_sec": round(dur, 2), "file": name,
                                         "llm": llm, "flow": flow})
                    except Exception as ex:
                        print(f"{arm:20}{lang:8}{gender:8}{n:>6}   ERROR {str(ex)[:60]}", flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)

    json.dump(manifest, open(os.path.join(OUT, f"manifest_zeroshot_nts{'_'.join(map(str,STEPS))}.json"), "w"), indent=1)
    print(f"\nwrote {len(manifest)} zero-shot clips to {OUT}", flush=True)


if __name__ == "__main__":
    main()
