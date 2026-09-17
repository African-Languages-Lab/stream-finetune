"""The igbo half of the six-language set again, with the diacritics stripped.

Everything is held identical to the igbo clips already generated -- same checkpoints, same
reference voices, same sentence, same modes, stock hyperparameters. The ONLY change is the
spelling of the target text:

  igbo        Ụfọdụ ihe omume enwere ike ime ma anọrọ ya gụnyere ịchụ nta, ịkụ azụ, ...
  igbo-plain  Ufodu ihe omume enwere ike ime ma anoro ya gunyere ichu nta, iku azu, ...

19 characters differ. The stripping is a plain Unicode NFD decomposition with combining marks
removed, so ị->i, ọ->o, ụ->u; no words are otherwise altered.

Why it is worth hearing: the igbo corpus is written in proper orthography, and the ASCII forms
are a rare minority spelling as far as the model is concerned -- akwụkwọ appears 22,546 times
against akwukwo 585, ụmụaka 3,054 against umuaka 87. If the plain clips mispronounce where the
diacritic ones do not, the fix is a text-normalisation step at the endpoint rather than
anything in the model.

Files are labelled igbo-plain so they sit beside their igbo counterparts and overwrite nothing.
"""
import json
import os
import shutil
import sys
import unicodedata as ud

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
OUT = "/leonardo_scratch/large/userexternal/atsado00/vet_six"   # same dir, new files
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"


def strip_diacritics(s):
    return "".join(c for c in ud.normalize("NFD", s) if not ud.combining(c))


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


def main():
    from cosyvoice.cli.cosyvoice import CosyVoice3

    # reuse exactly what the igbo clips in this folder were generated from
    manifest = json.load(open(f"{OUT}/manifest.json"))
    igbo = [e for e in manifest if e["language"] == "igbo"]
    if not igbo:
        raise SystemExit("no igbo entries in manifest.json; run vet_six_languages.py first")
    target_dia = igbo[0]["target_text"]
    target = strip_diacritics(target_dia)
    print(f"diacritics : {target_dia}\nplain      : {target}", flush=True)

    arms = {}
    for e in igbo:
        arms.setdefault(e["arm"], (e["llm"], e["flow"]))
    base_for = {a: (f"{CK}/combined/combined" if a == "combined"
                    else f"{CK}/individual/individual_ig-NG") for a in arms}

    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))["igbo"]
    out = []
    print(f"\n{'arm':20}{'gender':8}{'mode':14}{'dur':>8}  file", flush=True)
    for arm, (llm, flow) in arms.items():
        d = os.path.join(OUT, "work_plain", arm)
        try:
            model = CosyVoice3(build(base_for[arm], llm, flow, d), fp16=False)
        except Exception as e:
            print(f"{arm}: BUILD FAILED {str(e)[:90]}", flush=True)
            continue
        for gender in ("male", "female"):
            entry = prompts[gender]
            wav = os.path.join(FINAL, entry["prompt_wav"])
            transcript = (entry.get("prompt_text") or "").strip()
            for mode in ("cross_lingual", "zero_shot"):
                if mode == "zero_shot" and not transcript:
                    continue
                name = f"{arm}__igbo-plain__{gender}__{mode}.wav"
                try:
                    if mode == "cross_lingual":
                        r = model.inference_cross_lingual(EOP + target, wav, stream=False)
                    else:
                        r = model.inference_zero_shot(target, EOP + transcript, wav, stream=False)
                    au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy()
                                         for x in list(r)])
                    sf.write(os.path.join(OUT, name), au, 24000)
                    dur, peak = len(au) / 24000, float(np.abs(au).max())
                    print(f"{arm:20}{gender:8}{mode:14}{dur:7.2f}s  {name}", flush=True)
                    out.append({"arm": arm, "language": "igbo", "orthography": "plain",
                                "gender": gender, "mode": mode,
                                "duration_sec": round(dur, 2), "peak": round(peak, 4),
                                "file": name, "llm": llm, "flow": flow,
                                "target_text": target, "target_text_diacritics": target_dia})
                except Exception as ex:
                    print(f"{arm:20}{gender:8}{mode:14}  ERROR {str(ex)[:55]}", flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)

    json.dump(out, open(f"{OUT}/manifest_igbo_plain.json", "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {len(out)} plain-orthography igbo clips to {OUT}", flush=True)


if __name__ == "__main__":
    main()
