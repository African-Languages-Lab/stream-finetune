"""Regenerate the yoruba FEMALE clips only, with a correctly-sexed reference.

The female reference shipped in the first run was not female. I picked the yoruba pair by
median F0 with a fixed 165 Hz threshold, but the fleurs_yo_ng pool is unimodal and male --
median 125 Hz, 57 of 60 clips below 165 Hz -- so the "female" slot was filled by the
highest-pitched male in a male-only set. Two mistakes compounded: YIN was run on
un-normalised audio with fmax=400, which invites octave-doubling on quiet 16 kHz recordings
and inflated the estimates, and I assumed the pool was mixed instead of checking that the
distribution actually had two modes.

The male reference is unchanged and its four clips are left alone.

  female  BEFORE  fleurs_yo_ng/6447513343752058905.wav   15.96 s  16 kHz  (male)
          AFTER   Naija_Voices_Yoruba a20c1d3dd042ab99    8.76 s  48 kHz  F0 188.5 Hz

The replacement is also a better reference on its own terms: 48 kHz against 24 kHz output
rather than 16 kHz upsampled, peak 0.822 with no clipping, and rms 0.185.

Everything else is held identical to the run being patched -- same checkpoints, same target
sentence read back from manifest.json, both inference modes, stock hyperparameters -- so only
four files change.
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
KALDI = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/individual_yo-NG"
OUT = "/leonardo_scratch/large/userexternal/atsado00/vet_six"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"

NEW_FEMALE_UID = "a20c1d3dd042ab99"


def load_kaldi(split):
    wav, txt = {}, {}
    for name, store in (("wav.scp", wav), ("text", txt)):
        p = f"{KALDI}/{split}/{name}"
        if not os.path.exists(p):
            continue
        for line in open(p, errors="ignore"):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                store[parts[0]] = parts[1]
    return wav, txt


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

    manifest = json.load(open(f"{OUT}/manifest.json"))
    yo = [e for e in manifest if e["language"] == "yoruba"]
    if not yo:
        raise SystemExit("no yoruba entries in manifest.json")
    target = yo[0]["target_text"]

    wav, txt = load_kaldi("train")
    ref_wav = wav.get(NEW_FEMALE_UID)
    ref_text = (txt.get(NEW_FEMALE_UID) or "").strip()
    if not ref_wav or not os.path.exists(ref_wav) or not ref_text:
        raise SystemExit(f"reference {NEW_FEMALE_UID} not resolvable")
    info = sf.info(ref_wav)
    print(f"new female reference: {NEW_FEMALE_UID}  {info.duration:.2f}s {info.samplerate}Hz")
    print(f"  transcript: {ref_text}")
    print(f"  target    : {target}")
    print(f"  ratio target/prompt = {len(target)/len(ref_text):.2f} (needs > 0.5 for zero-shot)\n")
    shutil.copy2(ref_wav, f"{OUT}/reference_voices/yoruba__female.wav")

    # exactly the checkpoints the run being patched used, per arm
    arms = {}
    for e in yo:
        arms.setdefault(e["arm"], (e["llm"], e["flow"]))
    base_for = {a: (f"{CK}/combined/combined" if a == "combined"
                    else f"{CK}/individual/individual_yo-NG") for a in arms}

    out = []
    print(f"{'arm':22}{'mode':16}{'dur':>8}  file", flush=True)
    for arm, (llm, flow) in arms.items():
        d = os.path.join(OUT, "work_yofix", arm)
        try:
            model = CosyVoice3(build(base_for[arm], llm, flow, d), fp16=False)
        except Exception as e:
            print(f"{arm}: BUILD FAILED {str(e)[:90]}", flush=True)
            continue
        for mode in ("cross_lingual", "zero_shot"):
            name = f"{arm}__yoruba__female__{mode}.wav"   # same names: replaces in place
            try:
                if mode == "cross_lingual":
                    r = model.inference_cross_lingual(EOP + target, ref_wav, stream=False)
                else:
                    r = model.inference_zero_shot(target, EOP + ref_text, ref_wav, stream=False)
                au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in list(r)])
                sf.write(os.path.join(OUT, name), au, 24000)
                dur, peak = len(au) / 24000, float(np.abs(au).max())
                print(f"{arm:22}{mode:16}{dur:7.2f}s  {name}", flush=True)
                out.append({"arm": arm, "language": "yoruba", "gender": "female", "mode": mode,
                            "duration_sec": round(dur, 2), "peak": round(peak, 4), "file": name,
                            "reference_uid": NEW_FEMALE_UID,
                            "reference": os.path.basename(ref_wav),
                            "reference_sr": info.samplerate,
                            "reference_f0_hz": 188.5,
                            "prompt_text": ref_text, "target_text": target,
                            "llm": llm, "flow": flow,
                            "note": "replaces the fleurs_yo_ng clip that was male"})
            except Exception as ex:
                print(f"{arm:22}{mode:16}  ERROR {str(ex)[:60]}", flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)

    json.dump(out, open(f"{OUT}/manifest_yoruba_female_fix.json", "w"), indent=1,
              ensure_ascii=False)
    print(f"\nwrote {len(out)} corrected yoruba female clips", flush=True)


if __name__ == "__main__":
    main()
