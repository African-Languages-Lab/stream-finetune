"""A native-sounding chichewa male voice, with a transcript so it runs zero-shot.

Background. chichewa/male is the only one of the 48 preset voices with no transcript, removed
deliberately in 83c84aa ("new clip, and no transcript so it runs cross-lingual") because its
transcript did not match its audio. That made it the single voice the endpoint serves through
inference_cross_lingual -- on POST / and POST /stream alike, since both branch on the same
`mode` flag.

Cross-lingual is now known to be the worse path for naturalness. frontend_cross_lingual is
literally frontend_zero_shot with the LLM's conditioning deleted:

    del model_input['prompt_text']                # the reference words
    del model_input['llm_prompt_speech_token']    # the reference speech tokens

The voice still carries, through flow_prompt_speech_token and the speaker embedding, but the
LLM loses its demonstration of how the language actually sounds -- which is why cross-lingual
output was judged to sound like a non-native speaker struggling.

So the fix for chichewa/male is not to keep it transcript-free; it is to give it a clip whose
transcript is trustworthy. This candidate comes from zambezi_voice_nyanja in the ny-MW dev
set, with the transcript taken from the corpus's own text field:

    89e06a43d82ac064   F0 159.9 Hz (male), 13.59 s, 16 kHz, peak 0.834, rms 0.048
    "bambo wina wovala malaya amizeremizere ndi magalasi a maso akukambirana pa telefoni
     ali m chipinda chake"

Chosen over the fleurs_ny_mw clips, which are all recorded far too quietly -- their loudest is
peak 0.117, and F0 estimation fails on them entirely.

Both modes are rendered so the pair can be compared directly: if zero-shot sounds native and
cross-lingual does not, the transcript should be restored to prompts.json and chichewa/male
stops being the odd one out.
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
import re  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

PRE = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CK = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints"
KALDI = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data/individual_ny-MW"
OUT = "/leonardo_scratch/large/userexternal/atsado00/vet_six"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"

REF_UID = "89e06a43d82ac064"


def load_kaldi(split="dev"):
    wav, txt = {}, {}
    for name, store in (("wav.scp", wav), ("text", txt)):
        p = f"{KALDI}/{split}/{name}"
        if not os.path.exists(p):
            return {}, {}
        for line in open(p, errors="ignore"):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                store[parts[0]] = parts[1]
    return wav, txt


def best_ckpt(stage_dir):
    import glob
    rows = []
    for y in glob.glob(f"{stage_dir}/*.yaml"):
        try:
            t = open(y).read()
            st = int(re.search(r"^step:\s*(\d+)", t, re.M).group(1))
            ls = re.search(r"loss_dict:.*?\n\s+loss:\s*([\d.]+)", t, re.S)
            n = os.path.basename(y)[:-5]
            if ls and os.path.exists(os.path.join(stage_dir, n + ".pt")):
                rows.append((float(ls.group(1)), st, n + ".pt"))
        except Exception:
            pass
    return min(rows) if rows else None


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
    os.makedirs(f"{OUT}/reference_voices", exist_ok=True)

    wav, txt = load_kaldi()
    ref_wav, ref_text = wav.get(REF_UID), txt.get(REF_UID, "").strip()
    if not ref_wav or not os.path.exists(ref_wav) or not ref_text:
        raise SystemExit(f"reference {REF_UID} not resolvable")
    info = sf.info(ref_wav)
    print(f"reference : {REF_UID}  {info.duration:.2f}s {info.samplerate}Hz", flush=True)
    print(f"transcript: {ref_text}", flush=True)
    shutil.copy2(ref_wav, f"{OUT}/reference_voices/chichewa__male_new.wav")

    # a different sentence to speak, from the same corpus, long enough to clear the
    # len(text) < 0.5*len(prompt_text) guard that collapses zero-shot
    cands = [(u, t) for u, t in txt.items()
             if u != REF_UID and u in wav and 100 <= len(t) <= 220 and len(t.split()) >= 12]
    if not cands:
        raise SystemExit("no usable target sentence")
    cands.sort(key=lambda c: (abs(len(c[1]) - 150), c[0]))
    target = cands[0][1]
    print(f"target    : {target}", flush=True)
    print(f"            ratio target/prompt = {len(target)/len(ref_text):.2f} (needs > 0.5)\n",
          flush=True)

    arms = {}
    b = f"{CK}/individual/individual_ny-MW"
    bl, bf = best_ckpt(f"{b}/llm"), best_ckpt(f"{b}/flow")
    if bl and bf:
        arms["individual_chichewa"] = (b, bl, bf)
    cb = f"{CK}/combined/combined"
    cl, cf = best_ckpt(f"{cb}/llm"), best_ckpt(f"{cb}/flow")
    if cl and cf:
        arms["combined"] = (cb, cl, cf)

    out = []
    print(f"{'arm':22}{'mode':16}{'dur':>8}  file", flush=True)
    for arm, (base, bl, bf) in arms.items():
        d = os.path.join(OUT, "work_ny", arm)
        try:
            model = CosyVoice3(build(base, bl[2], bf[2], d), fp16=False)
        except Exception as e:
            print(f"{arm}: BUILD FAILED {str(e)[:90]}", flush=True)
            continue
        print(f"  [{arm}] llm={bl[2]} (cv {bl[0]:.4f})  flow={bf[2]} (cv {bf[0]:.4f})", flush=True)
        for mode in ("zero_shot", "cross_lingual"):
            name = f"{arm}__chichewa-newref__male__{mode}.wav"
            try:
                if mode == "zero_shot":
                    r = model.inference_zero_shot(target, EOP + ref_text, ref_wav, stream=False)
                else:
                    r = model.inference_cross_lingual(EOP + target, ref_wav, stream=False)
                au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in list(r)])
                sf.write(os.path.join(OUT, name), au, 24000)
                dur, peak = len(au) / 24000, float(np.abs(au).max())
                print(f"{arm:22}{mode:16}{dur:7.2f}s  {name}", flush=True)
                out.append({"arm": arm, "language": "chichewa", "gender": "male", "mode": mode,
                            "duration_sec": round(dur, 2), "peak": round(peak, 4), "file": name,
                            "reference_uid": REF_UID,
                            "reference_wav": ref_wav, "prompt_text": ref_text,
                            "target_text": target,
                            "llm": bl[2], "llm_cv": bl[0], "flow": bf[2], "flow_cv": bf[0]})
            except Exception as ex:
                print(f"{arm:22}{mode:16}  ERROR {str(ex)[:60]}", flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)

    json.dump(out, open(f"{OUT}/manifest_chichewa_newref.json", "w"), indent=1,
              ensure_ascii=False)
    print(f"\nwrote {len(out)} chichewa clips to {OUT}", flush=True)


if __name__ == "__main__":
    main()
