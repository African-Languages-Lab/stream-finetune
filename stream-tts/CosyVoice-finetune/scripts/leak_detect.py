"""Measure, for every deployed reference, whether the model reads the reference aloud.

Spot reports are not enough: the user found hausa/female, oromo/male, lingala/female and
ewe/female by ear, and says there are more. So detect it mechanically over all 48.

Two independent signals, because either alone can mislead:

1. UNEXPLAINED DURATION. Speaking rate is calibrated per voice from that voice's OWN
   reference (prompt chars / reference seconds) rather than a global constant, since these
   languages differ hugely in characters per second (5.5 for luganda, 15.4 for tsonga).
   The expected output length for the target text follows from that rate. A leak speaks the
   reference first, so the excess lands at roughly one whole reference duration.

2. ACOUSTIC RE-UTTERANCE. If the reference is spoken, the start of the output is that same
   sentence in that same voice, so subsequence-DTW of the reference MFCCs against the output
   scores far better than chance. Duration alone cannot tell a leak from a model that simply
   rambles; this can.

Length was my earlier hypothesis and it is wrong -- ewe/female leaks at 19.1s, the second
longest reference of the 48 -- so nothing here filters on duration.
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
FT = os.path.dirname(HERE)
REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))
sys.path.insert(0, HERE)

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from try_new_models import PROBES, build  # noqa: E402

# checkpoint directories still carry the old ISO codes; the serving names are plain
ISO = {
    "hausa": "ha-NG", "twi": "tw-GH", "igbo": "ig-NG", "ewe": "ee-GH", "berber": "ber-MA",
    "umbundu": "umb-AO", "amharic": "am-ET", "arabic": "ar-AR", "fula": "ff-SN",
    "luganda": "lg-UG", "lingala": "ln-CD", "malagasy": "mg-MG", "sepedi": "nso-ZA",
    "chichewa": "ny-MW", "oromo": "or-KE", "somali": "so-SO", "sesotho": "st-ZA",
    "swahili": "sw-KE", "tigrinya": "ti-ER", "tswana": "tn-BW", "tsonga": "ts-ZA",
    "venda": "ve-ZA", "xhosa": "xh-ZA", "zulu": "zu-ZA",
}
FINAL = os.path.join(FT, "voice_selection", "final")
REPS = 2

# The decisive comparison. Every official CosyVoice3 example passes the marker as the END OF A
# SYSTEM PREAMBLE, with the reference transcript AFTER it:
#     inference_zero_shot(tts_text, "You are a helpful assistant.<|endofprompt|>" + transcript, wav)
# Production appends it instead, putting the transcript BEFORE the marker. In llm.py the two
# text halves are concatenated (text = cat[prompt_text, text]) and the marker is the only thing
# telling the model which span the reference speech tokens are aligned to -- so placement decides
# whether the transcript is treated as already-spoken or as text still owed. Both forms satisfy
# the EOP_TOKEN_ID assert, so only synthesis can say which is right.
# The fine-tuning parquet's `text` column holds bare transcripts -- no marker, no preamble --
# so training taught the model to speak EVERY text token it is handed. That is why feeding it
# [transcript][EOP][target] yields transcript-then-target: it is doing what it was trained to do.
# Cross-lingual passes no transcript at all, so there is nothing left to leak.
PREAMBLE = "You are a helpful assistant.<|endofprompt|>"
FORMATS = {
    # (mode, prompt_text builder, tts_text builder)
    "appended": ("zero_shot", lambda t: t + "<|endofprompt|>", lambda x: x),
    "preamble": ("zero_shot", lambda t: PREAMBLE + t, lambda x: x),
    "crosslingual": ("cross_lingual", None, lambda x: PREAMBLE + x),
}


def mfcc(y, sr):
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=256)
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)


def reutterance(ref_y, out_y, sr):
    """Normalised subsequence-DTW cost of the reference against the output. Lower = the
    reference is present in the output. Returns None when the output is too short to hold it."""
    if len(out_y) < 0.6 * len(ref_y) or len(ref_y) < sr * 0.5:
        return None
    R, O = mfcc(ref_y, sr), mfcc(out_y, sr)
    if R.shape[1] < 8 or O.shape[1] < 8:
        return None
    try:
        D, _ = librosa.sequence.dtw(X=R, Y=O, subseq=True, metric="cosine")
    except Exception:
        return None
    return float(D[-1, :].min() / R.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wavdir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--langs", nargs="*")
    args = ap.parse_args()
    os.makedirs(args.wavdir, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)

    from cosyvoice.cli.cosyvoice import CosyVoice3

    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}

    for lang in (args.langs or sorted(prompts)):
        if lang in res:
            print(f"{lang}: done", flush=True)
            continue
        if lang not in ISO:
            print(f"{lang}: no checkpoint mapping", flush=True)
            continue
        target = PROBES.get(lang)
        if not target:
            print(f"{lang}: no probe text", flush=True)
            continue
        try:
            d = build(ISO[lang], args.workdir)
            model = CosyVoice3(d, fp16=False)
        except Exception as e:
            print(f"{lang}: build/load failed: {e}", flush=True)
            res[lang] = {"error": str(e)}
            json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
            continue

        out = {}
        for g in ("male", "female"):
            e = prompts[lang].get(g)
            if not e:
                continue
            wav_path = os.path.join(FINAL, e["prompt_wav"])
            ptext = e["prompt_text"]
            ref_y, ref_sr = librosa.load(wav_path, sr=16000, mono=True)
            ref_dur = len(ref_y) / ref_sr
            # this voice's own speaking rate, in characters per second
            rate = len(ptext) / ref_dur
            expected = len(target) / rate

            byfmt = {}
            for fname, (mode, build_prompt, build_text) in FORMATS.items():
                runs = []
                for i in range(REPS):
                    try:
                        if mode == "zero_shot":
                            r = list(model.inference_zero_shot(
                                build_text(target), build_prompt(ptext), wav_path, stream=False))
                        else:
                            r = list(model.inference_cross_lingual(
                                build_text(target), wav_path, stream=False))
                        au = r[0]["tts_speech"].squeeze(0).cpu().numpy()
                    except Exception as ex:
                        runs.append({"error": str(ex)[:200]})
                        continue
                    dur = len(au) / 24000
                    p = os.path.join(args.wavdir, f"{lang}__{g}__{fname}_{i}.wav")
                    sf.write(p, au, 24000)
                    oy = librosa.resample(au, orig_sr=24000, target_sr=16000)
                    dt = reutterance(ref_y, oy, 16000)
                    runs.append({
                        "dur": round(dur, 2),
                        "excess_refs": round((dur - expected) / ref_dur, 2),
                        "dtw": round(dt, 4) if dt is not None else None,
                        "wav": os.path.basename(p),
                    })
                good = [r for r in runs if "dur" in r]
                byfmt[fname] = {
                    "med_excess_refs": round(float(np.median([r["excess_refs"] for r in good])), 2) if good else None,
                    "med_dtw": round(float(np.median([r["dtw"] for r in good if r["dtw"] is not None])), 4)
                    if any(r["dtw"] is not None for r in good) else None,
                    "med_dur": round(float(np.median([r["dur"] for r in good])), 2) if good else None,
                    "runs": runs,
                }
            out[g] = {
                "ref_dur": round(ref_dur, 2), "ref_chars": len(ptext),
                "rate_cps": round(rate, 2), "target_chars": len(target),
                "expected_dur": round(expected, 2), "formats": byfmt,
            }
            bits = " | ".join(
                f"{f}: dur={v['med_dur']} excess={v['med_excess_refs']} dtw={v['med_dtw']}"
                for f, v in byfmt.items())
            print(f"  {lang:10} {g:6} ref={ref_dur:5.1f}s exp={expected:5.1f}s  {bits}", flush=True)

        res[lang] = out
        json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
        del model
        torch.cuda.empty_cache()

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
