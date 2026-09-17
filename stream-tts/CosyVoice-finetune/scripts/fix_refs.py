"""Replace the two references that still fail, and find out WHY umbundu crashes.

chichewa/male leaks its reference on every retry, and umbundu/female does not merely produce
bad audio -- it raises "Calculated padded input size per channel: (3)", a convolution error
that means the LLM emitted about three frames of speech tokens. That is a collapse severe
enough to break the vocoder rather than just sound wrong.

For chichewa the job is a straight bake-off: three tier-1 candidates against the deployed
5.3s clip, several runs each, scored by unexplained duration -- the same leak metric used
across all 48 voices.

For umbundu the question is first WHERE the fault lies, so the deployed female clip is tried
against the deployed male clip on the same model and the same text. If the male reference is
fine, the clip is at fault and a replacement fixes it; if both collapse, the model is at fault
and no reference will help.
"""
import json
import os
import shutil
import sys
import traceback

REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))
FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(FT, "scripts"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from try_new_models import PROBES, build  # noqa: E402

EOP = "You are a helpful assistant.<|endofprompt|>"
OUT = "/leonardo_scratch/large/userexternal/atsado00/fixrefs/out"
WORK = "/leonardo_scratch/large/userexternal/atsado00/fixrefs/work"
FINAL = os.path.join(FT, "voice_selection", "final")
RELEN = os.path.join(FT, "voice_selection", "relen_wavs")
ISO = {"chichewa": "ny-MW", "umbundu": "umb-AO"}
REPS = 3


def run(model, target, ptext, wav, ref_dur, tag):
    """One generation; returns (duration, excess-in-reference-lengths, peak) or an error string."""
    expected = len(target) / (len(ptext) / ref_dur)
    try:
        res = list(model.inference_zero_shot(target, EOP + ptext, wav, stream=False))
        au = np.concatenate([r["tts_speech"].squeeze(0).cpu().numpy() for r in res])
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {str(e)[:90]}", expected
    dur = len(au) / 24000
    pk = float(np.abs(au).max()) if au.size else 0.0
    sf.write(os.path.join(OUT, f"{tag}.wav"), au, 24000)
    return dur, (dur - expected) / ref_dur, pk, None, expected


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3

    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))
    results = {}

    # ---------- chichewa: bake off the candidates against the deployed clip ----------
    print("=== chichewa/male: candidates vs the deployed 5.3s clip ===", flush=True)
    cands = json.load(open("/leonardo_scratch/large/userexternal/atsado00/fixrefs/chichewa_male_cands.json"))
    dep = prompts["chichewa"]["male"]
    trials = [("DEPLOYED_5.3s", os.path.join(FINAL, dep["prompt_wav"]), dep["prompt_text"])]
    for f, e in sorted(cands.items()):
        trials.append((f.replace(".wav", ""), os.path.join(RELEN, f), e["transcript"]))

    model = CosyVoice3(build(ISO["chichewa"], WORK), fp16=False)
    target = PROBES["chichewa"]
    print(f"{'reference':22}{'ref':>6}{'exp':>7}{'runs (dur / excess)':>34}", flush=True)
    ch = {}
    for name, wav, ptext in trials:
        ref_dur = sf.info(wav).duration
        rows = [run(model, target, ptext, wav, ref_dur, f"chichewa_{name}_{i}") for i in range(REPS)]
        ok = [r for r in rows if r[0] is not None]
        cells = "  ".join(f"{r[0]:.1f}s/{r[1]:+.2f}" if r[0] is not None else "ERR" for r in rows)
        exp = rows[0][4]
        worst = max((r[1] for r in ok), default=None)
        ch[name] = {"ref_dur": round(ref_dur, 1), "expected": round(exp, 1),
                    "runs": [{"dur": r[0], "excess": r[1], "peak": r[2], "error": r[3]} for r in rows],
                    "worst_excess": worst}
        print(f"{name:22}{ref_dur:5.1f}s{exp:7.1f}{cells:>34}", flush=True)
    results["chichewa"] = ch
    del model
    torch.cuda.empty_cache()

    # ---------- umbundu: is it the clip, or the model? ----------
    print("\n=== umbundu: deployed female clip vs deployed male clip, same model & text ===", flush=True)
    model = CosyVoice3(build(ISO["umbundu"], WORK), fp16=False)
    target = PROBES["umbundu"]
    umb = {}
    for g in ("female", "male"):
        e = prompts["umbundu"][g]
        wav = os.path.join(FINAL, e["prompt_wav"])
        ref_dur = sf.info(wav).duration
        rows = [run(model, target, e["prompt_text"], wav, ref_dur, f"umbundu_{g}_{i}") for i in range(REPS)]
        cells = "  ".join(f"{r[0]:.1f}s/{r[1]:+.2f}" if r[0] is not None else f"ERR({r[3][:34]})" for r in rows)
        umb[g] = {"ref_dur": round(ref_dur, 1),
                  "runs": [{"dur": r[0], "excess": r[1], "peak": r[2], "error": r[3]} for r in rows]}
        print(f"  umbundu/{g:8} ref={ref_dur:5.1f}s  {cells}", flush=True)
    results["umbundu"] = umb

    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    print("\nwritten", f"{OUT}/results.json", flush=True)


if __name__ == "__main__":
    main()
