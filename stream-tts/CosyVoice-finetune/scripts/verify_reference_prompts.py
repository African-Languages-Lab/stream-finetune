"""Verify reference prompts by actually synthesizing with them, before they are shipped.

Acoustic scoring (SNR, noise floor, clipping) cannot detect the fault that broke Twi: a clip
whose transcript does not match its audio. CosyVoice3 degenerates to a fraction of a second
when the reference text and reference audio disagree, and that is invisible to every signal
measurement -- it only shows up when you synthesize.

So: load each language's model once, run every candidate prompt through it on real sentences,
and keep only the prompts that reliably produce speech of a plausible length. Prompts are
tested several times because generation is stochastic and collapses intermittently; a prompt
that collapses on any run is rejected rather than averaged.

Usage:
    python verify_reference_prompts.py --candidates cands.json --out verified.json [--langs hausa twi]

candidates.json: {"<language>": [{"prompt_wav": "...", "prompt_text": "...", "tag": "..."} , ...]}
"""
import argparse
import json
import os
import sys
import time
import traceback

REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# short, ordinary sentences -- long enough that a collapse is unambiguous
PROBES = {
    "hausa": ["Sannu da safe, yaya kake? Ina fatan kana lafiya a yau.",
              "Muna farin cikin ganin ku a nan yau."],
    "twi": ["Maakye, wo ho te sɛn nnɛ? Mewɔ anidasoɔ sɛ wo ho yɛ.",
            "Yɛn ani agye sɛ wobaa ha nnɛ."],
    "igbo": ["Ndewo, kedu ka i mere? Enwere m olileanya na i di mma taa.",
             "Obi di anyi uto na i biara ebe a taa."],
    "swahili": ["Habari ya asubuhi, hujambo? Natumaini u mzima leo.",
                "Tunafurahi kukuona hapa leo."],
    "zulu": ["Sawubona, unjani namuhla? Ngiyethemba ukuthi uphilile.",
             "Siyajabula ukukubona lapha namuhla."],
}
GENERIC = ["Test sentence for reference verification, spoken clearly and at a normal pace.",
           "A second test sentence to confirm the reference voice behaves consistently."]

RUNS = 3
MIN_SEC_PER_WORD = 0.14      # a real reading is well above this
ABS_MIN_SEC = 1.0


def probes_for(lang):
    return PROBES.get(lang, GENERIC)


def verify_language(lang, cands, repo_fmt="all-lab/cosyvoice3-individual-{}", token=None):
    from huggingface_hub import snapshot_download

    from cosyvoice.cli.cosyvoice import CosyVoice3

    repo = repo_fmt.format(lang)
    print(f"\n=== {lang} ({repo}) ===", flush=True)
    local = snapshot_download(repo_id=repo, token=token,
                              ignore_patterns=["cosyvoice/*", "third_party/*",
                                               "handler.py", "requirements.txt"])
    model = CosyVoice3(local, fp16=False)
    sr = int(model.sample_rate)
    texts = probes_for(lang)

    results = []
    for c in cands:
        wav, text, tag = c["prompt_wav"], c["prompt_text"], c.get("tag", "")
        if not os.path.exists(wav):
            print(f"  [{tag}] MISSING {wav}", flush=True)
            continue
        prompt_text = text.strip()
        if not prompt_text.endswith("<|endofprompt|>"):
            prompt_text += "<|endofprompt|>"

        durations, failures = [], 0
        for t in texts:
            need = max(ABS_MIN_SEC, MIN_SEC_PER_WORD * len(t.split()) * 0.5)
            for _ in range(RUNS):
                try:
                    out = list(model.inference_zero_shot(t, prompt_text, wav, stream=False))
                    a = out[0]["tts_speech"]
                    d = a.shape[-1] / sr
                    pk = float(a.abs().max())
                    durations.append(round(d, 2))
                    if d < need or pk < 0.02:
                        failures += 1
                except Exception as e:
                    print(f"    error: {type(e).__name__}: {e}", flush=True)
                    failures += 1
        ok = failures == 0 and durations
        results.append({**c, "durations": durations, "failures": failures, "verified": bool(ok)})
        print(f"  [{tag or os.path.basename(wav)}] durations={durations} failures={failures} "
              f"-> {'PASS' if ok else 'REJECT'}", flush=True)

    del model
    gc_cuda()
    return results


def gc_cuda():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", nargs="*")
    args = ap.parse_args()

    cands = json.load(open(args.candidates))
    langs = args.langs or list(cands)
    token = os.environ.get("HF_TOKEN_ORGOWNER") or os.environ.get("HF_TOKEN")

    out = {}
    if os.path.exists(args.out):
        try:
            out = json.load(open(args.out))
        except Exception:
            out = {}

    for lang in langs:
        if lang in out:
            print(f"{lang}: already verified, skipping", flush=True)
            continue
        try:
            out[lang] = verify_language(lang, cands[lang], token=token)
        except Exception:
            traceback.print_exc()
            out[lang] = []
        json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)

    print("\n================ SUMMARY ================", flush=True)
    for lang, rs in out.items():
        good = [r for r in rs if r["verified"]]
        print(f"  {lang:10} {len(good)}/{len(rs)} prompts verified", flush=True)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
