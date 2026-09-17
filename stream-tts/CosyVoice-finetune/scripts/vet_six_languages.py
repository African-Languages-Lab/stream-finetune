"""Individual vs combined, six languages, both inference modes.

english, hausa, igbo, yoruba, twi, ewe
  x  individual (that language's own fine-tune) and combined (one multilingual model)
  x  male and female reference voice
  x  cross-lingual (no transcript) and zero-shot (transcript sent)
= 48 clips.

Best-CV llm and flow for each arm, original pretrained vocoder, stock CosyVoice3 calls with
default arguments -- n_timesteps stays at the upstream 10 and nothing is monkeypatched.

TARGET TEXT comes from each language's own kaldi dev set rather than being written by hand.
That guarantees the orthography matches what the model trained on, which matters: igbo clips
generated from plain-ASCII text mispronounced words the corpus spells with diacritics
(akwụkwọ appears 22,546 times against akwukwo 585).

REFERENCE VOICES are the vetted voice_selection/final pair for hausa, igbo, twi and ewe.
Two languages have none:
  english  has no reference voice at all, so it borrows the hausa clips. The voice carries,
           the language does not -- the accent you hear on english is that clip's.
  yoruba   is absent from voice_selection/final, so its pair was selected from fleurs_yo_ng
           in its own dev set: professionally read, full tone marks, unclipped, and split by
           estimated F0 (148.6 Hz vs 171.0 Hz). NOT from Waxal_NLP_Yoruba, whose short
           code-mixed exercise sentences previously produced a near-silent yoruba sample.

Note the yo-NG dev set is contaminated: 306 of its 829 entries are english afrispeech-200
clips. Everything here is filtered to the yoruba-sourced audio.
"""
import json
import os
import re
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
KALDI = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data"
FINAL = os.path.join(FT, "voice_selection", "final")
OUT = "/leonardo_scratch/large/userexternal/atsado00/vet_six"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"

# language -> (regime code, voice_selection key or None, kaldi regime for target text)
LANGS = {
    "english": ("english", None,     "en-UG"),
    "hausa":   ("ha-NG",   "hausa",  "ha-NG"),
    "igbo":    ("ig-NG",   "igbo",   "ig-NG"),
    "yoruba":  ("yo-NG",   None,     "yo-NG"),
    "twi":     ("tw-GH",   "twi",    "tw-GH"),
    "ewe":     ("ee-GH",   "ewe",    "ee-GH"),
}
VOICE_FALLBACK = {"english": "hausa"}     # english borrows the hausa reference clips

# yoruba pair chosen from fleurs_yo_ng by level and estimated F0 (see module docstring)
YORUBA_VOICES = {
    "male":   "3cc1e7c3bfc3d528",
    "female": "abd023477606ed24",
}
# sources whose sentences are short code-mixed exercises, excluded as target text
BAD_SOURCES = ("Waxal_NLP",)

# Every dev set except ewe carries english afrispeech clips filed under the wrong language:
# yo-NG 306/829 (37%), ig-NG 206/708 (29%), tw-GH 195/1935 (10%), ha-NG 111/3077 (4%).
# Target text must come from the language's OWN corpus directory, or the model is asked to
# read english. On top of that, require the orthography the corpus actually uses -- an
# undiacriticised sentence is a minority spelling these models barely saw.
CORPUS_DIR = {"english": "english", "hausa": "hausa", "igbo": "igbo",
              "yoruba": "yoruba", "twi": "twi", "ewe": "ewe"}
ORTHOGRAPHY = {
    "igbo":   re.compile(r"[ịọụṅ]"),
    "yoruba": re.compile(r"[ẹọṣ].*[àáèéìíòóùú]|[àáèéìíòóùú].*[ẹọṣ]"),
    "twi":    re.compile(r"[ɛɔ]"),
    "ewe":    re.compile(r"[ɖƒɛɔŋʋ]"),
}


def best_ckpt(stage_dir):
    """Lowest-CV checkpoint that actually has a .pt on disk."""
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


def load_kaldi(code):
    d = f"{KALDI}/individual_{code}/dev"
    wav, txt = {}, {}
    for name, store in (("wav.scp", wav), ("text", txt)):
        p = f"{d}/{name}"
        if not os.path.exists(p):
            return {}, {}
        for line in open(p, errors="ignore"):
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                store[parts[0]] = parts[1]
    return wav, txt


def pick_target(code, lang, exclude_uids=(), exclude_texts=()):
    """A natural sentence from this language's own dev set, in its own orthography.

    exclude_texts matters because FLEURS repeats sentences across speakers -- the yoruba
    reference sentence appears 3 times under different ids, so filtering by id alone would
    hand the model the exact words its reference clip already says, making zero-shot trivial.
    """
    wav, txt = load_kaldi(code)
    want_dir = f"/{CORPUS_DIR[lang]}/"
    orth = ORTHOGRAPHY.get(lang)
    norm = lambda s: " ".join(s.lower().split())
    bad_texts = {norm(s) for s in exclude_texts if s}
    cands = []
    for uid, t in txt.items():
        if uid in exclude_uids or uid not in wav:
            continue
        path = wav[uid]
        if any(b in path for b in BAD_SOURCES):
            continue
        if want_dir not in path:
            continue          # reject clips filed under this language but sourced elsewhere
        if orth and not orth.search(t):
            continue          # reject undiacriticised text
        if norm(t) in bad_texts:
            continue          # same words as a reference clip, under a different id
        # en-UG's longest dev sentence is 88 chars, so the floor has to sit below that
        if 55 <= len(t) <= 220 and len(t.split()) >= 8:
            cands.append((uid, t))
    if not cands:
        return None
    # deterministic, and prefer a substantial sentence over a fragment
    cands.sort(key=lambda c: (abs(len(c[1]) - 150), c[0]))
    return cands[0][1]


def yoruba_voice(gender):
    wav, txt = load_kaldi("yo-NG")
    uid = YORUBA_VOICES[gender]
    return wav.get(uid), txt.get(uid, ""), uid


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
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/reference_voices", exist_ok=True)
    from cosyvoice.cli.cosyvoice import CosyVoice3

    prompts = json.load(open(os.path.join(FINAL, "prompts.json")))

    # ---- resolve voices and target text per language ----
    plan = {}
    for lang, (code, vkey, tcode) in LANGS.items():
        voices, exclude = {}, set()
        if lang == "yoruba":
            for g in ("male", "female"):
                w, t, uid = yoruba_voice(g)
                if w and os.path.exists(w):
                    voices[g] = {"wav": w, "text": t, "uid": uid}
                    exclude.add(uid)
        else:
            key = vkey or VOICE_FALLBACK[lang]
            for g in ("male", "female"):
                e = prompts[key][g]
                voices[g] = {"wav": os.path.join(FINAL, e["prompt_wav"]),
                             "text": (e.get("prompt_text") or "").strip(),
                             "uid": e["prompt_wav"]}
        target = pick_target(tcode, lang, exclude,
                             [v.get("text", "") for v in voices.values()])
        if not voices or not target:
            print(f"[{lang}] SKIP (voices={bool(voices)} target={bool(target)})", flush=True)
            continue
        plan[lang] = {"voices": voices, "target": target}
        print(f"[{lang}] target: {target[:100]}", flush=True)
        for g, v in voices.items():
            try:
                i = sf.info(v["wav"])
                print(f"    {g:6} {os.path.basename(v['wav'])} {i.duration:.2f}s {i.samplerate}Hz"
                      f" transcript={'yes' if v['text'] else 'NO'}", flush=True)
                shutil.copy2(v["wav"], f"{OUT}/reference_voices/{lang}__{g}.wav")
            except Exception as e:
                print(f"    {g}: cannot read reference ({e})", flush=True)

    # ---- build the arms ----
    arms = {}
    for lang, (code, _, _) in LANGS.items():
        if lang not in plan:
            continue
        base = f"{CK}/individual/individual_{code}"
        bl, bf = best_ckpt(f"{base}/llm"), best_ckpt(f"{base}/flow")
        if bl and bf:
            arms[f"individual_{lang}"] = (base, bl[2], bf[2], [lang], bl, bf)
    cbase = f"{CK}/combined/combined"
    cl, cf = best_ckpt(f"{cbase}/llm"), best_ckpt(f"{cbase}/flow")
    if cl and cf:
        arms["combined"] = (cbase, cl[2], cf[2], list(plan.keys()), cl, cf)

    manifest = []
    print(f"\n{'arm':22}{'lang':9}{'gender':8}{'mode':14}{'dur':>8}  file", flush=True)
    for arm, (base, llm, flow, langs, bl, bf) in arms.items():
        d = os.path.join(OUT, "work", arm)
        try:
            model = CosyVoice3(build(base, llm, flow, d), fp16=False)
        except Exception as e:
            print(f"{arm}: BUILD FAILED {str(e)[:90]}", flush=True)
            continue
        print(f"  [{arm}] llm={llm} (cv {bl[0]:.4f})  flow={flow} (cv {bf[0]:.4f})", flush=True)
        for lang in langs:
            target = plan[lang]["target"]
            for gender, v in plan[lang]["voices"].items():
                for mode in ("cross_lingual", "zero_shot"):
                    if mode == "zero_shot" and not v["text"]:
                        continue
                    name = f"{arm}__{lang}__{gender}__{mode}.wav"
                    try:
                        if mode == "cross_lingual":
                            r = model.inference_cross_lingual(EOP + target, v["wav"], stream=False)
                        else:
                            r = model.inference_zero_shot(target, EOP + v["text"], v["wav"],
                                                          stream=False)
                        au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy()
                                             for x in list(r)])
                        sf.write(os.path.join(OUT, name), au, 24000)
                        dur, peak = len(au) / 24000, float(np.abs(au).max())
                        print(f"{arm:22}{lang:9}{gender:8}{mode:14}{dur:7.2f}s  {name}", flush=True)
                        manifest.append({"arm": arm, "language": lang, "gender": gender,
                                         "mode": mode, "duration_sec": round(dur, 2),
                                         "peak": round(peak, 4), "file": name,
                                         "llm": llm, "llm_cv": bl[0], "llm_step": bl[1],
                                         "flow": flow, "flow_cv": bf[0], "flow_step": bf[1],
                                         "reference": os.path.basename(v["wav"]),
                                         "target_text": target})
                    except Exception as ex:
                        print(f"{arm:22}{lang:9}{gender:8}{mode:14}  ERROR {str(ex)[:55]}",
                              flush=True)
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(d, ignore_errors=True)

    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {len(manifest)} clips to {OUT}", flush=True)


if __name__ == "__main__":
    main()
