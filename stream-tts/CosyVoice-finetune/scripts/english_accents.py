"""Generate English from the COMBINED english model, cloning a different accent each time.

The combined model was trained on Ghanaian + Nigerian + Ugandan data pooled together
(731 shards: 724 GH, 3 NG, 4 UG). The open question is whether pooling produced one model
that can take on whichever accent its reference clip has, or an averaged accent that ignores
the reference. Cloning five distinctly-accented references and listening is how you tell.

The Arabic speaker is deliberately included as the out-of-domain case: that voice is not in
the English training data at all, so it tests whether the model carries over a speaker it has
never heard in this language.

Both modes are generated for every reference. Zero-shot uses the clip's own transcript;
cross-lingual passes none. That matters here because a corpus transcript that does not match
its audio collapses generation, and these English corpora have not been checked for that.
"""
import csv
import json
import os
import random
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")
REPO = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "third_party", "Matcha-TTS"))
FT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(FT, "scripts"))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
SO = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out"
PRE = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
CK = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_english"
FINAL = os.path.join(FT, "voice_selection", "final")
OUT = "/leonardo_scratch/large/userexternal/atsado00/english_accents"
SHARED = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "hift.pt"]
EOP = "You are a helpful assistant.<|endofprompt|>"

# "Accent Classification Dataset (Ghana)" is an empty directory -- the real Ghanaian audio is
# ghana-english-asr-2700hrs. The South African English corpus is 8 kHz throughout, below
# CosyVoice's 16 kHz floor, so every clip is correctly rejected; a South African speaker comes
# from the Afrikaans reference set instead. The language of a reference clip does not have to
# match the language being spoken -- that is the whole point of cloning, and it is also what
# the Arabic case tests.
ACCENTS = {
    "nigerian": "Crowdsourced high-quality Nigerian English speech data set",
    "ghanaian": "ghana-english-asr-2700hrs",
    "ugandan":  "salt",
}
# Long on purpose. A short line cannot show an accent -- vowel quality, rhythm and intonation
# only become audible over sustained speech, and the chunk joins that appear past ~80 tokens
# are themselves worth hearing.
SENTENCES = [
    "Good morning everyone, and thank you all for joining us here today. "
    "Before we begin, I would like to remind you that the meeting will be recorded, "
    "and a copy of the slides will be shared with everyone afterwards. "
    "We have three items on the agenda this morning. First, a short update on the "
    "progress we have made since the last review. Second, the budget for the coming "
    "quarter, which we will need to agree before the end of the week. And third, any "
    "other business that anyone would like to raise.",

    "The train to the city leaves at half past seven from platform four, so if you are "
    "planning to travel with us, please arrive at the station a little earlier than that. "
    "The journey takes about two hours and there is a short stop halfway, where you can "
    "buy tea, coffee and something to eat. Once we arrive, a bus will be waiting outside "
    "the main entrance to take everyone to the hotel. Please keep your ticket with you "
    "for the whole journey, because an inspector will check it on the way.",
]
LO, HI = 6.0, 20.0


def measure(path):
    """Same bar as the reference selector: real speech, quiet floor, one speaker."""
    try:
        if sf.info(path).samplerate < 16000:
            return None
        y, sr = librosa.load(path, sr=16000, mono=True)
    except Exception:
        return None
    dur = len(y) / sr
    if not (LO <= dur <= HI):
        return None
    peak = float(np.max(np.abs(y)))
    if peak < 0.05 or peak > 0.999:
        return None
    r = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    db = 20 * np.log10(np.maximum(r, 1e-9))
    noise, speech = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    if noise > -55 or (speech - noise) < 40:
        return None
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256)
    e = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][:len(f0)]
    v = (e > np.percentile(e, 55)) & (f0 > 63) & (f0 < 380)
    if v.sum() < 20:
        return None
    med = float(np.median(f0[v]))
    return {"dur": round(dur, 1), "noise": round(noise, 1), "snr": round(speech - noise, 1),
            "f0": round(med, 1), "gender": "male" if med < 155 else "female"}


def find_refs():
    """One male and one female clip per accent, straight from the corpus CSV."""
    want = {ds: acc for acc, ds in ACCENTS.items()}
    rows = {acc: [] for acc in ACCENTS}
    with open(f"{SO}/english/speech_transcribed_english.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ds = row.get("dataset_name", "")
            if ds not in want:
                continue
            t = (row.get("transcript") or "").strip()
            if len(t) < 25 or len(t.split()) < 6:
                continue
            rows[want[ds]].append((os.path.join(SO, "english", "audio", ds, row["audio_id"] + ".wav"), t, ds))
    out = {}
    for acc, cands in rows.items():
        random.Random(7).shuffle(cands)
        picked = {}
        for path, t, ds in cands[:400]:
            if len(picked) == 2:
                break
            if not os.path.exists(path):
                continue
            m = measure(path)
            if not m or m["gender"] in picked:
                continue
            picked[m["gender"]] = {"path": path, "transcript": t, "dataset": ds, "metrics": m}
        out[acc] = picked
        if picked:
            desc = ", ".join("{} {}s snr{}".format(g, v["metrics"]["dur"], v["metrics"]["snr"])
                             for g, v in picked.items())
        else:
            desc = "NONE FOUND"
        print(f"  {acc:15} {desc}", flush=True)
    # The Arabic speaker comes from the vetted serving references, and the South African one
    # from the Afrikaans candidates -- neither corpus offers usable English audio, and a
    # reference clip does not need to be in the language being spoken.
    ar = json.load(open(os.path.join(FINAL, "prompts.json")))["arabic"]
    out["arabic"] = {g: {"path": os.path.join(FINAL, ar[g]["prompt_wav"]),
                         "transcript": ar[g]["prompt_text"], "dataset": "vetted arabic reference",
                         "metrics": {"dur": round(sf.info(os.path.join(FINAL, ar[g]["prompt_wav"])).duration, 1)}}
                     for g in ("male", "female")}
    print(f"  {'arabic':15} male + female (vetted serving references)", flush=True)

    CANDS = "/leonardo_scratch/large/userexternal/atsado00/unserved_refs"
    try:
        af = json.load(open(f"{CANDS}/json/afrikaans.json"))["afrikaans"]
        sa = {}
        for g in ("male", "female"):
            lst = af.get(g) or []
            if lst:
                sa[g] = {"path": os.path.join(CANDS, "wavs", lst[0]["file"]),
                         "transcript": lst[0]["transcript"], "dataset": "afrikaans (South African speaker)",
                         "metrics": {"dur": round(sf.info(os.path.join(CANDS, "wavs", lst[0]["file"])).duration, 1)}}
        out["south_african"] = sa
        print(f"  {'south_african':15} {' + '.join(sa) or 'NONE'} (Afrikaans reference, 8kHz English corpus unusable)", flush=True)
    except Exception as e:
        print(f"  south_african   unavailable: {str(e)[:60]}", flush=True)
    return out


def build(d):
    os.makedirs(d, exist_ok=True)
    for stage, dst in (("llm", "llm.pt"), ("flow", "flow.pt")):
        src = f"{CK}/{stage}/epoch_1_step_70000.pt"
        sd = torch.load(src, map_location="cpu", weights_only=True)
        torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, f"{d}/{dst}")
    for a in SHARED:                    # hift.pt = ORIGINAL pretrained vocoder
        shutil.copy2(f"{PRE}/{a}", f"{d}/{a}")
    if not os.path.exists(f"{d}/CosyVoice-BlankEN"):
        shutil.copytree(f"{PRE}/CosyVoice-BlankEN", f"{d}/CosyVoice-BlankEN")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== picking one male and one female reference per accent ===", flush=True)
    refs = find_refs()
    json.dump({a: {g: {k: v for k, v in d.items() if k != "path"} for g, d in gs.items()}
               for a, gs in refs.items()},
              open(f"{OUT}/references.json", "w"), ensure_ascii=False, indent=2)

    from cosyvoice.cli.cosyvoice import CosyVoice3
    model = CosyVoice3(build(os.path.join(OUT, "work")), fp16=False)
    print("\n=== generating (combined english: llm+flow epoch_1_step_70000, original vocoder) ===",
          flush=True)
    print(f"{'accent':16}{'gender':8}{'mode':6}{'sent':6}{'dur':>7}", flush=True)
    for acc, gs in refs.items():
        for g, d in gs.items():
            for si, sent in enumerate(SENTENCES):
                for mode in ("zs", "xl"):
                    try:
                        r = (model.inference_zero_shot(sent, EOP + d["transcript"], d["path"], stream=False)
                             if mode == "zs" else
                             model.inference_cross_lingual(EOP + sent, d["path"], stream=False))
                        au = np.concatenate([x["tts_speech"].squeeze(0).cpu().numpy() for x in list(r)])
                        p = f"{OUT}/{acc}__{g}__{mode}__s{si+1}.wav"
                        sf.write(p, au, 24000)
                        print(f"{acc:16}{g:8}{mode:6}{si+1:<6}{len(au)/24000:7.2f}s", flush=True)
                    except Exception as e:
                        print(f"{acc:16}{g:8}{mode:6}{si+1:<6}  ERROR {str(e)[:50]}", flush=True)
    # copy the reference clips alongside so the accent can be compared against its source
    for acc, gs in refs.items():
        for g, d in gs.items():
            try:
                y, sr = librosa.load(d["path"], sr=24000, mono=True)
                sf.write(f"{OUT}/REFERENCE__{acc}__{g}.wav", y, 24000)
            except Exception:
                pass
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
