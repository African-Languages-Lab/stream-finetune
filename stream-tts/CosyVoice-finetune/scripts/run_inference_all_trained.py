"""
Generates one zero-shot-cloned speech sample for EVERY individual-regime language that has
real (non-init) checkpoints for all 3 CosyVoice3 stages (llm, flow, hifigan) -- not just the
4 previously bundled/pushed-to-HF languages. Assembles each regime's bundle on the fly
(shared static assets + that regime's own latest checkpoints, cleaned for inference) rather
than requiring a pre-built HF bundle dir.

Voice reference: a real training-data clip + its ground-truth transcript (from that regime's
own kaldi dev/ split), 8-15s long -- run_inference_test.py found sub-6s in-domain clips can
produce broken near-zero-length output, so this stays clear of that range.

Spoken text: one shared, freshly-authored (not from any training data) English sentence,
used as tts_text for every regime -- this is a genuine cross-lingual zero-shot generation
test (CosyVoice3's own advertised capability: clone a voice in language A, make it speak
language B), not resynthesis of something the model could have memorized.
"""
import glob
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

CKPT_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual"
KALDI_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/kaldi_data"
PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_all"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_all_trained"

TTS_TEXT = ("Bright mornings in the highlands make the market square come alive with color "
            "and sound.")

STATIC_ASSETS = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN"]


def latest_ckpt(d):
    pts = [p for p in glob.glob(f"{d}/*.pt") if os.path.basename(p) != "init.pt"]
    if not pts:
        return None
    return max(pts, key=os.path.getmtime)


def discover_regimes():
    regimes = []
    for path in sorted(glob.glob(f"{CKPT_ROOT}/individual_*")):
        lang = os.path.basename(path)[len("individual_"):]
        stage_ckpts = {}
        for stage in ["llm", "flow", "hifigan"]:
            c = latest_ckpt(f"{path}/{stage}")
            stage_ckpts[stage] = c
        if all(stage_ckpts.values()):
            regimes.append((lang, stage_ckpts))
    return regimes


def clean_llm_or_flow(raw_path, out_path):
    sd = torch.load(raw_path, map_location="cpu")
    cleaned = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    torch.save(cleaned, out_path)


def clean_hifigan(raw_path, out_path):
    sd = torch.load(raw_path, map_location="cpu")
    if any(k.startswith("generator.") for k in sd.keys()):
        cleaned = {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    else:
        cleaned = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    torch.save(cleaned, out_path)


def assemble_bundle(lang, stage_ckpts):
    bundle_dir = f"{WORK_ROOT}/{lang}"
    os.makedirs(bundle_dir, exist_ok=True)
    for asset in STATIC_ASSETS:
        dst = f"{bundle_dir}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean_llm_or_flow(stage_ckpts["llm"], f"{bundle_dir}/llm.pt")
    clean_llm_or_flow(stage_ckpts["flow"], f"{bundle_dir}/flow.pt")
    clean_hifigan(stage_ckpts["hifigan"], f"{bundle_dir}/hift.pt")
    return bundle_dir


def pick_reference_and_target(lang):
    """Picks two DIFFERENT real utterances from this language's own dev set: one as the
    voice-cloning prompt (audio + ground-truth transcript), one whose ground-truth transcript
    (only -- its audio is never used) becomes the spoken text. Both are real training-data
    text, both in-language -- this avoids two separate confounds found the first time round:
    resynthesis (same utterance for prompt and target) proving nothing but playback, and
    cross-lingual generation (an English target sentence against a model fine-tuned hard on a
    single non-English language) producing audio-shaped noise instead of real speech."""
    text_path = f"{KALDI_ROOT}/individual_{lang}/dev/text"
    wav_path = f"{KALDI_ROOT}/individual_{lang}/dev/wav.scp"
    texts = {}
    with open(text_path) as f:
        for line in f:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]
    wavs = {}
    with open(wav_path) as f:
        for line in f:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) == 2:
                wavs[parts[0]] = parts[1]
    import soundfile as sf

    candidates = []
    for uid, txt in texts.items():
        if uid not in wavs or not os.path.exists(wavs[uid]) or not txt.strip():
            continue
        try:
            dur = sf.info(wavs[uid]).duration
        except Exception:
            continue
        candidates.append((uid, wavs[uid], txt, dur))

    in_range = [c for c in candidates if 8.0 <= c[3] <= 15.0]
    pool = in_range if len(in_range) >= 2 else [c for c in candidates if c[3] >= 6.0]
    if len(pool) < 2:
        return None, None, None
    prompt_uid, prompt_wav, prompt_text, _ = pool[0]
    target_uid, _, target_text, _ = pool[1]
    return prompt_wav, prompt_text, target_text


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    regimes = discover_regimes()
    print(f"=== {len(regimes)} regimes with all 3 stages trained: {[r[0] for r in regimes]} ===")

    results = []
    for lang, stage_ckpts in regimes:
        print(f"\n=== {lang} ===")
        prompt_wav, prompt_text, target_text = pick_reference_and_target(lang)
        if prompt_wav is None:
            print(f"  SKIP: fewer than 2 usable dev-set utterances found")
            results.append((lang, "SKIP", "no reference clip"))
            continue
        try:
            bundle_dir = assemble_bundle(lang, stage_ckpts)
            model = CosyVoice3(bundle_dir, fp16=False)
            full_prompt_text = prompt_text + "<|endofprompt|>"
            out_path = f"{OUT_DIR}/{lang}.wav"
            res = list(model.inference_zero_shot(target_text, full_prompt_text, prompt_wav, stream=False))
            audio = res[0]["tts_speech"]
            torchaudio.save(out_path, audio, model.sample_rate)
            dur = audio.shape[1] / model.sample_rate
            peak = audio.abs().max().item()
            print(f"  prompt={prompt_wav}")
            print(f"  prompt_text={prompt_text!r}")
            print(f"  target_text={target_text!r}")
            print(f"  -> {out_path} ({dur:.2f}s, peak={peak:.4f})")
            results.append((lang, "OK", f"{dur:.2f}s peak={peak:.4f}"))
            del model
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append((lang, "FAILED", str(e)))

    print("\n=== SUMMARY ===")
    for lang, status, info in results:
        print(f"  {lang:10s} {status:8s} {info}")


if __name__ == "__main__":
    main()
