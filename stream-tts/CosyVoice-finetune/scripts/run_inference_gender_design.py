"""
Redo of the Hausa/Igbo generation demo: 2 real voice clones + 2 "voice design" variants per
language. CosyVoice3 has no cloning-free / parametric gender-pitch control (confirmed via code
inspection this session), so "voice design" here means: take the SAME reference clip used for
one of the clones, and layer a free-text instruct2 style instruction ("speak in a low-pitched
masculine voice" / "speak in a high-pitched feminine voice") on top of it via
inference_instruct2. This isolates whether the instruct mechanism itself has any real effect,
since voice_clone_N and the matching design file share the identical reference clip and only
differ in whether an instruct tag was applied.

Reference clips were picked from real (non-synthetic) datasets and pitch-classified with
librosa.pyin (median F0) to get one genuinely low-pitched and one genuinely high-pitched
speaker per language -- not hand-picked by assumption.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"

LANGS = {
    "hausa": {
        "bundle_dir": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_hausa_drive",
        "out_dir": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_hausa_gender_design",
        "best_llm": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_ha-NG/llm/epoch_1_whole.pt",
        "best_flow": None,  # no finetuned Hausa flow exists -> use original
        "target_text": "Yin rahoton abokin aiki na kan wata halayya mara kyau.<|endofprompt|>",
        "low_ref": {
            "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/hausa/audio/TWB-Voice-1.0 Hausa - hau/30333.wav",
            "text": "Binta ta shawo kan Kabiru ta canza shawara.<|endofprompt|>",
            "pitch_hz": 108.2,
        },
        "high_ref": {
            "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/hausa/audio/TWB-Voice-1.0 Hausa - hau/79150.wav",
            "text": "Amurka za ta fice daga Yarjejeniyar Bello.<|endofprompt|>",
            "pitch_hz": 212.0,
        },
    },
    "igbo": {
        "bundle_dir": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_igbo_drive",
        "out_dir": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_igbo_gender_design",
        "best_llm": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_ig-NG/llm/epoch_2_whole.pt",
        "best_flow": "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints_archived_20260818/individual/individual_ig-NG/flow/epoch_32_step_276000.pt",
        "target_text": "Mgbe onye si n'ụlọ ọrụ bịara, nkewa ụlọ ahụ enwuru ọkụ.<|endofprompt|>",
        "low_ref": {
            "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/Waxal_NLP_Igbo/ibo_1248.wav",
            "text": "Iri na otu agbakọnyere atọ pụtara iri na anọ.<|endofprompt|>",
            "pitch_hz": 112.0,
        },
        "high_ref": {
            "wav": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/fleurs_ig_ng/10036538191701311998.wav",
            "text": "Ndị Babịlọn rụrụ otu n'ime chi ha otu isi ụlọ nsọ nke e were ka ebe obibi chi ahụ.<|endofprompt|>",
            "pitch_hz": 194.9,
        },
    },
}

MALE_INSTRUCT = "Speak in a low-pitched, masculine voice.<|endofprompt|>"
FEMALE_INSTRUCT = "Speak in a high-pitched, feminine voice.<|endofprompt|>"


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def build_bundle(bundle_dir, best_llm, best_flow):
    os.makedirs(bundle_dir, exist_ok=True)
    assets = ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]
    if best_flow is None:
        assets.append("flow.pt")
    for asset in assets:
        dst = f"{bundle_dir}/{asset}"
        if not os.path.exists(dst):
            os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)
    clean(best_llm, f"{bundle_dir}/llm.pt")
    if best_flow is not None:
        clean(best_flow, f"{bundle_dir}/flow.pt")


def main():
    for lang, cfg in LANGS.items():
        print(f"\n=== {lang} ===", flush=True)
        os.makedirs(cfg["out_dir"], exist_ok=True)
        build_bundle(cfg["bundle_dir"], cfg["best_llm"], cfg["best_flow"])
        model = CosyVoice3(cfg["bundle_dir"], fp16=False)
        sr = model.sample_rate

        def save_with_retry_sr(gen_fn, out_path, max_attempts=4, min_dur=3.0):
            last = None
            for attempt in range(max_attempts):
                try:
                    results = list(gen_fn())
                    audio = results[0]["tts_speech"]
                    dur = audio.shape[1] / sr
                    peak = audio.abs().max().item()
                    if dur >= min_dur:
                        torchaudio.save(out_path, audio, sr)
                        print(f"  SAVED {out_path} ({dur:.2f}s, peak={peak:.4f}, attempt {attempt + 1})", flush=True)
                        return True
                    else:
                        print(f"  attempt {attempt + 1}: too short ({dur:.2f}s, peak={peak:.4f}), retrying", flush=True)
                except RuntimeError as e:
                    last = e
                    print(f"  attempt {attempt + 1} failed: {e}", flush=True)
            print(f"  FAILED after {max_attempts} attempts: {last}", flush=True)
            return False

        target = cfg["target_text"]
        low = cfg["low_ref"]
        high = cfg["high_ref"]

        print("-- voice_clone_1 (low-pitch reference, plain zero-shot) --", flush=True)
        save_with_retry_sr(
            lambda: model.inference_zero_shot(target, low["text"], low["wav"], stream=False),
            f"{cfg['out_dir']}/voice_clone_1.wav",
        )

        print("-- voice_clone_2 (high-pitch reference, plain zero-shot) --", flush=True)
        save_with_retry_sr(
            lambda: model.inference_zero_shot(target, high["text"], high["wav"], stream=False),
            f"{cfg['out_dir']}/voice_clone_2.wav",
        )

        print("-- male_low_pitch (same low-pitch reference + instruct2 masculine/low-pitch tag) --", flush=True)
        save_with_retry_sr(
            lambda: model.inference_instruct2(target, MALE_INSTRUCT, low["wav"], stream=False),
            f"{cfg['out_dir']}/male_low_pitch.wav",
        )

        print("-- female_high_pitch (same high-pitch reference + instruct2 feminine/high-pitch tag) --", flush=True)
        save_with_retry_sr(
            lambda: model.inference_instruct2(target, FEMALE_INSTRUCT, high["wav"], stream=False),
            f"{cfg['out_dir']}/female_high_pitch.wav",
        )

        del model
        torch.cuda.empty_cache()

    print("\nall done", flush=True)


if __name__ == "__main__":
    main()
