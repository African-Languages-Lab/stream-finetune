"""
Feeds a REAL speech file directly into the hifigan vocoder, bypassing llm and flow
entirely: real wav -> mel spectrogram (same feat_extractor used at training time) ->
HiFTGenerator.inference() -> waveform. No text, no tokens, no synthesis -- this is a
pure vocoder resynthesis test (mel-to-wav round trip on ground-truth mel), useful for
judging the vocoder in isolation from whatever the llm/flow stages are doing.

Runs the same real mel through two vocoders for comparison:
  - pretrained (original, non-finetuned) hift.pt
  - our finetuned yo-NG hifigan (highest checkpoint) -- yo-NG chosen because it has the
    most-trained finetuned hifigan of any regime at time of writing
"""
import os
import sys

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.utils.file_utils import load_wav

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_vocoder_direct"
os.makedirs(OUT_DIR, exist_ok=True)

REF_WAV = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-trial/reference-samples/yoruba_ref.wav"
FINETUNED_HIFIGAN = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_yo-NG/hifigan/epoch_84_whole.pt"

VARIANTS = {
    "pretrained": f"{PRETRAINED_DIR}/hift.pt",
    "finetuned_yo-NG": FINETUNED_HIFIGAN,
}


def clean_hifigan_state_dict(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if any(k.startswith("generator.") for k in sd.keys()):
        return {k[len("generator."):]: v for k, v in sd.items() if k.startswith("generator.")}
    return {k: v for k, v in sd.items() if torch.is_tensor(v)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== loading config from {PRETRAINED_DIR}/cosyvoice3.yaml ===", flush=True)
    with open(f"{PRETRAINED_DIR}/cosyvoice3.yaml", "r") as f:
        configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": f"{PRETRAINED_DIR}/CosyVoice-BlankEN"})
    feat_extractor = configs["feat_extractor"]

    print(f"=== loading + resampling {REF_WAV} to 24kHz ===", flush=True)
    speech = load_wav(REF_WAV, 24000)
    orig_out = f"{OUT_DIR}/original.wav"
    torchaudio.save(orig_out, speech, 24000)
    print(f"  original duration: {speech.shape[1] / 24000:.2f}s -> {orig_out}", flush=True)

    print("=== extracting ground-truth mel via feat_extractor (same fn used at training time) ===", flush=True)
    mel = feat_extractor(speech).to(device)  # (1, 80, T)
    print(f"  mel shape: {tuple(mel.shape)}", flush=True)

    for name, ckpt_path in VARIANTS.items():
        print(f"=== [{name}] loading hifigan from {ckpt_path} ===", flush=True)
        hift = configs["hift"]
        hift.load_state_dict(clean_hifigan_state_dict(ckpt_path), strict=True)
        hift.to(device).eval()

        with torch.no_grad():
            tts_speech, tts_source = hift.inference(speech_feat=mel)

        out_path = f"{OUT_DIR}/{name}.wav"
        torchaudio.save(out_path, tts_speech.cpu(), 24000)
        peak = tts_speech.abs().max().item()
        dur = tts_speech.shape[-1] / 24000
        print(f"  -> {out_path} ({dur:.2f}s, peak={peak:.4f})", flush=True)

    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
