"""
Regenerates just the Igbo voice_clone case with a fresh reference clip -- the original
(ibo_1436.wav) collapsed to near-silence (0.48s) with ig-NG's best llm+flow despite the same
checkpoint working fine on two other prompts, so the prompt itself was the problem, not the
checkpoint. Reuses the already-built ft-llm/ft-flow bundle directly.
"""
import torchaudio
import sys

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

BUNDLE_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_igbo_drive"
OUT_PATH = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_igbo_drive/voice_clone.wav"

TARGET_TEXT = "Ndị mmadụ ka na-agbaso usoro ọzọ iji chekwaba ahụ ike ha n'oge ọrịa na-agbasa."
PROMPT_WAV = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/Waxal_NLP_Igbo/ibo_1266.wav"
PROMPT_TEXT = "Emeka nyere goolu nke ise n'ime ọtụtụ goolu enyere na egwuregwu ahụ.<|endofprompt|>"

model = CosyVoice3(BUNDLE_DIR, fp16=False)
for attempt in range(5):
    results = list(model.inference_zero_shot(TARGET_TEXT, PROMPT_TEXT, PROMPT_WAV, stream=False))
    audio = results[0]["tts_speech"]
    dur = audio.shape[1] / model.sample_rate
    peak = audio.abs().max().item()
    print(f"attempt {attempt + 1}: {dur:.2f}s peak={peak:.4f}", flush=True)
    if dur >= 3.0:
        torchaudio.save(OUT_PATH, audio, model.sample_rate)
        print(f"SAVED: {OUT_PATH} ({dur:.2f}s, peak={peak:.4f})", flush=True)
        break
else:
    print("all attempts produced short/degenerate output, not overwriting", flush=True)
