"""
Igbo: 4 combos, all with best (pre-divergence) llm and original vocoder.
  - cloning: a real Igbo reference clip (specific to this one request) x {best flow, original flow}
  - house_voice: a DIFFERENT fixed reference clip (the practical stand-in for "no cloning" --
    flow architecturally requires some real voice conditioning, confirmed by testing a truly
    empty prompt directly, so a constant default clip is the closest real equivalent) x
    {best flow, original flow}
Same target text in all four for a fair, direct comparison.
"""
import os
import sys

import torch
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3

PRETRAINED_DIR = "/leonardo_work/AIH4A_udutech/cosyvoice_eval/Fun-CosyVoice3-0.5B"
WORK_ROOT = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/bundle_igbo_combo"
OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_igbo_combo"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_TEXT = "Olusegun Obasanjo chịrị Naijiria ugboro abụọ, dịka onye agha na onye ọchịchị onye kwuo uche ya."

CLONE_WAV = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/Waxal_NLP_Igbo/ibo_1436.wav"
CLONE_TEXT = "Onye nkụzi ụmụaka ahụ sere otu okirikiri na-acha anụnụ anụnụ n'etiti rombusu abụọ na-acha edo edo.<|endofprompt|>"

HOUSE_WAV = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/igbo/audio/Waxal_NLP_Igbo/ibo_767.wav"
HOUSE_TEXT = "Okpomọkụ dị elu maka taa ga-abụ selsịọsụ iri atọ na otu mana obere mmiri ga-ezo ma ọ na eweta aka n'elekere mbụ nke ụtụtụ.<|endofprompt|>"

BEST_LLM = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints/individual/individual_ig-NG/llm/epoch_1_whole.pt"
BEST_FLOW = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/checkpoints_archived_20260818/individual/individual_ig-NG/flow/epoch_32_step_276000.pt"

CASES = [
    ("cloning_best_flow", CLONE_WAV, CLONE_TEXT, BEST_FLOW),
    ("cloning_original_flow", CLONE_WAV, CLONE_TEXT, None),
    ("house_voice_best_flow", HOUSE_WAV, HOUSE_TEXT, BEST_FLOW),
    ("house_voice_original_flow", HOUSE_WAV, HOUSE_TEXT, None),
]


def clean(src, dst):
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in sd.items() if torch.is_tensor(v)}, dst)


def main():
    for name, prompt_wav, prompt_text, flow_ckpt in CASES:
        bundle_dir = f"{WORK_ROOT}/{name}"
        os.makedirs(bundle_dir, exist_ok=True)
        for asset in ["cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx", "CosyVoice-BlankEN", "hift.pt"]:
            dst = f"{bundle_dir}/{asset}"
            if not os.path.exists(dst):
                os.symlink(f"{PRETRAINED_DIR}/{asset}", dst)

        print(f"=== [{name}] preparing llm (best) ===", flush=True)
        clean(BEST_LLM, f"{bundle_dir}/llm.pt")

        print(f"=== [{name}] preparing flow ({'best' if flow_ckpt else 'original'}) ===", flush=True)
        if flow_ckpt is not None:
            clean(flow_ckpt, f"{bundle_dir}/flow.pt")
        else:
            dst = f"{bundle_dir}/flow.pt"
            if not os.path.exists(dst):
                os.symlink(f"{PRETRAINED_DIR}/flow.pt", dst)

        print(f"=== [{name}] loading bundle ===", flush=True)
        model = CosyVoice3(bundle_dir, fp16=False)

        print(f"=== [{name}] zero-shot synthesis ===", flush=True)
        last_err = None
        for attempt in range(4):
            try:
                results = list(model.inference_zero_shot(TARGET_TEXT, prompt_text, prompt_wav, stream=False))
                audio = results[0]["tts_speech"]
                out_path = f"{OUT_DIR}/{name}.wav"
                torchaudio.save(out_path, audio, model.sample_rate)
                dur = audio.shape[1] / model.sample_rate
                peak = audio.abs().max().item()
                print(f"=== [{name}] SAVED: {out_path} ({dur:.2f}s, peak={peak:.4f}) ===", flush=True)
                last_err = None
                break
            except RuntimeError as e:
                last_err = e
                print(f"=== [{name}] attempt {attempt + 1} failed ({e}), retrying ===", flush=True)
        if last_err is not None:
            print(f"=== [{name}] FAILED after retries: {last_err} ===", flush=True)
        del model

    print("\nall combos done", flush=True)


if __name__ == "__main__":
    main()
