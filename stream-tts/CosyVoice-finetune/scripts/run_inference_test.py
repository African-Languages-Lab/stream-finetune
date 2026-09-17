"""
Sanity-check inference for the two most-trained CosyVoice3 regimes: en-UG (all 3 stages
genuinely trained) and af-ZA (llm+flow trained, no hifigan yet -> borrows the base pretrained
vocoder). Runs zero-shot voice cloning against real training-data reference clips (verified
ground-truth text, not ASR-predicted) to check (a) basic cloning fidelity via resynthesis,
(b) distinct voice identity across different reference clips, (c) genuine text generation via
a cross-text case (clone voice A, but have it say voice B's sentence).
"""
import sys
import torchaudio

sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice")
sys.path.insert(0, "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-tts/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import CosyVoice3
from cosyvoice.utils.file_utils import load_wav

OUT_DIR = "/leonardo_scratch/large/userexternal/atsado00/cosyvoice_ft/inference_test"

GETTYSBURG = ("Four score and seven years ago our fathers brought forth on this continent a new "
              "nation, conceived in liberty, and dedicated to the proposition that all men are "
              "created equal.")
ENGLISH_REF_WAV = "/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-trial/reference-samples/english_ref.wav"

CASES = {
    "en-UG": [
        # (case_name, prompt_wav, prompt_text, tts_text)
        ("voiceA_resynth", "salt/1.wav", "Eggplants always grow best under warm conditions.",
         "Eggplants always grow best under warm conditions."),
        ("voiceB_resynth", "salt/2.wav", "Farmland is sometimes a challenge to farmers.",
         "Farmland is sometimes a challenge to farmers."),
        ("voiceA_newtext", "salt/1.wav", "Eggplants always grow best under warm conditions.",
         "Farmland is sometimes a challenge to farmers."),
        # englishRef: a voice the model has never seen in training at all -- true zero-shot
        # cloning test, not just resynthesizing in-domain data it may have memorized.
        ("englishRef_resynth", "EXTERNAL", GETTYSBURG, GETTYSBURG),
        ("englishRef_newtext", "EXTERNAL", GETTYSBURG,
         "Eggplants always grow best under warm conditions."),
    ],
    "af-ZA": [
        ("voiceA_resynth", "ast_afrikaans_afrikaans", None, None),  # filled below
        ("voiceB_resynth", "fleurs_afrikaans", None, None),
        ("voiceA_newtext", "ast_afrikaans_afrikaans", None, None),
    ],
}

AF_ZA_WAVS = {
    "voiceA": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/afrikaans/audio/common_voice_afrikaans/cv_afr_common_voice_af_38087445.wav",
    "voiceB": "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/afrikaans/audio/fleurs_afrikaans/fleurs_afr_train_10010386886416577453.wav",
}
AF_ZA_TEXT = {
    "voiceA": "Die afsluiting van die proses en die aanstelling van diensverskaffers is 'n voortdurende proses.",
    "voiceB": "Families met kinders is nie 'n algemene verskynsel nie, maar party koshuise laat hulle in privaat kamers toe.",
}

EN_UG_WAV_DIR = "/leonardo_scratch/large/userexternal/atsado00/all_lab_data_prep/speech/speech_out/english/audio"


def run_lang(lang, model_dir, cases):
    print(f"\n=== loading {lang} from {model_dir} ===")
    model = CosyVoice3(model_dir, fp16=False)
    for case_name, prompt_wav, prompt_text, tts_text in cases:
        if lang == "af-ZA":
            voice_key = "voiceA" if "voiceA" in case_name or case_name == "voiceA_newtext" else "voiceB"
            prompt_wav_path = AF_ZA_WAVS[voice_key]
            prompt_text = AF_ZA_TEXT[voice_key]
            tts_text = AF_ZA_TEXT["voiceB"] if case_name == "voiceA_newtext" else AF_ZA_TEXT[voice_key]
        elif prompt_wav == "EXTERNAL":
            prompt_wav_path = ENGLISH_REF_WAV
        else:
            prompt_wav_path = f"{EN_UG_WAV_DIR}/{prompt_wav}"

        # inference_zero_shot's prompt_wav must be a file path -- CosyVoice's own frontend
        # loads and resamples it internally (at 24kHz, not the 16kHz a caller might assume);
        # pre-loading it ourselves and passing a tensor breaks that internal reload.
        #
        # CosyVoice3's LLM (llm.py) hard-asserts the literal special token <|endofprompt|>
        # is present in prompt_text -- it marks where the reference clip's transcript ends and
        # generation begins. Confirmed real requirement (crashed without it): our training
        # data used it via a separate "instruct" field, but inference_zero_shot's plain
        # prompt_text needs it appended directly, not implicit.
        prompt_text = prompt_text + "<|endofprompt|>"
        out_path = f"{OUT_DIR}/{lang}_{case_name}.wav"
        print(f"[{lang}/{case_name}] prompt={prompt_wav_path}")
        print(f"  prompt_text={prompt_text!r}")
        print(f"  tts_text={tts_text!r}")
        results = list(model.inference_zero_shot(tts_text, prompt_text, prompt_wav_path, stream=False))
        assert len(results) == 1
        audio = results[0]["tts_speech"]
        torchaudio.save(out_path, audio, model.sample_rate)
        dur = audio.shape[1] / model.sample_rate
        peak = audio.abs().max().item()
        print(f"  -> {out_path} ({dur:.2f}s, peak amplitude {peak:.4f})")


if __name__ == "__main__":
    import sys as _sys
    only = _sys.argv[1] if len(_sys.argv) > 1 else None
    if only in (None, "en-UG"):
        run_lang("en-UG", f"{OUT_DIR}/en-UG", CASES["en-UG"])
    if only in (None, "af-ZA"):
        run_lang("af-ZA", f"{OUT_DIR}/af-ZA", CASES["af-ZA"])
    print("\n=== all inference cases complete ===")
