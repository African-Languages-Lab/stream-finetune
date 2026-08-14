"""
Transcribes the 6 reference samples in stream-trial/reference-samples/ using each
language's own fine-tuned individual Nemotron ASR model (not the base checkpoint).
"""
import json
import tempfile
from pathlib import Path

import soundfile as sf
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

REF_DIR = Path("/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-trial/reference-samples")
CKPT_ROOT = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/individual")
OUT_PATH = REF_DIR / "transcripts_individual_models.txt"
OUT_JSON = REF_DIR / "transcripts_individual_models.json"

# (reference wav, target_lang code, path to that language's fine-tuned checkpoint)
JOBS = [
    ("english_ref.wav", "en-NG", CKPT_ROOT / "en-NG/individual_en-NG/checkpoints/individual_en-NG--val_wer=0.9592-epoch=1-last.ckpt"),
    ("igbo_ref.wav", "ig-NG", CKPT_ROOT / "ig-NG/individual_ig-NG/checkpoints/individual_ig-NG.nemo"),
    ("hausa_ref.wav", "ha-NG", CKPT_ROOT / "ha-NG/individual_ha-NG/checkpoints/individual_ha-NG.nemo"),
    ("yoruba_ref.wav", "yo-NG", CKPT_ROOT / "yo-NG/individual_yo-NG/checkpoints/individual_yo-NG.nemo"),
    ("twi_ref.wav", "tw-GH", CKPT_ROOT / "tw-GH/individual_tw-GH/checkpoints/individual_tw-GH.nemo"),
    ("ewe_ref.wav", "ee-GH", CKPT_ROOT / "ee-GH/individual_ee-GH/checkpoints/individual_ee-GH.nemo"),
]


def load_model(ckpt_path):
    if str(ckpt_path).endswith(".nemo"):
        return nemo_asr.models.ASRModel.restore_from(str(ckpt_path), map_location="cuda")
    # .ckpt (unlike .nemo) has no embedded target-class info, so load_from_checkpoint must be
    # called on the concrete class directly -- the abstract ASRModel base can't be instantiated.
    return EncDecRNNTBPEModelWithPrompt.load_from_checkpoint(str(ckpt_path), map_location="cuda")


def main():
    results = []
    for wav_name, target_lang, ckpt_path in JOBS:
        wav_path = REF_DIR / wav_name
        print(f"=== {wav_name} ({target_lang}) using {ckpt_path.name} ===")
        if not ckpt_path.exists():
            print(f"  MISSING checkpoint: {ckpt_path}")
            results.append({"file": wav_name, "target_lang": target_lang, "transcript": None,
                             "error": f"checkpoint not found: {ckpt_path}"})
            continue
        model = load_model(ckpt_path)
        model.eval()

        # model.transcribe(audio=[path], target_lang=...) doesn't work for this
        # prompt-conditioned model: the raw-audio-path convenience path builds cuts with no
        # supervision, so the dataset's cut.supervisions[0].language lookup gets None and
        # crashes (default_lang is set in _setup_transcribe_dataloader's dl_config but is one
        # of the keys get_lhotse_dataloader_from_config explicitly ignores -- confirmed via
        # its own startup warning). Build a one-row manifest with target_lang set instead,
        # matching exactly the input_cfg/tags.target_lang shape training already proves
        # works, and feed transcribe() a real dataloader instead of raw paths.
        duration = sf.info(str(wav_path)).duration
        with tempfile.TemporaryDirectory() as td:
            manifest_path = Path(td) / "manifest.json"
            with open(manifest_path, "w") as f:
                f.write(json.dumps({
                    "audio_filepath": str(wav_path),
                    "text": "",
                    "duration": duration,
                    "target_lang": target_lang,
                }) + "\n")
            dl_config = DictConfig({
                "manifest_filepath": None,
                "input_cfg": [{
                    "type": "nemo",
                    "manifest_filepath": str(manifest_path),
                    "weight": 1.0,
                    "tags": {"target_lang": target_lang, "prompt_mode": "unified"},
                }],
                "sample_rate": model.preprocessor._sample_rate,
                "batch_size": 1,
                "shuffle": False,
                "use_lhotse": True,
                "use_bucketing": False,
                "drop_last": False,
                "num_workers": 0,
                "pin_memory": True,
                "channel_selector": 0,
                "prompt_dictionary": model.cfg.model_defaults.get("prompt_dictionary"),
                "num_prompts": model.cfg.model_defaults.get("num_prompts", 128),
                "subsampling_factor": model.cfg.get("subsampling_factor", 8),
                "lang_field": "target_lang",
                "prompt_mode_field": "prompt_mode",
                "default_prompt_mode": "unified",
            })
            dataloader = model._setup_dataloader_from_config(config=dl_config)
            hyps = model.transcribe(audio=dataloader, verbose=False)
        text = hyps[0].text if hasattr(hyps[0], "text") else str(hyps[0])
        print(f"  -> {text}")
        results.append({"file": wav_name, "target_lang": target_lang,
                         "checkpoint": str(ckpt_path), "transcript": text})
        del model

    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(f"[{r['target_lang']}] {r['file']}\n")
            f.write(f"  {r.get('transcript') or ('ERROR: ' + r.get('error', 'unknown'))}\n\n")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_PATH} and {OUT_JSON}")


if __name__ == "__main__":
    main()
