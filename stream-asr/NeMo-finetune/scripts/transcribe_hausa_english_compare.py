"""
Transcribes hausa_ref.wav and english_ref.wav using each language's individual and
clustered models (highest/final checkpoint for each -- all four regimes involved have
completed training, so this is each one's final .nemo export).
"""
import json
import tempfile
from pathlib import Path

import soundfile as sf
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig

REF_DIR = Path("/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-trial/reference-samples")
OUT_PATH = REF_DIR / "transcripts_hausa_english_individual_vs_cluster.txt"
OUT_JSON = REF_DIR / "transcripts_hausa_english_individual_vs_cluster.json"

CKPT_ROOT = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints")

JOBS = [
    ("hausa_ref.wav", "individual", "ha-NG", CKPT_ROOT / "individual/ha-NG/individual_ha-NG/checkpoints/individual_ha-NG.nemo"),
    ("hausa_ref.wav", "cluster", "ha-NG", CKPT_ROOT / "clusters/afroasiatic/cluster_afroasiatic/checkpoints/cluster_afroasiatic.nemo"),
    ("english_ref.wav", "individual", "en-NG", CKPT_ROOT / "individual/en-NG/individual_en-NG/checkpoints/individual_en-NG.nemo"),
    ("english_ref.wav", "cluster", "en-NG", CKPT_ROOT / "clusters/other_languages/cluster_other_languages/checkpoints/cluster_other_languages.nemo"),
]


def main():
    results = []
    for wav_name, kind, target_lang, ckpt_path in JOBS:
        wav_path = REF_DIR / wav_name
        duration = sf.info(str(wav_path)).duration
        print(f"=== {wav_name} / {kind} model ({ckpt_path.name}), target_lang={target_lang} ===")
        model = nemo_asr.models.ASRModel.restore_from(str(ckpt_path), map_location="cuda")
        model.eval()
        prompt_dictionary = model.cfg.train_ds.get("prompt_dictionary")

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
                "prompt_dictionary": prompt_dictionary,
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
        results.append({"file": wav_name, "kind": kind, "target_lang": target_lang,
                         "checkpoint": str(ckpt_path), "transcript": text})
        del model

    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(f"[{r['file']} / {r['kind']}] {r['checkpoint']}\n  {r['transcript']}\n\n")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_PATH} and {OUT_JSON}")


if __name__ == "__main__":
    main()
