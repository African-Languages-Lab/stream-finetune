"""
Transcribes igbo_ref.wav using both the individual ig-NG model and the clustered
west_africa_niger_congo model (which includes ig-NG), for comparison.
"""
import json
import tempfile
from pathlib import Path

import soundfile as sf
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

REF_DIR = Path("/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-trial/reference-samples")
WAV_PATH = REF_DIR / "igbo_ref.wav"
OUT_PATH = REF_DIR / "transcripts_igbo_individual_vs_cluster_vs_combined.txt"
OUT_JSON = REF_DIR / "transcripts_igbo_individual_vs_cluster_vs_combined.json"

JOBS = [
    ("individual", "ig-NG", Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/individual/ig-NG/individual_ig-NG/checkpoints/individual_ig-NG.nemo")),
    ("cluster", "ig-NG", Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/clusters/west_africa_niger_congo/cluster_west_africa_niger_congo/checkpoints/cluster_west_africa_niger_congo.nemo")),
    # still training (~16% through its 100k-step target as of this run) -- latest self-updating
    # "-last" checkpoint, expect low quality output given how early this is.
    ("combined", "ig-NG", Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/combined/nemotron_combined_multilingual/checkpoints/nemotron_combined_multilingual--val_wer=0.9950-epoch=0-last.ckpt")),
]


def main():
    duration = sf.info(str(WAV_PATH)).duration
    results = []
    for kind, target_lang, ckpt_path in JOBS:
        print(f"=== {kind} model ({ckpt_path.name}), target_lang={target_lang} ===")
        if str(ckpt_path).endswith(".nemo"):
            model = nemo_asr.models.ASRModel.restore_from(str(ckpt_path), map_location="cuda")
        else:
            # raw .ckpt (still-training combined) has no embedded target-class info, so
            # load_from_checkpoint must be called on the concrete class directly.
            model = EncDecRNNTBPEModelWithPrompt.load_from_checkpoint(str(ckpt_path), map_location="cuda")
        model.eval()

        # prompt_dictionary must come from train_ds's copy of the config, not
        # model_defaults -- confirmed via a real runtime failure that model_defaults holds a
        # different (base-model, non-expanded) dictionary lacking our target_lang codes,
        # while train_ds's own printed config (logged at model init) does contain them.
        prompt_dictionary = model.cfg.train_ds.get("prompt_dictionary")

        with tempfile.TemporaryDirectory() as td:
            manifest_path = Path(td) / "manifest.json"
            with open(manifest_path, "w") as f:
                f.write(json.dumps({
                    "audio_filepath": str(WAV_PATH),
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
        results.append({"kind": kind, "target_lang": target_lang, "checkpoint": str(ckpt_path), "transcript": text})
        del model

    with open(OUT_PATH, "w") as f:
        for r in results:
            f.write(f"[{r['kind']}] {r['checkpoint']}\n  {r['transcript']}\n\n")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_PATH} and {OUT_JSON}")


if __name__ == "__main__":
    main()
