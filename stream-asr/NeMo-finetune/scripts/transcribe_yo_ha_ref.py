"""
Transcribes yoruba_ref.wav and hausa_ref.wav with each language's own highest individual
Nemotron checkpoint -- needed as a real (non-empty) prompt_text for CosyVoice3 zero-shot
synthesis, since an empty prompt_text reproducibly crashes CosyVoice3's llm (mix_ratio-based
text/speech interleaving apparently can't handle it). Not claimed as ground truth, just a
working prompt_text to unblock the llm-vs-llm comparison.
"""
import json
from pathlib import Path

import soundfile as sf
import nemo.collections.asr as nemo_asr
from omegaconf import DictConfig
from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

REF_DIR = Path("/leonardo/home/userexternal/atsado00/all_lab_workspace/002/all_data/stream/stream-trial/reference-samples")
CKPT_ROOT = Path("/leonardo_scratch/large/userexternal/atsado00/nemotron_ft/checkpoints/individual")

CASES = [
    ("yoruba_ref.wav", "yo-NG"),
    ("hausa_ref.wav", "ha-NG"),
]


def latest_checkpoint(target_lang):
    import re
    d = CKPT_ROOT / target_lang / f"individual_{target_lang}" / "checkpoints"
    unfinished_stems = {p.name[: -len("-unfinished")] for p in d.glob("*-unfinished")}
    ckpt_candidates = [p for p in d.glob("*.ckpt") if p.name not in unfinished_stems]

    def epoch_key(p):
        m = re.search(r"epoch=(\d+)", p.name)
        return int(m.group(1)) if m else -1

    latest_ckpt = sorted(ckpt_candidates, key=epoch_key)[-1] if ckpt_candidates else None
    nemo_files = sorted(d.glob("*.nemo"))
    latest_nemo = nemo_files[-1] if nemo_files else None
    if latest_nemo and latest_ckpt:
        return latest_nemo if latest_nemo.stat().st_mtime >= latest_ckpt.stat().st_mtime else latest_ckpt
    return latest_nemo or latest_ckpt


def load_model(ckpt_path):
    if str(ckpt_path).endswith(".nemo"):
        return nemo_asr.models.ASRModel.restore_from(str(ckpt_path), map_location="cuda")
    return EncDecRNNTBPEModelWithPrompt.load_from_checkpoint(str(ckpt_path), map_location="cuda")


def transcribe(model, wav_path, target_lang):
    import tempfile
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
                "tags": {"target_lang": target_lang, "prompt_mode": "langID"},
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
            "default_prompt_mode": "langID",
        })
        dataloader = model._setup_dataloader_from_config(config=dl_config)
        hyps = model.transcribe(audio=dataloader, verbose=False)
    return hyps[0].text if hasattr(hyps[0], "text") else str(hyps[0])


def main():
    results = {}
    for wav_name, target_lang in CASES:
        ckpt = latest_checkpoint(target_lang)
        print(f"=== {wav_name} ({target_lang}) using {ckpt} ===", flush=True)
        model = load_model(ckpt)
        model.eval()
        text = transcribe(model, REF_DIR / wav_name, target_lang)
        print(f"  -> {text}", flush=True)
        results[wav_name] = text
        del model

    out_path = REF_DIR / "transcripts_yo_ha_ref.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
